// Per-layer LRU / reuse-distance profile of MoE expert routing traces.
//
// The hot/cold profile (profile_hot_experts.cpp + summarize_hot_experts.py) answers
// "how often is each expert selected". That is a *frequency* statistic, and the
// viewer's coverage curve is computed by statically pinning the top-k experts — it
// is NOT what the engine's cache actually does.
//
// The engine's per-layer decode window (LLAMA_MOE_DECODE_WINDOW / LLAMA_MOE_LAYER_WINDOWS)
// is a fully-associative LRU over experts, capacity c_i (see src/llama-moe-slot.cpp:
// victim = smallest slot_age, skipping experts requested this step and experts the
// lookahead prefetch predicted). Its miss rate depends on *reuse distance*, not on
// frequency, so it has to be measured by replaying the routing trace.
//
// This tool replays one phase of a `*-routes.jsonl` trace per layer, exactly in the
// engine's access order (one step == one token == 8 distinct experts), and reports:
//
//   lru<c>  expert loads per 1000 routing references, batch-LRU of capacity <c>
//   st<c>   same for statically pinning the train-top-<c> experts (frequency view)
//   hot32   share of LRU<c=32> misses that land on a train-top-8 expert
//   sd50/90/99  reuse distance (distinct experts between two uses), in references
//   compulsory  first-touch share of all references (upper bound hit rate is 1-this)
//
// Faithful-to-engine details: the 8 experts of one step are requested together and
// never evict each other; only the least-recently-used *other* resident is evicted.
// Ignored (documented in docs/hot-cold-expert/LOCALITY.md): prefill warming the
// window (real device: lower miss), lookahead-prefetch sharing slots (higher miss).
//
// Not a benchmark: replay only, no inference.
//
// usage: llama-moe-locality TRAIN.jsonl EVAL.jsonl NLAYERS NEXPERTS {d|p} NAME [CAPS.csv]
//   CAPS.csv: optional per-layer capacity "layer,slots" lines; adds lru_cfg/st_cfg
//             columns (e.g. the suggested window of docs/hot-cold-expert/).
// stdout: CSV, one row per layer.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static int NLAY = 0, NEXP = 0;
static const int NSTEP = 8;   // top-k experts requested per token

// ---------------------------------------------------------------- trace reading
struct mapped_file {
    const char * data = nullptr;
    size_t size = 0;
};

static bool map_file(const char * path, mapped_file & f) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return false; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror(path); close(fd); return false; }
    f.size = (size_t) st.st_size;
    f.data = (const char *) mmap(nullptr, f.size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (f.data == MAP_FAILED) { perror("mmap"); return false; }
    return true;
}

struct record {
    char phase = '?';
    int layer = -1;
    std::vector<int32_t> experts;   // flattened: 8 ids per token
};

// The collector writes one JSON object per line with sorted keys:
//   {"experts":[[a,b,...],[...]],"layer":17,"phase":"decode","sample":"..."}
static bool parse_record(const char * begin, const char * end, record & out) {
    const char * phase = (const char *) memmem(begin, (size_t) (end - begin), "\"phase\":\"", 9);
    if (!phase) return false;
    out.phase = phase[9];

    const char * layer = (const char *) memmem(begin, (size_t) (end - begin), "\"layer\":", 8);
    if (!layer) return false;
    out.layer = atoi(layer + 8);

    const char * experts = (const char *) memmem(begin, (size_t) (end - begin), "\"experts\":[[", 12);
    if (!experts) return false;

    out.experts.clear();
    const char * p = experts + 12;
    int depth = 2;
    while (p < end && depth > 0) {
        const char c = *p;
        if (c >= '0' && c <= '9') {
            int value = 0;
            while (p < end && *p >= '0' && *p <= '9') { value = value * 10 + (*p - '0'); ++p; }
            out.experts.push_back(value);
            continue;
        }
        if (c == '[') ++depth;
        else if (c == ']') --depth;
        ++p;
    }
    return !out.experts.empty();
}

// Walk every record of a trace, calling fn(record).
template <typename Fn>
static void for_each_record(const mapped_file & f, Fn && fn) {
    record r;
    const char * cur = f.data;
    const char * end = f.data + f.size;
    while (cur < end) {
        const char * nl = (const char *) memchr(cur, '\n', (size_t) (end - cur));
        if (!nl) nl = end;
        if (parse_record(cur, nl, r)) fn(r);
        cur = nl + 1;
    }
}

// ---------------------------------------------------------------- batch LRU
// Capacity-limited LRU over expert ids. Cost per step is O(top-k) because only the
// 8 requested experts ever move.
class lru_cache {
public:
    void init(int n_experts) {
        prev_.assign(n_experts, -1);
        next_.assign(n_experts, -1);
        resident_.assign(n_experts, 0);
        head_ = tail_ = -1;
        size_ = 0;
    }

    // One decode step: 8 experts requested simultaneously.
    // Returns the number of experts that had to be loaded.
    int step(const int32_t * experts, int n, int capacity, const std::vector<char> * hot, uint64_t * hot_misses) {
        int loaded = 0;
        for (int i = 0; i < n; ++i) {
            const int32_t e = experts[i];
            if (resident_[e]) continue;
            ++loaded;
            if (hot && hot_misses && (*hot)[e]) ++(*hot_misses);
        }
        // Make room without evicting an expert this step needs.
        while (size_ + loaded > capacity) {
            int victim = head_;
            while (victim >= 0 && contains(experts, n, victim)) victim = next_[victim];
            if (victim < 0) break;   // capacity < n: cannot fit the step, give up freeing
            unlink(victim);
            resident_[victim] = 0;
            --size_;
        }
        for (int i = 0; i < n; ++i) {
            const int32_t e = experts[i];
            if (resident_[e]) unlink(e);
            else { resident_[e] = 1; ++size_; }
            push_back(e);
        }
        return loaded;
    }

private:
    static bool contains(const int32_t * experts, int n, int e) {
        for (int i = 0; i < n; ++i) if (experts[i] == e) return true;
        return false;
    }
    void unlink(int e) {
        const int p = prev_[e], q = next_[e];
        if (p >= 0) next_[p] = q; else head_ = q;
        if (q >= 0) prev_[q] = p; else tail_ = p;
        prev_[e] = next_[e] = -1;
    }
    void push_back(int e) {   // most recently used
        prev_[e] = tail_;
        next_[e] = -1;
        if (tail_ >= 0) next_[tail_] = e; else head_ = e;
        tail_ = e;
    }

    std::vector<int32_t> prev_, next_;
    std::vector<char> resident_;
    int head_ = -1, tail_ = -1, size_ = 0;
};

// Replay `refs` (flattened 8-per-step) at one capacity; returns total loads.
static uint64_t replay(const std::vector<int32_t> & refs, int capacity,
                       const std::vector<char> * hot, uint64_t * hot_misses) {
    lru_cache cache;
    cache.init(NEXP);
    uint64_t loaded = 0;
    const size_t steps = refs.size() / NSTEP;
    for (size_t s = 0; s < steps; ++s) {
        loaded += (uint64_t) cache.step(&refs[s * NSTEP], NSTEP, capacity, hot, hot_misses);
    }
    return loaded;
}

// ---------------------------------------------------------------- reuse distance
// Number of *distinct* experts touched between two consecutive uses of one expert.
// This is a locality indicator, not an exact single-reference hit test for the
// batch LRU: all experts requested in the same step are protected from eviction,
// so a distance >= c can still hit. Replay the full step sequence for miss counts.
class fenwick {
public:
    void init(int n) { n_ = n; t_.assign((size_t) n + 2, 0); }
    void add(int i, int v) { for (++i; i <= n_ + 1; i += i & (-i)) t_[(size_t) i] += v; }
    int prefix(int i) const {   // count of marks at positions [0, i)
        int s = 0;
        for (; i > 0; i -= i & (-i)) s += t_[(size_t) i];
        return s;
    }
private:
    std::vector<int32_t> t_;
    int n_ = 0;
};

static void reuse_histogram(const std::vector<int32_t> & refs, std::vector<uint64_t> & hist,
                            uint64_t & compulsory) {
    fenwick bit;
    bit.init((int) refs.size() + 1);
    std::vector<int32_t> last((size_t) NEXP, -1);
    hist.assign(refs.size() + 2, 0);
    compulsory = 0;
    for (size_t i = 0; i < refs.size(); ++i) {
        const int32_t e = refs[i];
        const int32_t prev = last[(size_t) e];
        if (prev < 0) {
            ++compulsory;
        } else {
            const int distance = bit.prefix((int) i) - bit.prefix(prev + 1);
            ++hist[(size_t) distance];
            bit.add(prev, -1);
        }
        bit.add((int) i, +1);
        last[(size_t) e] = (int32_t) i;
    }
}

static double quantile(const std::vector<uint64_t> & hist, double q, uint64_t n) {
    const uint64_t target = (uint64_t) (q * (double) n);
    uint64_t acc = 0;
    for (size_t i = 0; i < hist.size(); ++i) {
        acc += hist[i];
        if (acc >= target) return (double) i;
    }
    return (double) hist.size();
}

// ---------------------------------------------------------------- main
static const int CAPACITIES[] = { 8, 16, 24, 32, 48, 64, 96, 128 };
static const int N_CAPACITIES = (int) (sizeof(CAPACITIES) / sizeof(CAPACITIES[0]));

int main(int argc, char ** argv) {
    const bool have_caps = (argc == 8);
    if (argc != 7 && argc != 8) {
        fprintf(stderr,
                "usage: %s TRAIN.jsonl EVAL.jsonl NLAYERS NEXPERTS {d|p} NAME [CAPS.csv]\n"
                "  NLAYERS/NEXPERTS must match the model (summarize_hot_experts.py has the table)\n",
                argv[0]);
        return 2;
    }
    const char * train_path = argv[1];
    const char * eval_path  = argv[2];
    NLAY = atoi(argv[3]);
    NEXP = atoi(argv[4]);
    const char phase = argv[5][0];
    const char * name = argv[6];
    if (NLAY <= 0 || NEXP <= 0 || NEXP > 1024) {
        fprintf(stderr, "bad NLAYERS/NEXPERTS\n");
        return 2;
    }

    std::vector<int> cfg((size_t) NLAY, 0);
    if (have_caps) {
        FILE * fp = fopen(argv[7], "r");
        if (!fp) { perror(argv[7]); return 4; }
        int layer = 0, slots = 0;
        while (fscanf(fp, "%d,%d", &layer, &slots) == 2) {
            if (layer >= 0 && layer < NLAY) cfg[(size_t) layer] = slots;
        }
        fclose(fp);
    }

    mapped_file train, eval;
    if (!map_file(train_path, train)) return 3;
    if (!map_file(eval_path, eval)) return 3;

    // Pass 1: training-set frequency per layer (the ranking the hot/cold profile uses).
    std::vector<std::vector<uint32_t>> count((size_t) NLAY, std::vector<uint32_t>((size_t) NEXP, 0));
    uint64_t train_steps = 0;
    for_each_record(train, [&](record & r) {
        if (r.phase != phase || r.layer < 0 || r.layer >= NLAY) return;
        auto & c = count[(size_t) r.layer];
        for (int32_t e : r.experts) if (e >= 0 && e < NEXP) ++c[(size_t) e];
        train_steps += r.experts.size() / NSTEP;
    });

    // Pass 2: keep the eval stream per layer, plus its frequency (for pin coverage).
    std::vector<std::vector<int32_t>> refs((size_t) NLAY);
    std::vector<std::vector<uint32_t>> eval_count((size_t) NLAY, std::vector<uint32_t>((size_t) NEXP, 0));
    uint64_t eval_steps = 0;
    for_each_record(eval, [&](record & r) {
        if (r.phase != phase || r.layer < 0 || r.layer >= NLAY) return;
        auto & v = refs[(size_t) r.layer];
        auto & c = eval_count[(size_t) r.layer];
        for (int32_t e : r.experts) {
            v.push_back(e);
            if (e >= 0 && e < NEXP) ++c[(size_t) e];
        }
        eval_steps += r.experts.size() / NSTEP;
    });

    fprintf(stderr, "[%s] phase=%c train_steps=%llu eval_steps=%llu\n", name, phase,
            (unsigned long long) train_steps, (unsigned long long) eval_steps);

    printf("model,layer,steps,refs,distinct");
    for (int i = 0; i < N_CAPACITIES; ++i) printf(",lru%d", CAPACITIES[i]);
    for (int i = 0; i < N_CAPACITIES; ++i) printf(",st%d", CAPACITIES[i]);
    printf(",hot32,sd50,sd90,sd99,compulsory,top8share,n95");
    if (have_caps) printf(",cap,lru_cfg,st_cfg");
    printf("\n");

    std::vector<int32_t> ranking((size_t) NEXP);
    std::vector<char> hot((size_t) NEXP, 0);
    std::vector<uint64_t> hist;
    for (int layer = 0; layer < NLAY; ++layer) {
        const size_t n_refs = refs[(size_t) layer].size();
        if (n_refs < (size_t) NSTEP) continue;
        const size_t n_steps = n_refs / NSTEP;

        for (int e = 0; e < NEXP; ++e) ranking[(size_t) e] = e;
        std::sort(ranking.begin(), ranking.end(), [&](int a, int b) {
            const uint32_t ca = count[(size_t) layer][(size_t) a], cb = count[(size_t) layer][(size_t) b];
            return ca != cb ? ca > cb : a < b;
        });
        uint64_t train_total = 0;
        for (uint32_t c : count[(size_t) layer]) train_total += c;
        for (int e = 0; e < NEXP; ++e) hot[(size_t) e] = 0;
        for (int k = 0; k < NSTEP && k < NEXP; ++k) hot[(size_t) ranking[(size_t) k]] = 1;

        int n95 = NEXP;
        {
            uint64_t acc = 0;
            const uint64_t target = (uint64_t) (0.95 * (double) train_total);
            for (int k = 0; k < NEXP; ++k) {
                acc += count[(size_t) layer][(size_t) ranking[(size_t) k]];
                if (acc >= target) { n95 = k + 1; break; }
            }
        }

        uint64_t eval_total = 0;
        int distinct = 0;
        for (int e = 0; e < NEXP; ++e) {
            if (eval_count[(size_t) layer][(size_t) e]) ++distinct;
            eval_total += eval_count[(size_t) layer][(size_t) e];
        }
        const double top8_share = train_total
            ? (double) (count[(size_t) layer][(size_t) ranking[0]] + count[(size_t) layer][(size_t) ranking[1]] +
                        count[(size_t) layer][(size_t) ranking[2]] + count[(size_t) layer][(size_t) ranking[3]] +
                        count[(size_t) layer][(size_t) ranking[4]] + count[(size_t) layer][(size_t) ranking[5]] +
                        count[(size_t) layer][(size_t) ranking[6]] + count[(size_t) layer][(size_t) ranking[7]]) /
                  (double) train_total
            : 0.0;

        printf("%s,%d,%zu,%zu,%d", name, layer, n_steps, n_refs, distinct);

        double hot32 = 0.0;
        for (int i = 0; i < N_CAPACITIES; ++i) {
            uint64_t hot_misses = 0;
            const uint64_t loads = replay(refs[(size_t) layer], CAPACITIES[i], &hot, &hot_misses);
            printf(",%.3f", 1000.0 * (double) loads / (double) n_refs);
            if (CAPACITIES[i] == 32 && loads) hot32 = 100.0 * (double) hot_misses / (double) loads;
        }
        for (int i = 0; i < N_CAPACITIES; ++i) {   // static pinning of the train-top-k
            uint64_t covered = 0;
            for (int k = 0; k < CAPACITIES[i] && k < NEXP; ++k) {
                covered += eval_count[(size_t) layer][(size_t) ranking[(size_t) k]];
            }
            printf(",%.3f", 1000.0 * (double) (eval_total - covered) / (double) eval_total);
        }

        uint64_t compulsory = 0;
        reuse_histogram(refs[(size_t) layer], hist, compulsory);
        uint64_t n_hist = 0;
        for (uint64_t v : hist) n_hist += v;
        printf(",%.1f,%.1f,%.1f,%.1f,%.2f,%.4f,%d", hot32,
               quantile(hist, 0.50, n_hist), quantile(hist, 0.90, n_hist), quantile(hist, 0.99, n_hist),
               100.0 * (double) compulsory / (double) n_refs, top8_share, n95);

        if (have_caps && cfg[(size_t) layer] > 0) {
            const int capacity = cfg[(size_t) layer];
            const uint64_t loads = replay(refs[(size_t) layer], capacity, nullptr, nullptr);
            uint64_t covered = 0;
            for (int k = 0; k < capacity && k < NEXP; ++k) {
                covered += eval_count[(size_t) layer][(size_t) ranking[(size_t) k]];
            }
            printf(",%d,%.3f,%.3f", capacity,
                   1000.0 * (double) loads / (double) n_refs,
                   1000.0 * (double) (eval_total - covered) / (double) eval_total);
        }
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
