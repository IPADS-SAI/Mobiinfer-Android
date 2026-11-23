// window_sweep.cpp - per-layer LRU load curve at fine capacity granularity,
// on both the train and the heldout routing trace.
//
// Purpose: feed a budget allocator. Given a total slot budget, the allocator
// needs load_i(c) for every layer i and every candidate capacity c.
//
// Capacities: 8, 16, ..., 256 (step 8, 32 points).
// Semantics: same as the engine (src/llama-moe-slot.cpp) - a decode step requests
// 8 distinct experts together, same-step experts never evict each other, victim is
// the least recently used among the rest.
//
// Output feeds tools/moe/solve_k.py (budget -> per-layer cache c_i + prefetch K_i).
//
// usage: window_sweep TRAIN.jsonl EVAL.jsonl NLAYERS NEXPERTS {d|p} [per-sample]
//   per-sample: reset the LRU at every sample boundary (matches the held-out
//   accounting in exps/layer-windows-20261122); default is one concatenated
//   stream per layer.
// stdout: CSV  layer,steps,refs,then train8..train256,held8..held256 (loads per 1000 refs)
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static int NLAY = 0, NEXP = 0;
static const int NSTEP = 8;
static const int NCAP = 32;                  // 8 .. 256 step 8

struct mapped { const char * p; size_t n; };
static bool mapf(const char * path, mapped & m) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return false; }
    struct stat st; fstat(fd, &st);
    m.n = (size_t) st.st_size;
    m.p = (const char *) mmap(nullptr, m.n, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    return m.p != MAP_FAILED;
}

struct rec { char phase = '?'; int layer = -1; std::string sample; std::vector<int32_t> e; };
static bool parse(const char * b, const char * e, rec & r) {
    const char * ph = (const char *) memmem(b, e - b, "\"phase\":\"", 9); if (!ph) return false;
    r.phase = ph[9];
    const char * sm = (const char *) memmem(b, e - b, "\"sample\":\"", 10);
    if (sm) {
        const char * q = sm + 10;
        const char * z = (const char *) memchr(q, '"', (size_t) (e - q));
        r.sample.assign(q, z ? (size_t) (z - q) : 0);
    }
    const char * ll = (const char *) memmem(b, e - b, "\"layer\":", 8); if (!ll) return false;
    r.layer = atoi(ll + 8);
    const char * ex = (const char *) memmem(b, e - b, "\"experts\":[[", 12); if (!ex) return false;
    r.e.clear();
    const char * q = ex + 12; int depth = 2;
    while (q < e && depth > 0) {
        const char c = *q;
        if (c >= '0' && c <= '9') { int v = 0; while (q < e && *q >= '0' && *q <= '9') { v = v * 10 + (*q - '0'); ++q; } r.e.push_back(v); continue; }
        if (c == '[') ++depth; else if (c == ']') --depth;
        ++q;
    }
    return !r.e.empty();
}
template <typename Fn> static void each(const mapped & f, Fn && fn) {
    rec r; const char * c = f.p, * end = f.p + f.n;
    while (c < end) {
        const char * nl = (const char *) memchr(c, '\n', (size_t) (end - c)); if (!nl) nl = end;
        if (parse(c, nl, r)) fn(r);
        c = nl + 1;
    }
}

// exact LRU over experts (linked list, O(1) per touched expert)
struct lru {
    std::vector<int32_t> prv, nxt;
    std::vector<char> res;
    int head = -1, tail = -1, size = 0;
    void init(int n) { prv.assign(n, -1); nxt.assign(n, -1); res.assign(n, 0); head = tail = -1; size = 0; }
    void unlink(int e) {
        const int p = prv[e], q = nxt[e];
        if (p >= 0) nxt[p] = q; else head = q;
        if (q >= 0) prv[q] = p; else tail = p;
        prv[e] = nxt[e] = -1;
    }
    void mru(int e) {
        prv[e] = tail; nxt[e] = -1;
        if (tail >= 0) nxt[tail] = e; else head = e;
        tail = e;
    }
};

static uint64_t loads_at(const std::vector<int32_t> & refs, const std::vector<int32_t> & sids,
                         int cap, bool per_sample) {
    lru c; c.init(NEXP);
    uint64_t miss = 0;
    const size_t steps = refs.size() / NSTEP;
    for (size_t t = 0; t < steps; ++t) {
        if (per_sample && t > 0 && sids[t] != sids[t - 1]) c.init(NEXP);
        const int32_t * S = &refs[t * NSTEP];
        int need = 0;
        for (int k = 0; k < NSTEP; ++k) if (!c.res[S[k]]) ++need;
        miss += (uint64_t) need;
        while (c.size + need > cap) {
            int v = c.head;
            while (v >= 0) {
                bool used = false;
                for (int k = 0; k < NSTEP; ++k) if (S[k] == v) { used = true; break; }
                if (!used) break;
                v = c.nxt[v];
            }
            if (v < 0) break;
            c.unlink(v); c.res[v] = 0; --c.size;
        }
        for (int k = 0; k < NSTEP; ++k) {
            const int e = S[k];
            if (c.res[e]) c.unlink(e); else { c.res[e] = 1; ++c.size; }
            c.mru(e);
        }
    }
    return miss;
}

int main(int argc, char ** argv) {
    if (argc != 6 && argc != 7) { fprintf(stderr, "usage: %s TRAIN EVAL NLAYERS NEXPERTS {d|p} [per-sample]\n", argv[0]); return 2; }
    const bool per_sample = argc == 7 && strcmp(argv[6], "per-sample") == 0;
    NLAY = atoi(argv[3]); NEXP = atoi(argv[4]); const char want = argv[5][0];
    mapped T, E;
    if (!mapf(argv[1], T) || !mapf(argv[2], E)) return 3;
    std::vector<std::vector<int32_t>> tr((size_t) NLAY), he((size_t) NLAY);
    std::vector<std::vector<int32_t>> trs((size_t) NLAY), hes((size_t) NLAY);
    std::unordered_map<std::string, int> sid;
    auto sid_of = [&](const std::string & s) {
        auto it = sid.find(s);
        if (it != sid.end()) return it->second;
        const int id = (int) sid.size();
        sid.emplace(s, id);
        return id;
    };
    auto add = [&](const rec & r, std::vector<std::vector<int32_t>> & dst,
                   std::vector<std::vector<int32_t>> & dsts) {
        if (r.phase != want || r.layer < 0 || r.layer >= NLAY) return;
        const int id = sid_of(r.sample);
        auto & refs = dst[(size_t) r.layer];
        auto & sids = dsts[(size_t) r.layer];
        for (size_t k = 0; k + NSTEP <= r.e.size(); k += NSTEP) {
            sids.push_back(id);
            for (int j = 0; j < NSTEP; ++j) refs.push_back(r.e[k + (size_t) j]);
        }
    };
    each(T, [&](rec & r) { add(r, tr, trs); });
    each(E, [&](rec & r) { add(r, he, hes); });

    printf("layer,steps,refs");
    for (int i = 0; i < NCAP; ++i) printf(",train%d", 8 * (i + 1));
    for (int i = 0; i < NCAP; ++i) printf(",held%d", 8 * (i + 1));
    printf("\n");
    for (int L = 0; L < NLAY; ++L) {
        const size_t nref = he[(size_t) L].size();
        if (nref < (size_t) NSTEP) continue;
        printf("%d,%zu,%zu", L, nref / NSTEP, nref);
        for (int i = 0; i < NCAP; ++i) {
            const double v = 1000.0 * (double) loads_at(tr[(size_t) L], trs[(size_t) L], 8 * (i + 1), per_sample) / (double) tr[(size_t) L].size();
            printf(",%.2f", v);
        }
        for (int i = 0; i < NCAP; ++i) {
            const double v = 1000.0 * (double) loads_at(he[(size_t) L], hes[(size_t) L], 8 * (i + 1), per_sample) / (double) nref;
            printf(",%.2f", v);
        }
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
