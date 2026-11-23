// UFS read bandwidth microbenchmark: buffered preadv vs O_DIRECT, N workers.
// Reports: mincore residency probe before read, logical GiB/s, physical read_bytes delta.
// Self-contained, POSIX + Linux (/proc/self/io). Build: cc -O2 -pthread ufs_read_bench.c -o ufs_read_bench
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

static long long now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static long long read_read_bytes(void) {
    FILE *f = fopen("/proc/self/io", "r");
    if (!f) return -1;
    char line[256]; long long v = -1;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "read_bytes:", 11) == 0) { v = atoll(line + 11); break; }
    }
    fclose(f);
    return v;
}

// mincore residency probe: map read-only, count resident pages, never dereference.
static void mincore_probe(const char *path, size_t len, size_t *resident, size_t *total) {
    *resident = 0; *total = 0;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    size_t pg = (size_t)sysconf(_SC_PAGESIZE);
    size_t npages = (len + pg - 1) / pg;
    void *m = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { close(fd); return; }
    unsigned char *vec = calloc(npages, 1);
    if (vec && mincore(m, len, vec) == 0) {
        for (size_t i = 0; i < npages; i++) if (vec[i] & 1) (*resident)++;
        *total = npages;
    }
    free(vec);
    munmap(m, len);
    close(fd);
}

typedef struct {
    const char *path;
    int direct;
    off_t start, end;      // [start, end)
    size_t chunk;          // per-pread size
    int use_fadv_random;
    long long elapsed_ns;
    ssize_t bytes_read;
    int err;
} job_t;

static void *worker(void *arg) {
    job_t *j = (job_t *)arg;
    int flags = O_RDONLY | O_CLOEXEC;
    if (j->direct) flags |= O_DIRECT;
    int fd = open(j->path, flags);
    if (fd < 0) { j->err = errno; return NULL; }
#ifdef POSIX_FADV_RANDOM
    if (j->direct == 0 && j->use_fadv_random) posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
#endif
    void *buf = NULL;
    size_t align = 4096;
    if (posix_memalign(&buf, align, j->chunk) != 0) { close(fd); j->err = ENOMEM; return NULL; }
    long long t0 = now_ns();
    off_t off = j->start;
    while (off < j->end) {
        size_t want = j->chunk;
        if ((off_t)want > j->end - off) want = (size_t)(j->end - off);
        ssize_t n = pread(fd, buf, want, off);
        if (n < 0) { if (errno == EINTR) continue; j->err = errno; break; }
        if (n == 0) break;
        j->bytes_read += n;
        off += n;
    }
    j->elapsed_ns = now_ns() - t0;
    free(buf);
    close(fd);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s FILE [reps] [chunk_kb] [fadv_random]\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    int reps = argc > 2 ? atoi(argv[2]) : 2;
    size_t chunk = (argc > 3 ? (size_t)atoi(argv[3]) : 1536) * 1024; // 1.5MiB default
    int use_fadv_random = argc > 4 ? atoi(argv[4]) : 0;

    struct stat st;
    if (stat(path, &st) != 0) { perror("stat"); return 1; }
    size_t flen = (size_t)st.st_size;

    int workers_set[] = {1, 2, 4, 8};
    int modes[] = {0, 1}; // 0=buffered, 1=O_DIRECT

    printf("{\"file\":\"%s\",\"size_bytes\":%zu,\"chunk_bytes\":%zu,\"fadv_random\":%d,\"runs\":[\n",
           path, flen, chunk, use_fadv_random);

    int first = 1;
    for (int mi = 0; mi < 2; mi++) {
        int direct = modes[mi];
        for (int wi = 0; wi < 4; wi++) {
            int nw = workers_set[wi];
            for (int rep = 0; rep < reps; rep++) {
                // cold: drop page cache advice per range; mincore probe before read
                int cf = open(path, O_RDONLY | O_CLOEXEC);
                if (cf >= 0) {
#ifdef POSIX_FADV_DONTNEED
                    posix_fadvise(cf, 0, 0, POSIX_FADV_DONTNEED);
#endif
                    close(cf);
                }
                size_t res = 0, tot = 0;
                mincore_probe(path, flen, &res, &tot);

                long long rb0 = read_read_bytes();

                pthread_t th[8]; job_t jobs[8];
                size_t per = (flen + nw - 1) / nw;
                long long t0 = now_ns();
                for (int i = 0; i < nw; i++) {
                    jobs[i] = (job_t){ .path = path, .direct = direct,
                        .start = (off_t)(per * i), .end = (off_t)(per * (i + 1) < flen ? per * (i + 1) : flen),
                        .chunk = chunk, .use_fadv_random = use_fadv_random };
                    jobs[i].end = jobs[i].end > (off_t)flen ? (off_t)flen : jobs[i].end;
                    pthread_create(&th[i], NULL, worker, &jobs[i]);
                }
                for (int i = 0; i < nw; i++) pthread_join(th[i], NULL);
                long long wall = now_ns() - t0;
                long long rb1 = read_read_bytes();

                ssize_t total = 0; int err = 0;
                for (int i = 0; i < nw; i++) { total += jobs[i].bytes_read; if (jobs[i].err) err = jobs[i].err; }

                double gib = (double)total / (1024.0*1024.0*1024.0);
                double secs = (double)wall / 1e9;
                double gbps = secs > 0 ? gib / secs : 0;
                long long phys = (rb0 >= 0 && rb1 >= 0) ? (rb1 - rb0) : -1;

                printf("%s{\"mode\":\"%s\",\"workers\":%d,\"rep\":%d,\"bytes\":%zd,\"wall_ms\":%.3f,"
                       "\"gib_s\":%.3f,\"phys_read_kib\":%lld,\"mincore_resident\":%zu,\"mincore_total\":%zu,"
                       "\"err\":%d}",
                       first ? "" : ",\n", direct ? "direct" : "buffered", nw, rep,
                       total, (double)wall/1e6, gbps, phys/1024, res, tot, err);
                first = 0;
                fflush(stdout);
            }
        }
    }
    printf("\n]}\n");
    return 0;
}
