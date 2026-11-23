#ifndef MOE_BENCH_SCHED_POLICY_H
#define MOE_BENCH_SCHED_POLICY_H

// Experimental policy: order, cold limit, release limit after hot completion,
// maximum pread size. Configuration is calibrated externally, not universal.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct {
    char order[32];
    int cap, release, chunk;
} sched_policy;

static int sched_policy_parse(const char *text, int workers, sched_policy *p) {
    char extra;
    if (sscanf(text, "%31[^:]:%d:%d:%d%c", p->order, &p->cap,
               &p->release, &p->chunk, &extra) != 4) return 0;
    int known = !strcmp(p->order, "random") || !strcmp(p->order, "cold") ||
        !strcmp(p->order, "hot") || !strcmp(p->order, "1c1h") ||
        !strcmp(p->order, "2c1h") || !strcmp(p->order, "1c2h") ||
        !strcmp(p->order, "proportional");
    if (!strncmp(p->order, "mask", 4)) {
        size_t n = strlen(p->order + 4);
        known = n > 0 && n <= 16 && strspn(p->order + 4, "ch") == n;
    }
    return known && p->cap >= 1 && p->cap <= workers &&
        (p->release == 0 || p->release == 1) &&
        (p->chunk == 0 || p->chunk == 262144 || p->chunk == 524288);
}

// Stable within each class; partially resident experts are sorted by missing
// bytes inside the cold class. The binary-mask oracle only covers hot/cold jobs.
static int sched_policy_order(const sched_policy *p, const int *ids,
                              const long long *missing, int n, int *out) {
    int cold[256], hot[256], nc = 0, nh = 0;
    if (n < 1 || n > 256) return 0;
    memcpy(out, ids, n * sizeof(*out));
    if (!strcmp(p->order, "random")) return 1;
    for (int i = 0; i < n; ++i) {
        int e = ids[i];
        if (missing[e]) cold[nc++] = e; else hot[nh++] = e;
    }
    for (int i = 1; i < nc; ++i) {
        int e = cold[i], j = i;
        while (j && missing[cold[j-1]] < missing[e]) {
            cold[j] = cold[j-1]; --j;
        }
        cold[j] = e;
    }
    const char *mask = !strncmp(p->order, "mask", 4) ? p->order + 4 : NULL;
    if (mask && ((int)strlen(mask) != n ||
        (int)strspn(mask, "ch") != n)) return 0;
    int c = 0, h = 0;
    for (int i = 0; i < n; ++i) {
        int choose;
        if (mask) choose = mask[i] == 'c';
        else if (!strcmp(p->order, "cold")) choose = 1;
        else if (!strcmp(p->order, "hot")) choose = 0;
        else if (!strcmp(p->order, "proportional"))
            choose = (i+1)*nc > c*n;
        else if (!strcmp(p->order, "2c1h")) choose = i%3 != 2;
        else if (!strcmp(p->order, "1c2h")) choose = i%3 == 0;
        else choose = i%2 == 0;
        if (mask && ((choose && c == nc) || (!choose && h == nh))) return 0;
        if (h == nh || (choose && c < nc)) out[i] = cold[c++];
        else out[i] = hot[h++];
    }
    return 1;
}
#endif
