// runp: launch a command, sample the whole process tree (and optionally per-UID)
// from uid shell, once per second, reading raw kernel counters that only a
// non-app SELinux domain may read:
//   per process (from /proc/<pid>/stat & /proc/<pid>/io):
//     utime/stime (jiffies), rchar, wchar, syscr, syscw, read_bytes, write_bytes
//   device wide: /proc/stat per-core busy ticks, /proc/diskstats io_ticks per device
// usage: runp <csv> <dwell_s> [--pkg <package>] -- <cmd> [args...]
//   with --pkg: the csv gets one row per second aggregating (a) the launched tree,
//   (b) every process of the package's uid, (c) whole device.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <time.h>
#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <linux/limits.h>

struct proc_io { unsigned long long rchar, wchar, syscr, syscw, rbytes, wbytes; };
struct proc_cpu { unsigned long long utime, stime, minflt, majflt; };
struct proc_io;
static int read_io(pid_t p, struct proc_io *io);
struct proc_io;
static int read_io(pid_t p, struct proc_io *io);

static long long now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

// Window alignment with llm_windows.log / sp_helper needs wall-clock epoch ms;
// scheduling/elapsed still uses CLOCK_MONOTONIC above.
static long long now_epoch_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static int read_status_uid(pid_t p) {
    char path[64]; snprintf(path, sizeof path, "/proc/%d/status", p);
    FILE *f = fopen(path, "r"); if (!f) return -1;
    char line[256]; int uid = -1, real = -1;
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "Uid:", 4)) {
            // Uid: real effective saved fs
            unsigned a, b, c, d;
            if (sscanf(line, "Uid:\t%u\t%u\t%u\t%u", &a, &b, &c, &d) >= 1) real = (int)a;
            break;
        }
    }
    fclose(f); uid = real; return uid;
}

static int read_cpu(pid_t p, struct proc_cpu *c) {
    // read(2) bulk: under the uid_change view FILE*/fgets on app /proc files returns zeros
    char path[64]; snprintf(path, sizeof path, "/proc/%d/stat", p);
    static char buf[4096];
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    ssize_t n = read(fd, buf, sizeof buf - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = 0;
    char *rp = strrchr(buf, ')'); if (!rp) return -1;
    unsigned long long vals[12] = {0}; int m = 0;
    char *tok = strtok(rp + 2, " ");
    if (!tok) return -1;
    tok = strtok(NULL, " ");                       // skip state
    while (tok && m < 12) { vals[m++] = strtoull(tok, NULL, 10); tok = strtok(NULL, " "); }
    if (m < 12) return -1;
    c->utime = vals[10]; c->stime = vals[11];
    c->minflt = vals[6]; c->majflt = vals[8];   // 缺页次数：major 才代表"页被换出/未命中"
    return 0;
}

static void read_launcher(pid_t p, struct proc_cpu *c, struct proc_io *io);
static int read_io(pid_t p, struct proc_io *io) {
    char path[64]; snprintf(path, sizeof path, "/proc/%d/io", p);
    FILE *f = fopen(path, "r"); if (!f) return -1;
    char line[256]; memset(io, 0, sizeof *io); int got = 0;
    while (fgets(line, sizeof line, f)) {
        char key[64]; unsigned long long val;
        if (sscanf(line, "%63[^:]: %llu", key, &val) == 2) {
            if (!strcmp(key, "rchar")) io->rchar = val;
            else if (!strcmp(key, "wchar")) io->wchar = val;
            else if (!strcmp(key, "syscr")) io->syscr = val;
            else if (!strcmp(key, "syscw")) io->syscw = val;
            else if (!strcmp(key, "read_bytes")) io->rbytes = val;
            else if (!strcmp(key, "write_bytes")) io->wbytes = val;
            got = 1;
        }
    }
    fclose(f); return got ? 0 : -1;
}

#define MAXP 4096
struct snap { pid_t pid; pid_t ppid; int uid; struct proc_cpu cpu; struct proc_io io; int io_ok; };

static int dbg_opened, dbg_nobrace, dbg_badppid, dbg_badcpu;
static int collect(struct snap *out) {
    DIR *d = opendir("/proc"); if (!d) return 0;
    struct dirent *e; int n = 0;
    while ((e = readdir(d)) && n < MAXP) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        pid_t p = atoi(e->d_name);
        char path[64]; snprintf(path, sizeof path, "/proc/%d/stat", p);
        static char buf[4096];       // same read(2) path as read_cpu: FILE* returns zeros here
        int fd = open(path, O_RDONLY); if (fd < 0) continue;
        ssize_t sz = read(fd, buf, sizeof buf - 1); close(fd);
        if (sz <= 0) continue;
        dbg_opened++;
        buf[sz] = 0;
        char *rp = strrchr(buf, ')'); if (!rp) { dbg_nobrace++; continue; }
        int ppid = 0; if (sscanf(rp + 2, " %*c %d", &ppid) != 1) { dbg_badppid++; continue; }
        struct proc_cpu c; if (read_cpu(p, &c)) { dbg_badcpu++; continue; }
        struct snap *s = &out[n++];
        s->pid = p; s->ppid = ppid; s->uid = read_status_uid(p);
        s->cpu = c; s->io_ok = (read_io(p, &s->io) == 0);
    }
    closedir(d); return n;
}

// is `p` (with ppid `pp`) a descendant of `root`?  fixed depth bound, no cycles
// is `p` (with ppid `pp`) a descendant of `root`?  fixed depth bound, no cycles
static int tree_hits; static int is_desc(const struct snap *snap, int n, pid_t p, pid_t root) {
    if (getenv("DBG")) fprintf(stderr, "check %d root %d\n", (int)p, (int)root);
    if (root <= 1) return 0;               // never trust pid 1/0 as "launcher"
    pid_t cur = p;
    for (int hop = 0; hop < 24; hop++) {
        if (cur == root) { tree_hits++; return 1; }
        if (cur <= 1) return 0;
        int found = 0;
        for (int i = 0; i < n; i++) if (snap[i].pid == cur) { cur = snap[i].ppid; found = 1; break; }
        if (!found) return 0;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// --uid-file mode: we cannot open /proc/<pid>/io of APP uids (uid_io S_IRWXU).
// But an app UID can open a file whose uid *is* that app uid. So we fork per target
// uid, setreuid() to it (shell may do that; no CAP needed) and read /proc there.
struct urec { pid_t pid; unsigned long long u, s, rchar, wchar, rbytes, wbytes, syscr, syscw; int io_ok; };

static void read_launcher(pid_t p, struct proc_cpu *c, struct proc_io *io) {
    // launcher's own /proc/<pid>/stat is unreadable from uid shell (uid has a uid-change
    // thread) -> getrusage(children) gives exact CPU of launched children.
    struct rusage ru; getrusage(RUSAGE_CHILDREN, &ru);
    c->utime = (unsigned long long)(ru.ru_utime.tv_sec * 100 + ru.ru_utime.tv_usec / 10000);
    c->stime = (unsigned long long)(ru.ru_stime.tv_sec * 100 + ru.ru_stime.tv_usec / 10000);
    if (read_io(p, io)) memset(io, 0, sizeof *io);
}

static int read_rec(pid_t p, struct urec *r, int as_uid) {
    struct proc_cpu c; struct proc_io io; memset(r, 0, sizeof *r);
    r->pid = p;
    int ok_cpu = 0;
    if (!as_uid) ok_cpu = (read_cpu(p, &c) == 0);
    if (as_uid) {
        // in ORIGINAL domain: /proc/<pid>/stat is still readable for app pids
        ok_cpu = (read_cpu(p, &c) == 0);
    }
    if (ok_cpu) { r->u = c.utime; r->s = c.stime; }
    r->io_ok = (read_io(p, &io) == 0);
    if (r->io_ok) { r->rchar = io.rchar; r->wchar = io.wchar; r->rbytes = io.rbytes; r->wbytes = io.wbytes; r->syscr = io.syscr; r->syscw = io.syscw; }
    return ok_cpu || r->io_ok;
}

struct snap;
static int is_desc(const struct snap *snap, int n, pid_t p, pid_t root);

// Runs inside a small child that setresuid()'d to the target uid (same UID => these
// /proc files are readable): rchar/wchar/syscr/syscw per pid + per-uid schedstat.
static void app_view(int uid, int *pids, int np, int fd) {
    char buf[256];
    for (int i = 0; i < np; i++) {
        struct proc_io io;
        if (read_io(pids[i], &io) == 0) {
            int k = snprintf(buf, sizeof buf, "P %d %llu %llu %llu %llu\n", pids[i],
                    (unsigned long long)io.rchar, (unsigned long long)io.wchar,
                    (unsigned long long)io.syscr, (unsigned long long)io.syscw);
            if (write(fd, buf, (size_t)k) != k) break;
        }
    }
}

// fork a child that setreuid(uid) then reads the records for `pidfile`-listed pids.
static void uid_reader(int uid, int pidfd_in, int resfd_out, int *pids, int np) {
    (void)pidfd_in;
    int g_ok = (setresgid(uid, uid, uid) == 0), u_ok = (setresuid(uid, uid, uid) == 0);
    (void)g_ok;
    if (u_ok) { app_view(uid, pids, np, resfd_out); _exit(0); }
    if (getenv("DBG")) fprintf(stderr, "reader switch failed: euid=%d egid=%d\n", geteuid(), getegid());
    {
        struct urec bad; memset(&bad, 0, sizeof bad);
        for (int i = 0; i < np; i++) write(resfd_out, &bad, sizeof bad);
        _exit(3);
    }
    if (getenv("DBG")) fprintf(stderr, "reader uid=%d euid=%d np=%d\n", uid, geteuid(), np);
    for (int i = 0; i < np; i++) {
        struct urec r; read_rec(pids[i], &r, 1);
        if (getenv("DBG")) fprintf(stderr, "  pid %d cpu_ok u=%llu s=%llu io_ok=%d rchar=%llu\n", r.pid, (unsigned long long)r.u, (unsigned long long)r.s, r.io_ok, (unsigned long long)r.rchar);
        if (write(resfd_out, &r, sizeof r) != (ssize_t)sizeof r) _exit(4);
    }
    _exit(0);
}

static int pids_by_name(const char *pkg, int *pids, int max) {
    // uid of an app process is hidden from shell, but pidof <pkg>[suffix] works -> we can
    // address exactly the app's processes; their uid crashes are tolerated by the readers.
    char cmd[256], out[2048] = "";
    snprintf(cmd, sizeof cmd, "pidof %s %s:push %s:remote %s:sandbox %s:support %s:msf 2>/dev/null", pkg, pkg, pkg, pkg, pkg, pkg);
    FILE *p = popen(cmd, "r");
    int n = 0;
    if (p) { int ch; size_t k = 0; while (k < sizeof out - 1 && (ch = fgetc(p)) != EOF) out[k++] = (char)ch; out[k] = 0; pclose(p); }
    if (getenv("DBG")) fprintf(stderr, "pidof: [%s]\n", out);
    char *tok = strtok(out, " \n");
    while (tok && n < max) { pids[n++] = atoi(tok); tok = strtok(NULL, " \n"); }
    return n;
}

static int collect_pids(int uid, int *pids, int max) {
    DIR *d = opendir("/proc"); if (!d) return 0;
    struct dirent *e; int n = 0;
    while ((e = readdir(d)) && n < max) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        pid_t p = atoi(e->d_name);
        if (read_status_uid(p) == uid) pids[n++] = p;
    }
    closedir(d); return n;
}

static void read_statdev(unsigned long long *busy, unsigned long long *iow, int *ncpu) {
    // /proc/stat parses but *reads back zeros* when read via FILE* under the uid_change
    // mount view (kernel strips the data), so read it with plain read(2) in bulk instead.
    *busy = 0; *iow = 0; *ncpu = 0;
    static char buf[16384];
    int fd = open("/proc/stat", O_RDONLY);
    if (fd < 0) return;
    ssize_t n = read(fd, buf, sizeof buf - 1);
    close(fd);
    if (n <= 0) return;
    buf[n] = 0;
    char *p = buf;
    while (p && *p) {
        char *nl = strchr(p, '\n');
        if (nl) *nl = 0;
        if (!strncmp(p, "cpu", 3) && (p[3] == ' ' || (p[3] >= '0' && p[3] <= '9'))) {
            unsigned long long v[10] = {0};
            char *tok = strtok(p + 4, " ");
            for (int i = 0; i < 10 && tok; i++, tok = strtok(NULL, " ")) v[i] = strtoull(tok, NULL, 10);
            // busy 必须排除 idle(v3) 和 iowait(v4)：iowait 是 CPU 不执行的时间
            unsigned long long bs = v[0] + v[1] + v[2] + v[5] + v[6] + v[7];
            if (p[3] == ' ') *iow = v[4];
            else { *busy += bs; (*ncpu)++; }
        }
        p = nl ? nl + 1 : NULL;
    }
    if (getenv("DBG")) fprintf(stderr, "statdev: busy=%llu iow=%llu ncpu=%d\n", *busy, *iow, *ncpu);
}

// one sampling tick: uid-switched children read the delta for every process of `uid`
// (created or grown since the previous tick), plus tree deltas if requested.
struct tick {
    unsigned long long u_cpu, u_rchar, u_wchar, u_rb, u_wb, u_sc, u_scw; int u_n;
    unsigned long long tr_cpu, tr_rchar, tr_wchar, tr_rb, tr_wb, tr_sc, tr_scw; int tr_n;
    unsigned long long u_minflt, u_majflt;
};
static struct snap s0[MAXP]; static int n0;
static unsigned long long iow_prev; static int iow_prev_ok;
static unsigned long long acc_busy, acc_iot, acc_tr_cpu, acc_u_cpu, acc_tr_r, acc_tr_w, acc_u_r, acc_u_w,
                            acc_tr_rb, acc_tr_wb, acc_u_rb, acc_u_wb, acc_tr_sc, acc_tr_scw, acc_u_sc, acc_u_scw, acc_iow;
static unsigned long long acc_iow;
static void tick_sample(int uid, pid_t root, struct tick *t) {
    memset(t, 0, sizeof *t);
    static struct snap s1[MAXP]; int n1 = collect(s1);
    if (root > 1) {
        for (int k = 0; k < n1; k++) {
            if (s1[k].pid == root) continue;
            if (!is_desc(s1, n1, s1[k].pid, root)) continue;
            t->tr_n++;
            int prev = -1;
            for (int j = 0; j < n0; j++) if (s0[j].pid == s1[k].pid) { prev = j; break; }
            if (prev >= 0 && s1[k].io_ok && s0[prev].io_ok) {
                t->tr_cpu += (s1[k].cpu.utime + s1[k].cpu.stime) - (s0[prev].cpu.utime + s0[prev].cpu.stime);
                t->tr_rchar += s1[k].io.rchar - s0[prev].io.rchar;
                t->tr_wchar += s1[k].io.wchar - s0[prev].io.wchar;
                t->tr_rb += s1[k].io.rbytes - s0[prev].io.rbytes;
                t->tr_wb += s1[k].io.wbytes - s0[prev].io.wbytes;
                t->tr_sc += s1[k].io.syscr - s0[prev].io.syscr;
                t->tr_scw += s1[k].io.syscw - s0[prev].io.syscw;
            }
        }
    }
    if (uid > 0) {
        static int pids[MAXP]; int np = collect_pids(uid, pids, MAXP);
        t->u_n = np;
        if (np > 0) {
            int fds[2]; if (pipe(fds) == 0) {
                pid_t c = fork();
                if (c == 0) { close(fds[0]); uid_reader(uid, -1, fds[1], pids, np); _exit(9); }
                close(fds[1]);
                char buf[1 << 16]; char acc[1 << 16]; size_t accn = 0;
                ssize_t got;
                while ((got = read(fds[0], buf, sizeof buf)) > 0) {
                    if (accn + (size_t)got > sizeof acc) accn = 0;    // overflow: drop
                    memcpy(acc + accn, buf, (size_t)got); accn += (size_t)got;
                }
                char *ln = acc, *end = acc + accn;
                while (ln < end) {
                    char *nl = memchr(ln, '\n', (size_t)(end - ln));
                    if (!nl) break;
                    *nl = 0;
                    if (ln[0] == 'P') {
                        int pid; unsigned long long rc, wc, scr, scw;
                        if (sscanf(ln + 1, "%d %llu %llu %llu %llu", &pid, &rc, &wc, &scr, &scw) == 5) {
                            int prev = -1;
                            for (int j = 0; j < n0; j++) if (s0[j].pid == pid) { prev = j; break; }
                            if (prev >= 0 && s0[prev].io_ok) {
                                t->u_rchar += rc - s0[prev].io.rchar;
                                t->u_wchar += wc - s0[prev].io.wchar;
                                t->u_sc += scr - s0[prev].io.syscr;
                                t->u_scw += scw - s0[prev].io.syscw;
                            }
                        }
                    }
                    ln = nl + 1;
                }
                close(fds[0]); int st; waitpid(c, &st, 0);
                if (getenv("DBG")) fprintf(stderr, "uid=%d np=%d child_exit=%d\n", uid, np, WIFEXITED(st) ? WEXITSTATUS(st) : -1);
            }
        }
    }
    memcpy(s0, s1, (size_t)n1 * sizeof(struct snap)); n0 = n1;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s csv dwell [--uid N] [--pkg pkg] [--] [cmd...]\n", argv[0]); return 2; }
    const char *csv = argv[1]; int dwell = atoi(argv[2]);
    int target_uid = -1; const char *pkg = NULL; int i = 3;
    while (i < argc && strncmp(argv[i], "--", 2) == 0) {
        if (!strcmp(argv[i], "--uid") && i + 1 < argc) { target_uid = atoi(argv[i + 1]); i += 2; }
        else if (!strcmp(argv[i], "--pkg") && i + 1 < argc) { pkg = argv[i + 1]; i += 2; }
        else if (!strcmp(argv[i], "--")) { i++; break; }
        else { fprintf(stderr, "bad option %s\n", argv[i]); return 2; }
    }
    const char *name_target = NULL;      // per-app mode by process name (uid may be hidden)
    if (pkg && target_uid < 0) name_target = pkg;
    if (pkg && target_uid >= 0) {
        char cmd[512];
        snprintf(cmd, sizeof cmd,
                 "dumpsys package %s 2>/dev/null | sed -n -e 's/^ *userId=\\([0-9]*\\).*/\\1/p' -e 's/^ *appId=\\([0-9]*\\).*/\\1/p' | head -1", pkg);
        FILE *p = popen(cmd, "r");
        if (p) {
            // use getc loop: FILE* reads of kernel files return zeros under the uid_change view
            int ch, k = 0; char nb[32] = {0};
            while (k < (int)sizeof nb - 1 && (ch = fgetc(p)) != EOF) { if (ch >= '0' && ch <= '9') nb[k++] = (char)ch; else if (k) break; }
            pclose(p);
            if (k) target_uid = atoi(nb);
        }
        if (getenv("DBG")) fprintf(stderr, "pkg=%s target_uid=%d\n", pkg, target_uid);
        name_target = pkg;               // always prefer exact process names: surfaced UIDs hide other procs
    }
    int has_cmd = (i < argc);
    FILE *out = fopen(csv, "w");
    if (!out) { perror(csv); return 1; }
    fprintf(out, "ts_ms,epoch_ms,elapsed_s,phase,rc,dev_busy_jiffies,dev_iowait_jiffies,dev_io_ticks,ncpu,"
                 "tree_n,tr_cpu,tr_rchar,tr_wchar,tr_rbytes,tr_wbytes,tr_syscr,tr_syscw,"
                 "uid_n,u_cpu,u_rchar,u_wchar,u_rbytes,u_wbytes,u_syscr,u_syscw,u_majflt,u_minflt,"
                 "launcher_utime,launcher_stime,launcher_rchar,launcher_wchar,launcher_rbytes,launcher_wbytes,"
                 "devload_runq,procs_blocked,target_uid\n");
    fflush(out);

    pid_t mypid = getpid();
    n0 = collect(s0);
    unsigned long long b0 = 0, iw0 = 0; int ncpu = 0; read_statdev(&b0, &iw0, &ncpu);
    struct proc_io pl; struct proc_cpu cl; read_launcher(mypid, &cl, &pl);
    long long t0 = now_ms();
    pid_t child = -1;
    if (has_cmd) {
        child = fork();
        if (child == 0) {
            setpgid(0, 0);
            signal(SIGPIPE, SIG_DFL);
            execvp(argv[i], &argv[i]);
            _exit(127);
        }
    }
    int sec = 0, child_done = has_cmd ? 0 : 1, status = 0;
    acc_busy = acc_iot = acc_tr_cpu = acc_u_cpu = 0;
    unsigned long long u_maj = 0, u_min = 0;
    while (sec < dwell) {
        long long next = t0 + (long long)(sec + 1) * 1000;
        long long wait = next - now_ms();
        if (wait > 0) usleep((useconds_t)wait * 1000);
        sec++;
        if (!child_done) { pid_t w = waitpid(child, &status, WNOHANG); if (w == child) child_done = 1; }
        int rc = child_done ? (WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status)) : -1;
        struct tick t; memset(&t, 0, sizeof t);
        if (getenv("DBG")) fprintf(stderr, "collect: opened=%d nobrace=%d badppid=%d badcpu=%d\n", dbg_opened, dbg_nobrace, dbg_badppid, dbg_badcpu);
        if (name_target) {
            // app uid hidden from shell: address the app's processes by name (pidof works)
            static int pp[MAXP]; static struct snap ss[MAXP];
            int np = pids_by_name(name_target, pp, MAXP);
            int nn = collect(ss);
            for (int k = 0; k < nn; k++) {
                int hit = 0;
                for (int j = 0; j < np; j++) if (ss[k].pid == pp[j]) { hit = 1; break; }
                if (!hit) continue;
                int prev = -1;
                for (int j = 0; j < n0; j++) if (s0[j].pid == ss[k].pid) { prev = j; break; }
                t.u_n++;
                if (prev >= 0) {
                    t.u_cpu += (ss[k].cpu.utime + ss[k].cpu.stime) - (s0[prev].cpu.utime + s0[prev].cpu.stime);
                    if (ss[k].cpu.majflt >= s0[prev].cpu.majflt)
                        t.u_majflt += ss[k].cpu.majflt - s0[prev].cpu.majflt;
                    if (ss[k].cpu.minflt >= s0[prev].cpu.minflt)
                        t.u_minflt += ss[k].cpu.minflt - s0[prev].cpu.minflt;
                    if (ss[k].io_ok && s0[prev].io_ok) {
                        t.u_rchar += ss[k].io.rchar - s0[prev].io.rchar; t.u_wchar += ss[k].io.wchar - s0[prev].io.wchar;
                        t.u_rb += ss[k].io.rbytes - s0[prev].io.rbytes;   t.u_wb += ss[k].io.wbytes - s0[prev].io.wbytes;
                        t.u_sc += ss[k].io.syscr - s0[prev].io.syscr;     t.u_scw += ss[k].io.syscw - s0[prev].io.syscw;
                    }
                }
            }
            memcpy(s0, ss, (size_t)nn * sizeof(struct snap)); n0 = nn;
        } else {
            tick_sample(target_uid, has_cmd ? mypid : -1, &t);
        }
        unsigned long long b1 = 0, iw1 = 0; read_statdev(&b1, &iw1, &ncpu);
        if (iow_prev_ok) acc_iow += iw1 - iow_prev;
        iow_prev = iw1; iow_prev_ok = 1;
        struct proc_io pl1; struct proc_cpu cl1; read_launcher(mypid, &cl1, &pl1);
        unsigned long long q = 0, blk = 0;
        FILE *lf = fopen("/proc/loadavg", "r");
        if (lf) { char lb[128]; if (fgets(lb, sizeof lb, lf)) { char *sl = strchr(lb, '/'); if (sl) { sscanf(sl + 1, "%llu", &q); char *sp = strchr(sl, ' '); if (sp) sscanf(sp + 1, "%llu", &blk);} } fclose(lf); }
        acc_busy += b1 - b0;
        acc_tr_cpu += t.tr_cpu; acc_u_cpu += t.u_cpu;
        acc_tr_r += t.tr_rchar; acc_tr_w += t.tr_wchar; acc_u_r += t.u_rchar; acc_u_w += t.u_wchar;
        acc_tr_rb += t.tr_rb; acc_tr_wb += t.tr_wb; acc_u_rb += t.u_rb; acc_u_wb += t.u_wb;
        acc_tr_sc += t.tr_sc; acc_tr_scw += t.tr_scw; acc_u_sc += t.u_sc; acc_u_scw += t.u_scw;
        u_maj += t.u_majflt; u_min += t.u_minflt;
        fprintf(out, "%lld,%lld,%lld,%s,%d,%llu,%llu,%llu,%d,"
                     "%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
                     "%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
                     "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%d\n",
                now_ms(), now_epoch_ms(), (now_ms() - t0) / 1000, child_done ? "after" : "run", rc,
                acc_busy, acc_iow, acc_iot, ncpu,
                t.tr_n, acc_tr_cpu, acc_tr_r, acc_tr_w, acc_tr_rb, acc_tr_wb, acc_tr_sc, acc_tr_scw,
                t.u_n, acc_u_cpu, acc_u_r, acc_u_w, acc_u_rb, acc_u_wb, acc_u_sc, acc_u_scw,
                (unsigned long long)u_maj, (unsigned long long)u_min,
                cl1.utime - cl.utime, cl1.stime - cl.stime, pl1.rchar - pl.rchar, pl1.wchar - pl.wchar,
                pl1.rbytes - pl.rbytes, pl1.wbytes - pl.wbytes,
                q, blk, target_uid);
        fflush(out);
        b0 = b1; cl = cl1; pl = pl1;
    }
    if (!child_done) { kill(-child, SIGTERM); usleep(300000); kill(-child, SIGKILL); int st; waitpid(child, &st, WNOHANG); }
    fclose(out);
    return 0;
}
