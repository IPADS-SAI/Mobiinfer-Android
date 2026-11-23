// MoE expert-read microbench: read K experts out of one layer file as full
// per-expert strides, comparing SEQUENTIAL vs RANDOM expert selection.
// Each expert = `stride` bytes (gate|up|down contiguous). One pread per expert
// (or chunked if chunk<stride). Cold per run: FADV_DONTNEED whole file + mincore
// probe + /proc/self/io read_bytes delta.
// build: clang -O2 -pthread expert_read_bench.c -o expert_read_bench
// run:   expert_read_bench FILE STRIDE K MODE TURNS DIRECT SEED WORKERS [REPS]
//   MODE = seq | rand ; TURNS = how many times to repeat the K-expert set
//   DIRECT = 0 buffered / 1 O_DIRECT ; SEED for rand
//   WORKERS = number of reader threads that pull experts from a shared queue
//             (W<=K; W=K means one thread per expert, the original behaviour)
//
// MERGE-READ comparison (env ER_MERGE_SPAN=N, default 0 = off):
//   When set and MODE=seq, consecutive experts are read as groups of N experts
//   with ONE large pread of N*stride bytes into a scratch buffer (contiguous
//   file span), instead of one pread per expert. This directly measures the
//   "coalesce consecutive experts into a big preadv" idea against the per-expert
//   baseline, at the same K/workers/direct. ER_MERGE_SPAN=1 == no merging.
//   Only meaningful for seq (random experts are not contiguous in the file).
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

static long long now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (long long)t.tv_sec*1000000000LL+t.tv_nsec;}
static long long read_rb(void){FILE*f=fopen("/proc/self/io","r");if(!f)return -1;char l[256];long long v=-1;while(fgets(l,sizeof l,f))if(!strncmp(l,"read_bytes:",11)){v=atoll(l+11);break;}fclose(f);return v;}
static void mincore_probe(const char*p,size_t len,size_t*res,size_t*tot){*res=0;*tot=0;int fd=open(p,O_RDONLY|O_CLOEXEC);if(fd<0)return;size_t pg=sysconf(_SC_PAGESIZE),n=(len+pg-1)/pg;void*m=mmap(NULL,len,PROT_READ,MAP_SHARED,fd,0);if(m==MAP_FAILED){close(fd);return;}unsigned char*v=calloc(n,1);if(v&&mincore(m,len,v)==0){for(size_t i=0;i<n;i++)if(v[i]&1)(*res)++;*tot=n;}free(v);munmap(m,len);close(fd);}

typedef struct{const char*path;int direct;off_t start;size_t len;ssize_t got;int err;}job_t;
static void do_read(job_t*j){int fl=O_RDONLY|O_CLOEXEC;if(j->direct)fl|=O_DIRECT;int fd=open(j->path,fl);if(fd<0){j->err=errno;return;}void*buf;if(posix_memalign(&buf,4096,j->len)){close(fd);j->err=ENOMEM;return;}off_t off=j->start;size_t rem=j->len;while(rem){ssize_t n=pread(fd,(char*)buf+(j->len-rem),rem,off);if(n<0){if(errno==EINTR)continue;j->err=errno;break;}if(!n)break;j->got+=n;off+=n;rem-=n;}free(buf);close(fd);}

// shared pull queue: W worker threads grab the next job index until exhausted
typedef struct{job_t*jobs;int n;int next;pthread_mutex_t mtx;}queue_t;
static void*pull_worker(void*a){queue_t*q=a;for(;;){pthread_mutex_lock(&q->mtx);int i=q->next++;pthread_mutex_unlock(&q->mtx);if(i>=q->n)break;do_read(&q->jobs[i]);}return NULL;}

int main(int argc,char**argv){
 if(argc<9){fprintf(stderr,"usage: %s FILE STRIDE K MODE TURNS DIRECT SEED WORKERS [REPS]\n",argv[0]);return 1;}
 const char*path=argv[1];size_t stride=strtoull(argv[2],0,10);int K=atoi(argv[3]);
 const char*mode=argv[4];int turns=atoi(argv[5]);int direct=atoi(argv[6]);unsigned seed=(unsigned)strtoul(argv[7],0,10);
 int W=atoi(argv[8]);if(W<1)W=1;if(W>K)W=K;
  int reps=argc>9?atoi(argv[9]):1;
  int span=0;{const char*e=getenv("ER_MERGE_SPAN");if(e)span=atoi(e);}
  if(span<0)span=0;
  struct stat st;if(stat(path,&st)){perror("stat");return 1;}
  int nexp=(int)(st.st_size/stride);
  if(K>nexp){fprintf(stderr,"K>experts\n");return 1;}
  srand(seed);
  // build expert id list for one "turn"
  int ids[4096];
  if(!strcmp(mode,"seq")){for(int i=0;i<K;i++)ids[i]=i;}
  else { // rand: sample K distinct in [0,nexp)
    char*used=calloc(nexp,1);int got=0;while(got<K){int e=rand()%nexp;if(!used[e]){used[e]=1;ids[got++]=e;}}free(used);
  }
  // merge-read: only for seq (contiguous in file). Build super-jobs of `span`
  // consecutive experts each; each super-job is one pread of span*stride bytes.
  int merging = (span>1 && !strcmp(mode,"seq"));
  int njob=K; job_t*sjob=NULL;
  if(merging){
    njob=(K+span-1)/span;
    sjob=calloc(njob,sizeof(job_t));
    for(int g=0;g<njob;g++){
      int first_ex=ids[g*span];
      int cnt=(g*span+span<=K)?span:(K-g*span);
      sjob[g]=(job_t){.path=path,.direct=direct,.start=(off_t)first_ex*stride,.len=(size_t)cnt*stride};
    }
  }
  printf("{\"file\":\"%s\",\"stride\":%zu,\"experts_total\":%d,\"K\":%d,\"mode\":\"%s\",\"turns\":%d,\"direct\":%d,\"seed\":%u,\"workers\":%d,\"merge_span\":%d,\"n_reads_per_turn\":%d,\"runs\":[\n",path,stride,nexp,K,mode,turns,direct,seed,W,merging?span:1,merging?njob:K);
  int first=1;
  for(int rep=0;rep<reps;rep++){
    int cf=open(path,O_RDONLY|O_CLOEXEC);if(cf>=0){posix_fadvise(cf,0,0,POSIX_FADV_DONTNEED);close(cf);}
    size_t res,tot;mincore_probe(path,st.st_size,&res,&tot);
    long long rb0=read_rb();
    long long t0=now_ns();long long bytes=0;int err=0;
    for(int t=0;t<turns;t++){ // one "decode step" reads K experts via W pull-workers
      if(merging){
        queue_t q;q.jobs=sjob;q.n=njob;q.next=0;pthread_mutex_init(&q.mtx,NULL);
        pthread_t th[64];
        for(int w=0;w<W;w++)pthread_create(&th[w],NULL,pull_worker,&q);
        for(int w=0;w<W;w++)pthread_join(th[w],NULL);
        for(int i=0;i<njob;i++){bytes+=sjob[i].got;if(sjob[i].err)err=sjob[i].err;sjob[i].got=0;}
        pthread_mutex_destroy(&q.mtx);
      } else {
        queue_t q;q.jobs=calloc(K,sizeof(job_t));q.n=K;q.next=0;pthread_mutex_init(&q.mtx,NULL);
        for(int i=0;i<K;i++)q.jobs[i]=(job_t){.path=path,.direct=direct,.start=(off_t)ids[i]*stride,.len=stride};
        pthread_t th[64];
        for(int w=0;w<W;w++)pthread_create(&th[w],NULL,pull_worker,&q);
        for(int w=0;w<W;w++)pthread_join(th[w],NULL);
        for(int i=0;i<K;i++){bytes+=q.jobs[i].got;if(q.jobs[i].err)err=q.jobs[i].err;}
        pthread_mutex_destroy(&q.mtx);free(q.jobs);
      }
    }
    long long wall=now_ns()-t0;long long rb1=read_rb();
    double gib=(double)bytes/(1024.0*1024.0*1024.0),secs=(double)wall/1e9;
    double gbps=secs>0?gib/secs:0;
    printf("%s{\"rep\":%d,\"experts_read\":%d,\"bytes\":%lld,\"wall_ms\":%.3f,\"gib_s\":%.3f,\"per_expert_ms\":%.4f,\"reads\":%d,\"phys_read_kib\":%lld,\"mc_res\":%zu,\"mc_tot\":%zu,\"err\":%d}",
      first?"":",\n",rep,K*turns,bytes,(double)wall/1e6,gbps,(double)wall/1e6/(K*turns),(merging?njob:K)*turns,(rb0>=0&&rb1>=0)?(rb1-rb0)/1024:-1,res,tot,err);
    first=0;fflush(stdout);
  }
  free(sjob);
  printf("\n]}\n");return 0;
}
