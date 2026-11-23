// MoE expert-read with controllable page-cache hit ratio (buffered reads).
// Pre-warms the first P% of the chosen K experts into page cache, then reads
// all K experts (mixture of resident / cold) WITHOUT dropping cache, so the
// measured time reflects a realistic partial-hit working set.
//
// NOTE: meaningful only for buffered reads. O_DIRECT bypasses page cache, so a
// pre-warmed expert gives no benefit there (it still hits the device).
//
// build: clang -O2 -pthread expert_read_prewarm.c -o expert_read_prewarm
// run:   expert_read_prewarm FILE STRIDE K MODE PREWARM_PCT WORKERS SEED [REPS]
//   MODE = seq | rand ; PREWARM_PCT in [0,100] ; meaning: fraction of the K
//   experts to preload into page cache before the timed read. 0 = all cold,
//   100 = all hot (pure page-cache read).
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

static int g_direct=0;
typedef struct{job_t*jobs;int n;int next;pthread_mutex_t mtx;}queue_t;
static void*pull_worker(void*a){queue_t*q=a;for(;;){pthread_mutex_lock(&q->mtx);int i=q->next++;pthread_mutex_unlock(&q->mtx);if(i>=q->n)break;do_read(&q->jobs[i]);}return NULL;}

// read a set of experts with W pull-workers; returns wall_ns, fills got/err
static long long read_set(const char*path,size_t stride,const int*ids,int n,int W,long long*bytes,int*err){
  queue_t q;q.jobs=calloc(n,sizeof(job_t));q.n=n;q.next=0;pthread_mutex_init(&q.mtx,NULL);
  for(int i=0;i<n;i++)q.jobs[i]=(job_t){.path=path,.direct=g_direct,.start=(off_t)ids[i]*stride,.len=stride};
  long long t0=now_ns();
  pthread_t th[64];for(int w=0;w<W;w++)pthread_create(&th[w],NULL,pull_worker,&q);
  for(int w=0;w<W;w++)pthread_join(th[w],NULL);
  long long wall=now_ns()-t0;
  long long b=0;int e=0;for(int i=0;i<n;i++){b+=q.jobs[i].got;if(q.jobs[i].err)e=q.jobs[i].err;}
  pthread_mutex_destroy(&q.mtx);free(q.jobs);
  if(bytes)*bytes=b;if(err)*err=e;return wall;
}

int main(int argc,char**argv){
 if(argc<8){fprintf(stderr,"usage: %s FILE STRIDE K MODE PREWARM_PCT WORKERS SEED [REPS]\n",argv[0]);return 1;}
 const char*path=argv[1];size_t stride=strtoull(argv[2],0,10);int K=atoi(argv[3]);
 const char*mode=argv[4];int pct=atoi(argv[5]);int W=atoi(argv[6]);unsigned seed=(unsigned)strtoul(argv[7],0,10);
 int reps=argc>8?atoi(argv[8]):1; if(W<1)W=1; if(W>K)W=K; if(pct<0)pct=0; if(pct>100)pct=100;
 {const char*d=getenv("ER_DIRECT");if(d&&*d!='0')g_direct=1;}
 struct stat st;if(stat(path,&st)){perror("stat");return 1;}
 int nexp=(int)(st.st_size/stride);
 if(K>nexp){fprintf(stderr,"K>experts\n");return 1;}
 srand(seed);
 int ids[4096];
 if(!strcmp(mode,"seq")){for(int i=0;i<K;i++)ids[i]=i;}
 else{char*used=calloc(nexp,1);int g=0;while(g<K){int e=rand()%nexp;if(!used[e]){used[e]=1;ids[g++]=e;}}free(used);}
 int pre=(int)((long long)K*pct/100);

 printf("{\"file\":\"%s\",\"stride\":%zu,\"experts_total\":%d,\"K\":%d,\"mode\":\"%s\",\"prewarm_pct\":%d,\"prewarm_experts\":%d,\"workers\":%d,\"seed\":%u,\"runs\":[\n",path,stride,nexp,K,mode,pct,pre,W,seed);
 int first=1;
 for(int rep=0;rep<reps;rep++){
   // 1) hard cold: drop whole file, confirm residency 0
   int cf=open(path,O_RDONLY|O_CLOEXEC);if(cf>=0){posix_fadvise(cf,0,0,POSIX_FADV_DONTNEED);close(cf);}
   // 2) prewarm the first `pre` experts (buffered read fills page cache);
   //    prewarm is ALWAYS buffered so it actually populates the cache, even in
   //    the O_DIRECT measurement — then we verify whether O_DIRECT benefits.
   if(pre>0){int save=g_direct;g_direct=0;long long b;int e;read_set(path,stride,ids,pre,W,&b,&e);g_direct=save;}
   // 3) residency now = pre/K ideally; measure how many of the K are resident
   size_t res=0,tot=0;mincore_probe(path,st.st_size,&res,&tot);
   double hit_ratio = (double)pre/K;
   long long rb0=read_rb();
   long long bytes=0;int err=0;
   long long wall=read_set(path,stride,ids,K,W,&bytes,&err);
   long long rb1=read_rb();
   double gib=(double)bytes/(1024.0*1024.0*1024.0),secs=(double)wall/1e9,gbps=secs>0?gib/secs:0;
   printf("%s{\"rep\":%d,\"mode\":\"%s\",\"prewarm_pct\":%d,\"experts_read\":%d,\"bytes\":%lld,\"wall_ms\":%.3f,\"gib_s\":%.3f,\"per_expert_ms\":%.4f,\"phys_read_kib\":%lld,\"mincore_whole_file_res\":%zu,\"mincore_whole_file_tot\":%zu,\"expected_hit_ratio\":%.3f,\"err\":%d}",
     first?"":",\n",rep,mode,pct,K,bytes,(double)wall/1e6,gbps,(double)wall/1e6/K,(rb0>=0&&rb1>=0)?(rb1-rb0)/1024:-1,res,tot,hit_ratio,err);
   first=0;fflush(stdout);
 }
 printf("\n]}\n");return 0;
}
