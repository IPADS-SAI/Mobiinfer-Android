// Worker-bubble / scheduling bench for MoE expert reads with partial page-cache
// residency.
//
// Question: under prewarm ratios (30/50/80%), do some workers finish early and
// idle while others still read (bubbles)? Does probing per-expert residency with
// mincore and assigning work (LPT) shrink the makespan / bubble rate?
//
// Two assignment strategies, same set of K experts, same W workers:
//   naive : dynamic pull-queue, workers grab experts in list order (blind to
//           work size). This is what a simple reader pool does.
//   lpt   : probe each expert's residency (mincore -> estimated work: cold = big,
//           hot = small), sort by work desc, greedily assign each to the least
//           loaded worker. Each worker then reads its fixed list.
//
// Metrics per run: makespan (max worker finish span), busy_sum, bubble_ms =
// sum over workers of (makespan - worker_finish_i)... see bubble definition below.
//   makespan      = max_i finish_i
//   busy_sum      = sum_i busy_i          (worker_i time actually reading)
//   bubble        = W*makespan - busy_sum  (total idle time across workers)
//   bubble_rate   = bubble / (W*makespan)
//
// Prewarm: FADV_DONTNEED whole file, then prewarm the first P% of chosen
// experts (buffered) so residency is mixed. All measured reads are buffered
// (page cache is the point). ER_DIRECT makes work uniform and is not used here.
//
// build: clang -O2 -pthread expert_bubble_bench.c -o expert_bubble_bench
// run:   expert_bubble_bench FILE STRIDE K MODE PREWARM_PCT WORKERS SEED [REPS]
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

typedef long long ll;
static ll now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (ll)t.tv_sec*1000000000LL+t.tv_nsec;}
static ll read_rb(void){FILE*f=fopen("/proc/self/io","r");if(!f)return -1;char l[256];ll v=-1;while(fgets(l,sizeof l,f))if(!strncmp(l,"read_bytes:",11)){v=atoll(l+11);break;}fclose(f);return v;}
static ll mincore_resident(const char*path,size_t off,size_t len){
  int fd=open(path,O_RDONLY|O_CLOEXEC);if(fd<0)return -1;
  size_t pg=sysconf(_SC_PAGESIZE),n=(len+pg-1)/pg;
  void*m=mmap(NULL,len,PROT_READ,MAP_SHARED,fd,(off_t)off);
  if(m==MAP_FAILED){close(fd);return -1;}
  unsigned char*v=calloc(n,1);ll res=-1;
  if(v&&mincore(m,len,v)==0){res=0;for(size_t i=0;i<n;i++)if(v[i]&1)res++;}
  free(v);munmap(m,len);close(fd);return res;
}
static ssize_t read_expert(const char*path,size_t off,size_t len){
  int fd=open(path,O_RDONLY|O_CLOEXEC);if(fd<0)return -1;
  void*buf;if(posix_memalign(&buf,4096,len)){close(fd);return -1;}
  size_t rem=len;off_t o=(off_t)off;char*p=(char*)buf;ssize_t got=0;
  while(rem>0){ssize_t n=pread(fd,p+got,rem,o);if(n<=0){if(errno==EINTR)continue;break;}got+=n;o+=n;rem-=n;}
  free(buf);close(fd);return got;
}

// ---- shared context for one timed run ----
typedef struct{
  const char*path; size_t stride; ll t0;
  // naive queue
  int*order; int K; int next;
  pthread_mutex_t mtx;
  // per-worker results (indexed by thread slot)
  ll busy[64]; ll finish[64]; ssize_t bytes[64]; int err[64];
} run_t;

// We implement naive inline via a small generic function below.
typedef struct{run_t*r;int id;const int*list;int n;}task_t;
static void*do_list(void*a){ // read a fixed list; record busy+finish from r->t0
  task_t*t=a;run_t*r=t->r;ll busy=0;ssize_t by=0;int e=0;
  for(int i=0;i<t->n;i++){
    ll s=now_ns();
    ssize_t g=read_expert(r->path,(size_t)t->list[i]*r->stride,r->stride);
    busy+=now_ns()-s;
    if(g<0){e=errno;break;}else by+=g;
  }
  r->busy[t->id]=busy;r->bytes[t->id]=by;r->err[t->id]=e;
  r->finish[t->id]=now_ns()-r->t0;
  return NULL;
}
static void*queue_worker(void*a){
  task_t*t=a;run_t*r=t->r;ll busy=0;ssize_t by=0;int e=0;
  for(;;){
    pthread_mutex_lock(&r->mtx);int i=r->next++;pthread_mutex_unlock(&r->mtx);
    if(i>=r->K)break;
    ll s=now_ns();
    ssize_t g=read_expert(r->path,(size_t)r->order[i]*r->stride,r->stride);
    busy+=now_ns()-s;
    if(g<0){e=errno;break;}else by+=g;
  }
  r->busy[t->id]=busy;r->bytes[t->id]=by;r->err[t->id]=e;
  r->finish[t->id]=now_ns()-r->t0;
  return NULL;
}

// run a strategy; fills makespan/busy/bubble. strat 0=naive queue, 1=LPT lists
static void run_strategy(run_t*r,int W,int strat,const int*wl,ll*makespan,ll*busy_sum,ll*bubble,ssize_t*bytes,int*err){
  for(int i=0;i<W;i++){r->busy[i]=0;r->finish[i]=0;r->bytes[i]=0;r->err[i]=0;}
  pthread_t th[64];task_t tk[64];
  pthread_mutex_init(&r->mtx,NULL);
  r->next=0;
  r->t0=now_ns();
  if(strat==0){
    // dynamic pull-queue, given order (naive=insertion order, cold-first=sorted)
    for(int w=0;w<W;w++){tk[w]=(task_t){.r=r,.id=w,.list=r->order,.n=r->K};pthread_create(&th[w],NULL,queue_worker,&tk[w]);}
  } else {
    // LPT: wh-r->order already sorted desc by work; assign round-robin by least
    // loaded worker (greedy) using work estimation wl[]; build per-worker lists.
    ll load[64]={0}; int cnt[64]={0};
    static int lists[64][4096];
    for(int i=0;i<r->K;i++){int e=r->order[i];int best=0;for(int w=1;w<W;w++)if(load[w]<load[best])best=w;
      lists[best][cnt[best]++]=e;load[best]+=wl?wl[e]:1;}
    for(int w=0;w<W;w++){tk[w]=(task_t){.r=r,.id=w,.list=lists[w],.n=cnt[w]};pthread_create(&th[w],NULL,do_list,&tk[w]);}
  }
  for(int w=0;w<W;w++)pthread_join(th[w],NULL);
  pthread_mutex_destroy(&r->mtx);
  ll mk=0,bs=0;ssize_t b=0;int e=0;
  for(int w=0;w<W;w++){if(r->finish[w]>mk)mk=r->finish[w];bs+=r->busy[w];b+=r->bytes[w];if(r->err[w])e=r->err[w];}
  *makespan=mk;*busy_sum=bs;*bubble=mk*W-bs;*bytes=b;*err=e;
}

int main(int argc,char**argv){
  if(argc<8){fprintf(stderr,"usage: %s FILE STRIDE K MODE PREWARM_PCT WORKERS SEED [REPS]\n",argv[0]);return 1;}
  const char*path=argv[1];size_t stride=strtoull(argv[2],0,10);int K=atoi(argv[3]);
  const char*mode=argv[4];int pct=atoi(argv[5]);int W=atoi(argv[6]);unsigned seed=(unsigned)strtoul(argv[7],0,10);
  int reps=argc>8?atoi(argv[8]):1; if(W<1)W=1; if(W>K)W=K; if(pct<0)pct=0; if(pct>100)pct=100; if(W>64)W=64;
  struct stat st;if(stat(path,&st)){perror("stat");return 1;}
  int nexp=(int)(st.st_size/stride); if(K>nexp){fprintf(stderr,"K>experts\n");return 1;}
  srand(seed);
  int ids[4096];
  if(!strcmp(mode,"seq")){for(int i=0;i<K;i++)ids[i]=i;}
  else{char*used=calloc(nexp,1);int g=0;while(g<K){int e=rand()%nexp;if(!used[e]){used[e]=1;ids[g++]=e;}}free(used);}
  int pre=(int)((ll)K*pct/100);

  run_t r={.path=path,.stride=stride,.K=K};r.order=calloc(K,sizeof(int));
  printf("{\"file\":\"%s\",\"stride\":%zu,\"K\":%d,\"mode\":\"%s\",\"prewarm_pct\":%d,\"workers\":%d,\"seed\":%u,\"runs\":[\n",path,stride,K,mode,pct,W,seed);
  int first=1;
  for(int rep=0;rep<reps;rep++){
    // probe work once (after a cold+prewarm) to build the LPT order; both
    // strategies then run from an identically-prepared cache state.
    int wl[4096]; int cold=0,hot=0;
    {
      int cf=open(path,O_RDONLY|O_CLOEXEC);if(cf>=0){posix_fadvise(cf,0,0,POSIX_FADV_DONTNEED);close(cf);}
      for(int i=0;i<pre;i++){ (void)read_expert(path,(size_t)ids[i]*stride,stride); }
      for(int i=0;i<K;i++){
        ll res=mincore_resident(path,(size_t)ids[i]*stride,stride);
        size_t pg=(stride+ (size_t)sysconf(_SC_PAGESIZE)-1)/(size_t)sysconf(_SC_PAGESIZE);
        int miss=(res<0)?(int)pg:(int)(pg-(size_t)res);
        wl[ids[i]]=miss; if(miss> (int)pg/2)cold++;else hot++;
      }
    }
    for(int i=0;i<K;i++)r.order[i]=ids[i];
    for(int i=1;i<K;i++){int key=r.order[i],j=i-1;while(j>=0&&wl[r.order[j]]<wl[key]){r.order[j+1]=r.order[j];j--;}r.order[j+1]=key;}

    // prepare identical cache state before each strategy
    ll mk0,bs0,bl0,mk1,bs1,bl1; ssize_t by0,by1; int e0,e1;
    { int cf=open(path,O_RDONLY|O_CLOEXEC);if(cf>=0){posix_fadvise(cf,0,0,POSIX_FADV_DONTNEED);close(cf);}
      for(int i=0;i<pre;i++){ (void)read_expert(path,(size_t)ids[i]*stride,stride); } }
    // naive: keep original insertion order in r.order, run dynamic queue
    for(int i=0;i<K;i++)r.order[i]=ids[i];
    run_strategy(&r,W,0,wl,&mk0,&bs0,&bl0,&by0,&e0); // naive (queue, orig order)

    // cold-first: same dynamic queue, but order sorted by work DESC (cold first)
    { int cf=open(path,O_RDONLY|O_CLOEXEC);if(cf>=0){posix_fadvise(cf,0,0,POSIX_FADV_DONTNEED);close(cf);}
      for(int i=0;i<pre;i++){ (void)read_expert(path,(size_t)ids[i]*stride,stride); } }
    for(int i=0;i<K;i++)r.order[i]=ids[i];
    for(int i=1;i<K;i++){int key=r.order[i],j=i-1;while(j>=0&&wl[r.order[j]]<wl[key]){r.order[j+1]=r.order[j];j--;}r.order[j+1]=key;}
    ll mk2,bs2,bl2; ssize_t by2; int e2;
    run_strategy(&r,W,0,wl,&mk2,&bs2,&bl2,&by2,&e2); // cold-first (queue, sorted)

    // LPT: static assignment, sorted order
    { int cf=open(path,O_RDONLY|O_CLOEXEC);if(cf>=0){posix_fadvise(cf,0,0,POSIX_FADV_DONTNEED);close(cf);}
      for(int i=0;i<pre;i++){ (void)read_expert(path,(size_t)ids[i]*stride,stride); } }
    run_strategy(&r,W,1,wl,&mk1,&bs1,&bl1,&by1,&e1); // LPT

    double rate0=(double)bl0/((double)W*mk0)*100.0, rate1=(double)bl1/((double)W*mk1)*100.0, rate2=(double)bl2/((double)W*mk2)*100.0;
    printf("%s{\"rep\":%d,\"cold\":%d,\"hot\":%d,"
      "\"naive\":{\"makespan_ms\":%.3f,\"bubble_ms\":%.3f,\"bubble_rate_pct\":%.2f},"
      "\"coldfirst\":{\"makespan_ms\":%.3f,\"bubble_ms\":%.3f,\"bubble_rate_pct\":%.2f},"
      "\"lpt\":{\"makespan_ms\":%.3f,\"bubble_ms\":%.3f,\"bubble_rate_pct\":%.2f},"
      "\"cf_speedup\":%.4f,\"lpt_speedup\":%.4f,\"err\":%d}",
      first?"":",\n",rep,cold,hot,
      mk0/1e6,bl0/1e6,rate0,
      mk2/1e6,bl2/1e6,rate2,
      mk1/1e6,bl1/1e6,rate1,
      (double)mk0/(double)mk2,(double)mk0/(double)mk1,(e0?e0:(e2?e2:e1)));
    first=0;fflush(stdout);
  }
  free(r.order);
  printf("\n]}\n");return 0;
}
