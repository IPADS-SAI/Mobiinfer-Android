// Weighted scheduling bench for MoE expert reads: compare dispatch orders by the
// weighted completion time sum C_w = sum_i p_i * C_i (p_i = urgency weight,
// C_i = wall time when expert i finishes). This models an online decode step:
// some experts are needed immediately (high p), others a few steps later (low p).
//
// Preps: FADV_DONTNEED whole file, prewarm the first P% of chosen experts so
// work is mixed (cold = big p_i... no, cold = big WORK). Work w_i is measured
// by mincore (missing pages). Urgency p_i is synthetic (random or assigned).
//
// Strategies (all dynamic pull-queue W workers; only the QUEUE ORDER differs):
//   naive   : original order
//   cold    : sort by work DESC (longest first)          -> minimizes makespan
//   hot     : sort by work ASC (shortest first)          -> SRPT
//   wsrpt   : sort by work/urgency DESC (Smith's rule)   -> minimizes sum p_i C_i
//
// Reports per strategy: makespan, sum(p_i*C_i), mean completion, and the
// completion time of the single most-urgent expert (critical arrival).
//
// build: clang -O2 -pthread expert_sched_bench.c -o expert_sched_bench
// run:   expert_sched_bench FILE STRIDE K PREWARM_PCT WORKERS SEED [REPS]
//   urgency model: p_i in {1,2,4,8} ~ geometric (some very urgent). WEIGHT_MODE env:
//     urg  (default): random geometric urgency independent of hot/cold
//     corr          : urgent experts tend to be the hot ones
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

typedef struct{
  const char*path;size_t stride;
  int*order;int K;int next;
  ll t0;
  ll comp[4096];      // completion ns per expert id (index by expert id)
  pthread_mutex_t mtx;
  ll busy[64];ll finish[64];int err[64];
}ctx_t;

typedef struct{ctx_t*c;int id;}tk_t;
static void*queue_worker(void*a){
  tk_t*t=a;ctx_t*c=t->c;ll busy=0;int e=0;
  for(;;){
    pthread_mutex_lock(&c->mtx);int i=c->next++;pthread_mutex_unlock(&c->mtx);
    if(i>=c->K)break;
    int ex=c->order[i];
    ll s=now_ns();
    ssize_t g=read_expert(c->path,(size_t)ex*c->stride,c->stride);
    busy+=now_ns()-s;
    if(g<0){e=errno;break;}
    c->comp[ex]=now_ns()-c->t0;   // completion time of this expert
  }
  c->busy[t->id]=busy;c->finish[t->id]=now_ns()-c->t0;c->err[t->id]=e;
  return NULL;
}

// run one strategy given an order; returns makespan, fills comp[] (indexed by expert id)
static ll run_order(ctx_t*c,int W){
  for(int i=0;i<W;i++){c->busy[i]=0;c->finish[i]=0;c->err[i]=0;}
  pthread_mutex_init(&c->mtx,NULL);c->next=0;
  c->t0=now_ns();
  pthread_t th[64];tk_t tk[64];
  for(int w=0;w<W;w++){tk[w]=(tk_t){.c=c,.id=w};pthread_create(&th[w],NULL,queue_worker,&tk[w]);}
  for(int w=0;w<W;w++)pthread_join(th[w],NULL);
  pthread_mutex_destroy(&c->mtx);
  ll mk=0;for(int w=0;w<W;w++)if(c->finish[w]>mk)mk=c->finish[w];
  return mk;
}

static void reset_cache(const char*path,int*ids,int pre,size_t stride){
  int cf=open(path,O_RDONLY|O_CLOEXEC);if(cf>=0){posix_fadvise(cf,0,0,POSIX_FADV_DONTNEED);close(cf);}
  for(int i=0;i<pre;i++)(void)read_expert(path,(size_t)ids[i]*stride,stride);
}

int main(int argc,char**argv){
  if(argc<7){fprintf(stderr,"usage: %s FILE STRIDE K PREWARM_PCT WORKERS SEED [REPS]\n",argv[0]);return 1;}
  const char*path=argv[1];size_t stride=strtoull(argv[2],0,10);int K=atoi(argv[3]);
  int pct=atoi(argv[4]);int W=atoi(argv[5]);unsigned seed=(unsigned)strtoul(argv[6],0,10);
  int reps=argc>7?atoi(argv[7]):1; if(W<1)W=1; if(W>K)W=K; if(pct<0)pct=0;if(pct>100)pct=100;if(W>64)W=64;
  const char*wm=getenv("WEIGHT_MODE");int corr=wm&&!strcmp(wm,"corr");
  struct stat st;if(stat(path,&st)){perror("stat");return 1;}
  int nexp=(int)(st.st_size/stride);if(K>nexp)K=nexp;
  srand(seed);
  int ids[4096];{char*used=calloc(nexp,1);int g=0;while(g<K){int e=rand()%nexp;if(!used[e]){used[e]=1;ids[g++]=e;}}free(used);}
  int pre=(int)((ll)K*pct/100);

  ctx_t c={.path=path,.stride=stride,.K=K};c.order=calloc(K,sizeof(int));
  printf("{\"file\":\"%s\",\"stride\":%zu,\"K\":%d,\"prewarm_pct\":%d,\"workers\":%d,\"seed\":%u,\"weight_mode\":\"%s\",\"runs\":[\n",
    path,stride,K,pct,W,seed,corr?"corr":"urg");
  int first=1;
  for(int rep=0;rep<reps;rep++){
    // work + urgency
    int wl[4096];double urg[4096];
    reset_cache(path,ids,pre,stride);
    size_t pg=(stride+(size_t)sysconf(_SC_PAGESIZE)-1)/(size_t)sysconf(_SC_PAGESIZE);
    for(int i=0;i<K;i++){
      int ex=ids[i];ll res=mincore_resident(path,(size_t)ex*stride,stride);
      int miss=(res<0)?(int)pg:(int)(pg-(size_t)res);wl[ex]=miss;
    }
    // urgency: geometric 1,2,4,8,... capped; corr mode ties urgency to cold/hot
    for(int i=0;i<K;i++){
      int ex=ids[i];
      if(corr){ // urgent ~ hot: urgency inversely related to work
        urg[ex]= 1.0 + 8.0*(1.0-(double)wl[ex]/(double)pg); // hot->8, cold->1
      } else {
        int g=1;while((rand()&1)&&g<16)g*=2; urg[ex]=(double)g; // 1,2,4,8
      }
    }
    // helper sorts
    double kw[4096],kratio[4096];
    for(int i=0;i<K;i++){int ex=ids[i];kw[ex]=(double)wl[ex];kratio[ex]= (urg[ex]>0)? (double)wl[ex]/urg[ex] : (double)wl[ex];}

    // strategy runners
    struct{const char*name;int desc;double*key;} strats[4]={
      {"naive",0,NULL},{"cold",1,kw},{"hot",0,kw},{"wsrpt",1,kratio}};
    printf("%s{\"rep\":%d,",first?"":",\n",rep);first=0;
    int sf=1;
    for(int si=0;si<4;si++){
      reset_cache(path,ids,pre,stride);
      if(strats[si].key==NULL){for(int i=0;i<K;i++)c.order[i]=ids[i];}
      else {
        // build order by key
        for(int i=0;i<K;i++)c.order[i]=ids[i];
        for(int i=1;i<K;i++){int k=c.order[i],j=i-1;while(j>=0&&(strats[si].desc?strats[si].key[c.order[j]]<strats[si].key[k]:strats[si].key[c.order[j]]>strats[si].key[k])){c.order[j+1]=c.order[j];j--;}c.order[j+1]=k;}
      }
      ll mk=run_order(&c,W);
      // weighted completion sum
      double cw=0,mean_c=0;int maxurg_ex=ids[0];double maxurg=urg[ids[0]];
      for(int i=0;i<K;i++){int ex=ids[i];double ci=(double)c.comp[ex];cw+=urg[ex]*ci;mean_c+=ci;}
      for(int i=0;i<K;i++)if(urg[ids[i]]>maxurg){maxurg=urg[ids[i]];maxurg_ex=ids[i];}
      double crit=(double)c.comp[maxurg_ex];
      printf("%s\"%s\":{\"makespan_ms\":%.3f,\"weighted_C_ms\":%.3f,\"mean_C_ms\":%.3f,\"crit_arrival_ms\":%.3f}",
        sf?"":",",strats[si].name,mk/1e6,cw/1e6,(mean_c/K)/1e6,crit/1e6);
      sf=0;
    }
    printf("}");
  }
  free(c.order);
  printf("\n]}\n");return 0;
}
