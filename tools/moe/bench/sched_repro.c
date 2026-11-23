// Independent re-implementation of the cold/hot dispatch scheduling probe.
// Deliberately different structure from the original probe.c:
//   - C11 atomics for a lock-free-ish task counter (plus a small mutex for the
//     alternating/cap logic), persistent worker threads started by a barrier,
//   - per-expert completion timestamps recorded, makespan = last completion,
//   - before EVERY policy: FADV_DONTNEED, re-prewarm the SAME hot membership,
//     per-expert mincore to build the real cold set, verify read_bytes delta.
// Goal: reproduce (a) cold-first is not always best once hot-copy work and
// overlap matter; (b) results are honest (validated residency + physical reads).
//
// build: clang -O2 -Wall -Wextra -pthread repro_sched.c -o repro_sched
// run:   repro_sched FILE K W PCT SEED REPS
//   prints one JSON line per (rep, policy).
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "sched_policy.h"

#define MAXE 256
#define STRIDE 1351680

static int g_fd, g_K, g_W;
static int g_ids[MAXE];
static long long g_miss[MAXE];   // missing bytes per expert (0 = hot)
static int g_cold[MAXE];
static int g_order[MAXE];
static int g_taken[MAXE];        // per-expert claimed flag
static int g_done;               // number of experts completed
static long long g_begin[MAXE], g_finish[MAXE];
static long long g_t0;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cv = PTHREAD_COND_INITIALIZER;
static pthread_barrier_t g_start, g_end;
static int g_cap;                 // max concurrent cold readers (<=W)
static int g_active_cold;
static int g_stop;
static int g_chunk, g_release, g_hotleft;

static long long now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (long long)t.tv_sec*1000000000LL+t.tv_nsec;}
static long long read_bytes(void){FILE*f=fopen("/proc/self/io","r");if(!f)return -1;char l[256];long long v=-1;while(fgets(l,sizeof l,f))if(!strncmp(l,"read_bytes:",11)){v=atoll(l+11);break;}fclose(f);return v;}

static void read_all(void*buf,int ex){
  char*p=buf;size_t off=0;
  while(off<STRIDE){size_t size=STRIDE-off;if(g_chunk && size>(size_t)g_chunk)size=g_chunk;ssize_t r=pread(g_fd,p+off,size,(off_t)ex*STRIDE+off);if(r<0){if(errno==EINTR)continue;perror("pread");abort();}if(r==0)abort();off+=(size_t)r;}
}

static void*worker(void*arg){
  (void)arg; void*buf=malloc(STRIDE); if(!buf)abort(); memset(buf,1,STRIDE);
  for(;;){
    pthread_barrier_wait(&g_start);
    if(g_stop) break;
    for(;;){
      int ex=-1;
      pthread_mutex_lock(&g_mu);
      while(1){
        // scan the order for the first not-yet-taken expert; if it is cold and
        // the cold-cap is reached, KEEP SCANNING (skip it) so hot jobs and other
        // admissible colds can still run — we come back for it later.
        int found=-1;
        for(int i=0;i<g_K;i++){
          int e=g_order[i];
          if(g_taken[e])continue;
          if(g_cold[e]&&g_active_cold>=(g_release && !g_hotleft ? g_W : g_cap))continue;   // blocked cold, skip
          found=e;break;
        }
        if(found>=0){ex=found;g_taken[ex]=1;if(g_cold[ex])g_active_cold++;break;}
        if(g_done>=g_K)break;                            // nothing left at all
        pthread_cond_wait(&g_cv,&g_mu);                  // only blocked colds remain
      }
      pthread_mutex_unlock(&g_mu);
      if(ex<0)break;
      g_begin[ex]=now_ns()-g_t0;
      read_all(buf,ex);
      g_finish[ex]=now_ns()-g_t0;
      pthread_mutex_lock(&g_mu);
      if(g_cold[ex])g_active_cold--;else g_hotleft--;
      g_done++;
      pthread_cond_broadcast(&g_cv);
      pthread_mutex_unlock(&g_mu);
    }
    pthread_barrier_wait(&g_end);
  }
  free(buf);return NULL;
}

static uint32_t g_rng;
static uint32_t rnd(void){g_rng^=g_rng<<13;g_rng^=g_rng>>17;g_rng^=g_rng<<5;return g_rng;}
static void shuffle(int*a,int n){for(int i=n-1;i>0;i--){int j=rnd()%(i+1);int t=a[i];a[i]=a[j];a[j]=t;}}

enum {P_RANDOM,P_COLD,P_HOT,P_ALT,P_CAP1,P_CAP2,P_CAP4};
static const char*pname(int p){const char*n[]={"random","cold","hot","alternating","cap1","cap2","cap4"};return n[p];}

int main(int argc,char**argv){
  if(argc!=7){fprintf(stderr,"usage: %s FILE K W PCT SEED REPS\n",argv[0]);return 1;}
  const char*path=argv[1]; g_K=atoi(argv[2]); g_W=atoi(argv[3]);
  int pct=atoi(argv[4]),seed=atoi(argv[5]),reps=atoi(argv[6]);
  if(g_K<1||g_K>MAXE||g_W<1||g_W>16||pct<0||pct>100||reps<1)return 1;
  const char *spec=getenv("SCHED_POLICY"); sched_policy custom;
  if(spec && !sched_policy_parse(spec,g_W,&custom)){fprintf(stderr,"invalid SCHED_POLICY\n");return 1;}
  g_fd=open(path,O_RDONLY); if(g_fd<0){perror("open");return 1;}
  struct stat st; if(fstat(g_fd,&st)||st.st_size!=(off_t)MAXE*STRIDE){fprintf(stderr,"bad file size\n");return 1;}
  if(posix_fadvise(g_fd,0,0,POSIX_FADV_RANDOM))return 1;
  g_rng=seed?seed:1;

  // choose K experts, and an INDEPENDENT random hot membership of size floor(K*pct/100)
  int pool[MAXE];for(int i=0;i<MAXE;i++)pool[i]=i;shuffle(pool,MAXE);
  for(int i=0;i<g_K;i++)g_ids[i]=pool[i];
  int warm[MAXE]={0}; for(int i=0;i<g_K*pct/100;i++)warm[pool[i]]=1;   // membership by pool, not queue order
  shuffle(g_ids,g_K);                                                   // queue order independent of membership

  pthread_barrier_init(&g_start,NULL,g_W+1);pthread_barrier_init(&g_end,NULL,g_W+1);
  pthread_t th[16];for(int i=0;i<g_W;i++)if(pthread_create(th+i,NULL,worker,NULL))return 1;

  void*tmp=malloc(STRIDE);if(!tmp)return 1;memset(tmp,0,STRIDE);
  long pg=sysconf(_SC_PAGESIZE);if(pg<=0||STRIDE%pg)return 1;size_t np=STRIDE/pg;unsigned char*vec=malloc(np);if(!vec)return 1;

  const char *profile=getenv("SCHED_PROFILE");
  const char *layout=getenv("SCHED_LAYOUT");if(!layout)layout="prefix";
  if(strcmp(layout,"prefix") && strcmp(layout,"spread")){fprintf(stderr,"bad layout\n");return 1;}
  int target[MAXE];for(int e=0;e<MAXE;e++)target[e]=warm[e]?(int)np:0;
  if(profile){
    FILE *f=fopen(profile,"r");int count,pages;
    if(!f){perror("profile");return 1;}
    if(fscanf(f,"%d %d",&count,&pages)!=2||count!=MAXE||pages!=(int)np){fprintf(stderr,"profile geometry mismatch; explicit conversion required\n");fclose(f);return 1;}
    for(int e=0;e<MAXE;e++)if(fscanf(f,"%d",target+e)!=1||target[e]<0||target[e]>(int)np){fclose(f);return 1;}
    char extra;if(fscanf(f," %c",&extra)==1){fclose(f);return 1;}fclose(f);
  }
  unsigned char *expected=calloc(MAXE,np);if(!expected)return 1;
  for(int e=0;e<MAXE;e++)for(size_t j=0;j<np;j++)
    expected[e*np+j]=!strcmp(layout,"prefix") ? j<(size_t)target[e] :
      ((j+1)*target[e]/np != j*target[e]/np);
  int pols[]={P_RANDOM,P_COLD,P_HOT,P_ALT,P_CAP1,P_CAP2,P_CAP4};
  const int NP=spec?1:(int)(sizeof(pols)/sizeof(pols[0]));
  for(int rep=0;rep<reps;rep++){
    int porder[16];for(int i=0;i<NP;i++)porder[i]=i;shuffle(porder,NP);   // randomize policy order
    for(int s=0;s<NP;s++){
      int pol=pols[porder[s]];
      // rebuild identical cache state: drop whole file, prewarm exact hot membership
      g_chunk=0;
      if(posix_fadvise(g_fd,0,0,POSIX_FADV_DONTNEED))return 1;
      for(int i=0;i<g_K;i++){
        int e=g_ids[i];
        if(target[e]==(int)np){read_all(tmp,e);continue;}
        for(size_t j=0;j<np;j++)if(expected[e*np+j]){
          size_t got=0;while(got<(size_t)pg){ssize_t n=pread(g_fd,(char*)tmp+got,pg-got,(off_t)e*STRIDE+j*pg+got);
            if(n<0&&errno==EINTR)continue;
            if(n<=0){fprintf(stderr,"prewarm failed\n");return 1;}got+=(size_t)n;}
        }
      }
      // probe real residency -> miss bytes, cold flag; validate vs intended
      long long probe0=now_ns();long long missing=0;int valid=1;
      for(int i=0;i<g_K;i++){
        int e=g_ids[i];void*m=mmap(NULL,STRIDE,PROT_READ,MAP_SHARED,g_fd,(off_t)e*STRIDE);
        if(m==MAP_FAILED)abort();
        if(mincore(m,STRIDE,vec))abort();
        long long res=0;for(size_t j=0;j<np;j++)if(vec[j]&1)res++;
        munmap(m,STRIDE);
        long long missb=(long long)(np-res)*pg;g_miss[e]=missb;g_cold[e]=missb>0;missing+=missb;
        // Exact page-position check, not just an aggregate resident count.
        for(size_t j=0;j<np;j++)if((vec[j]&1)!=expected[e*np+j])valid=0;
      }
      long long probe_ns=now_ns()-probe0;
      long long plan0=now_ns();
      // build order
      memcpy(g_order,g_ids,g_K*sizeof(int));
      if(pol==P_COLD||pol==P_HOT||pol==P_CAP1||pol==P_CAP2||pol==P_CAP4){
        for(int i=1;i<g_K;i++){int x=g_order[i],j=i-1;
          int desc=(pol!=P_HOT);
          while(j>=0&&(desc?g_miss[g_order[j]]<g_miss[x]:g_miss[g_order[j]]>g_miss[x])){g_order[j+1]=g_order[j];j--;}
          g_order[j+1]=x;}
      } else if(pol==P_ALT){
        int c[MAXE],h[MAXE],nc=0,nh=0,n=0;
        for(int i=0;i<g_K;i++){if(g_cold[g_ids[i]])c[nc++]=g_ids[i];else h[nh++]=g_ids[i];}
        for(int i=0;i<nc||i<nh;i++){if(i<nc)g_order[n++]=c[i];if(i<nh)g_order[n++]=h[i];}
      }
      g_cap = pol==P_CAP1?1:pol==P_CAP2?2:pol==P_CAP4?4:g_W; if(g_cap>g_W)g_cap=g_W;
      if(spec){
        if(!sched_policy_order(&custom,g_ids,g_miss,g_K,g_order)){fprintf(stderr,"policy does not match batch\n");return 1;}
        g_cap=custom.cap;g_release=custom.release;g_chunk=custom.chunk;
      }
      long long plan_ns=now_ns()-plan0;
      memset(g_taken,0,sizeof g_taken);g_done=0;g_active_cold=0;
      g_hotleft=0;for(int i=0;i<g_K;i++)g_hotleft+=!g_cold[g_ids[i]];
      memset(g_finish,0,sizeof g_finish);
      long long rb0=read_bytes();g_t0=now_ns();
      pthread_barrier_wait(&g_start);pthread_barrier_wait(&g_end);
      long long elapsed=now_ns()-g_t0;long long rb1=read_bytes();long long phys=rb1-rb0;
      if(rb0<0||rb1<0||phys<missing)valid=0;
      long long mk=0;for(int i=0;i<g_K;i++)if(g_finish[g_ids[i]]>mk)mk=g_finish[g_ids[i]];
      printf("{\"profile\":%s,\"layout\":\"%s\",\"K\":%d,\"W\":%d,\"pct\":%d,\"seed\":%d,\"plan_ms\":%.6f,\"rep\":%d,\"policy\":\"%s\",\"cap\":%d,\"valid\":%d,\"missing\":%lld,\"read_bytes\":%lld,\"extra_read_bytes\":%lld,"
             "\"probe_ms\":%.4f,\"ms\":%.4f,\"wall_ms\":%.4f,\"jobs\":[",
             profile?"true":"false",layout,g_K,g_W,pct,seed,plan_ns/1e6,rep,spec?spec:pname(pol),g_cap,valid,missing,phys,phys-missing,probe_ns/1e6,mk/1e6,elapsed/1e6);
      for(int i=0;i<g_K;i++){int e=g_ids[i];printf("%s[%d,%d,%.4f,%.4f]",i?",":"",e,g_cold[e],g_begin[e]/1e6,g_finish[e]/1e6);}
      printf("],\"resident_pages\":[");
      for(int i=0;i<g_K;i++)printf("%s%lld",i?",":"",(long long)np-g_miss[g_ids[i]]/pg);
      printf("]}\n");fflush(stdout);
      usleep(5000);
    }
  }
  g_stop=1;pthread_barrier_wait(&g_start);for(int i=0;i<g_W;i++)pthread_join(th[i],NULL);
  free(expected);free(tmp);free(vec);close(g_fd);return 0;
}
