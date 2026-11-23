// Measure the per-read cost breakdown for MoE expert reads:
//   1) fixed startup overhead (open + posix_memalign + close, no read)
//   2) cold per-expert pread latency (FADV_DONTNEED the expert range first)
//   3) mincore probe cost for one expert range vs the whole file
//
// Answers "how expensive is it to probe page-cache residency before reading?"
// and "how much of a read is fixed overhead vs actual data transfer?".
//
// build: clang -O2 -pthread latency_probe.c -o latency_probe
// run:   latency_probe FILE STRIDE [N] [SCOPE]
//   STRIDE = bytes per expert (from the layer manifest); N = samples (default 50)
//   SCOPE  = expert (mincore one expert range) | file (mincore whole file)
//
// Output is JSON with p50/p95/mean in microseconds. Run on the device (Termux);
// no root needed for reading your own files.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

static long long now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (long long)t.tv_sec*1000000000LL+t.tv_nsec;}
static int cmp_ll(const void*a,const void*b){long long x=*(const long long*)a,y=*(const long long*)b;return x<y?-1:x>y;}
static long long pctl(const long long*v,int n,double p){if(n==0)return 0;int i=(int)(p*(n-1));if(i<0)i=0;if(i>=n)i=n-1;return v[i];}
static double mean_ll(const long long*v,int n){if(n==0)return 0;double s=0;for(int i=0;i<n;i++)s+=v[i];return s/n;}

// mmap an [off,off+len) window, mincore it (no dereference), unmap. Returns ns or -1.
static long long mincore_range(int fd,size_t off,size_t len){
  long long t0=now_ns();
  void*m=mmap(NULL,len,PROT_READ,MAP_SHARED,fd,(off_t)off);
  if(m==MAP_FAILED)return -1;
  size_t pg=sysconf(_SC_PAGESIZE),n=(len+pg-1)/pg;
  unsigned char*v=malloc(n);
  if(!v){munmap(m,len);return -1;}
  int rc=mincore(m,len,v);
  munmap(m,len);free(v);
  return rc==0?now_ns()-t0:-1;
}

int main(int argc,char**argv){
  if(argc<3){fprintf(stderr,"usage: %s FILE STRIDE [N] [SCOPE]\n",argv[0]);return 1;}
  const char*path=argv[1];size_t stride=strtoull(argv[2],0,10);
  int N=argc>3?atoi(argv[3]):50; const char*scope=argc>4?argv[4]:"expert";
  if(N<1)N=1;
  struct stat st;if(stat(path,&st)){perror("stat");return 1;}
  int nexp=(int)(st.st_size/stride);
  if(nexp<1){fprintf(stderr,"stride > file\n");return 1;}

  long long *ov=malloc((size_t)N*sizeof(long long));
  long long *rv=malloc((size_t)N*sizeof(long long));
  long long *mv=malloc((size_t)N*sizeof(long long));

  // 1) fixed startup overhead: open + posix_memalign(stride) + close, no data
  for(int i=0;i<N;i++){
    long long t0=now_ns();
    int fd=open(path,O_RDONLY|O_CLOEXEC);
    void*buf=NULL;posix_memalign(&buf,4096,stride);
    close(fd);free(buf);
    ov[i]=now_ns()-t0;
  }
  qsort(ov,(size_t)N,sizeof(long long),cmp_ll);

  // 2) per-expert cold read + 3) mincore(expert range) cost
  for(int i=0;i<N;i++){
    int e=(i*37)%nexp;                 // spread across the file, deterministic
    size_t off=(size_t)e*stride;
    int cf=open(path,O_RDONLY|O_CLOEXEC);
    posix_fadvise(cf,(off_t)off,stride,POSIX_FADV_DONTNEED);
    close(cf);
    int fd=open(path,O_RDONLY|O_CLOEXEC);
    long long pm = strcmp(scope,"file")==0 ? mincore_range(fd,0,st.st_size)
                                           : mincore_range(fd,off,stride);
    mv[i]=pm;
    void*buf=NULL;posix_memalign(&buf,4096,stride);
    long long t0=now_ns();
    size_t rem=stride;off_t o=(off_t)off;char*p=(char*)buf;ssize_t got=0;
    while(rem>0){ssize_t n=pread(fd,p+got,rem,o);if(n<=0){if(errno==EINTR)continue;break;}got+=n;o+=n;rem-=n;}
    rv[i]=now_ns()-t0;
    free(buf);close(fd);
  }
  qsort(rv,(size_t)N,sizeof(long long),cmp_ll);
  qsort(mv,(size_t)N,sizeof(long long),cmp_ll);

  // 4) per-expert HOT read: prewarm the expert into page cache, then time a
  //    buffered re-read (pure page-cache -> user memcpy, no UFS). Shows how much
  //    faster a resident expert is vs a cold one.
  long long *hv=malloc((size_t)N*sizeof(long long));
  long long *hres=malloc((size_t)N*sizeof(long long));
  for(int i=0;i<N;i++){
    int e=(i*37)%nexp; size_t off=(size_t)e*stride;
    // warm it
    { int wf=open(path,O_RDONLY|O_CLOEXEC);void*b=NULL;posix_memalign(&b,4096,stride);
      size_t rem=stride;off_t o=(off_t)off;ssize_t g=0;
      while(rem>0){ssize_t n=pread(wf,(char*)b+g,rem,o);if(n<=0)break;g+=n;o+=n;rem-=n;}
      free(b);close(wf); }
    int fd=open(path,O_RDONLY|O_CLOEXEC);
    { // sanity: count resident pages of this expert (expect full)
      size_t pg=sysconf(_SC_PAGESIZE),n=(stride+pg-1)/pg;
      void*m=mmap(NULL,stride,PROT_READ,MAP_SHARED,fd,(off_t)off);
      if(m!=MAP_FAILED){unsigned char*v=calloc(n,1);if(v&&mincore(m,stride,v)==0){long long r=0;for(size_t j=0;j<n;j++)if(v[j]&1)r++;hres[i]=r;}free(v);munmap(m,stride);}
    }
    void*buf=NULL;posix_memalign(&buf,4096,stride);
    long long t0=now_ns();
    size_t rem=stride;off_t o=(off_t)off;char*p=(char*)buf;ssize_t got=0;
    while(rem>0){ssize_t n=pread(fd,p+got,rem,o);if(n<=0){if(errno==EINTR)continue;break;}got+=n;o+=n;rem-=n;}
    hv[i]=now_ns()-t0;
    free(buf);close(fd);
  }
  qsort(hv,(size_t)N,sizeof(long long),cmp_ll);

  double us=1000.0;
  printf("{\n");
  printf("  \"stride\":%zu,\"experts_total\":%d,\"N\":%d,\"scope\":\"%s\",\n",stride,nexp,N,scope);
  printf("  \"startup_open_malloc_close_us\": {\"p50\":%.3f,\"p95\":%.3f,\"mean\":%.3f},\n",
      pctl(ov,N,.5)/us, pctl(ov,N,.95)/us, mean_ll(ov,N)/us);
  printf("  \"cold_expert_read_us\": {\"p50\":%.1f,\"p95\":%.1f,\"min\":%.1f,\"max\":%.1f,\"mean\":%.1f},\n",
      pctl(rv,N,.5)/us,pctl(rv,N,.95)/us,rv[0]/us,rv[N-1]/us,mean_ll(rv,N)/us);
  printf("  \"hot_expert_read_us\": {\"p50\":%.1f,\"p95\":%.1f,\"min\":%.1f,\"max\":%.1f,\"mean\":%.1f},\n",
      pctl(hv,N,.5)/us,pctl(hv,N,.95)/us,hv[0]/us,hv[N-1]/us,mean_ll(hv,N)/us);
  printf("  \"hot_over_cold_ratio\": %.4f,\n", pctl(rv,N,.5)>0 ? (double)pctl(hv,N,.5)/(double)pctl(rv,N,.5) : 0);
  printf("  \"mincore_us\": {\"p50\":%.3f,\"p95\":%.3f,\"mean\":%.3f},\n",
      pctl(mv,N,.5)/us,pctl(mv,N,.95)/us,mean_ll(mv,N)/us);
  double read_p50 = pctl(rv,N,.5)/us;
  printf("  \"mincore_over_read_pct\": %.2f\n", read_p50>0 ? 100.0*(pctl(mv,N,.5)/us)/read_p50 : 0);
  printf("}\n");
  free(ov);free(rv);free(mv);free(hv);free(hres);
  return 0;
}
