// directio: 强制走 block 层（不经页缓存）的读/写，用来给 IO 观测做"真值"对照。
//   directio w <path> <MB>    O_DIRECT 写 MB 兆并 fsync
//   directio r <path> <MB>    O_DIRECT 读
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
int main(int c,char**v){
  if(c<4){fprintf(stderr,"usage: %s w|r path MB\n",v[0]);return 2;}
  int wr = v[1][0]=='w'; long mb=atol(v[3]); size_t bs=1<<20;
  void*buf; if(posix_memalign(&buf,4096,bs)){perror("memalign");return 1;}
  memset(buf,0x5a,bs);
  int fl = (wr?O_WRONLY|O_CREAT:O_RDONLY)|O_DIRECT;
  double t0=now(); int fd=open(v[2],fl,0644); if(fd<0){perror("open");return 1;}
  long done=0; for(;done<mb;done++){ ssize_t r = wr? write(fd,buf,bs) : read(fd,buf,bs); if(r!=(ssize_t)bs){fprintf(stderr,"%s at chunk %ld: %s\n",wr?"write":"read",done,strerror(errno));break;} }
  if(wr) fsync(fd); close(fd);
  double dt=now()-t0;
  printf("%s %ld MiB in %.3fs = %.0f MiB/s%s\n", wr?"write":"read", done, dt, done/dt, done<mb?" (PARTIAL)":"");
  return done<mb?1:0;
}
