#include <assert.h>
#include "sched_policy.h"

int main(void) {
    sched_policy p;
    int ids[] = {0,1,2,3,4}, out[5];
    long long missing[] = {0,4096,0,8192,2048};
    assert(sched_policy_parse("cold:4:0:0",4,&p));
    assert(sched_policy_order(&p,ids,missing,5,out));
    assert(out[0]==3 && out[1]==1 && out[2]==4 && out[3]==0 && out[4]==2);
    assert(!sched_policy_parse("cold:0:0:0",4,&p));
    assert(!sched_policy_parse("cold:5:0:0",4,&p));
    assert(!sched_policy_parse("cold:4:2:0",4,&p));
    assert(!sched_policy_parse("cold:4:0:123",4,&p));
    assert(!sched_policy_parse("unknown:4:0:0",4,&p));
    assert(!sched_policy_parse("cold:4:0:0oops",4,&p));
    const char *names[]={"random","cold","hot","1c1h","2c1h","1c2h","proportional"};
    for(int mode=0;mode<7;mode++)for(int bits=0;bits<32;bits++) {
        char text[80];snprintf(text,sizeof text,"%s:4:1:524288",names[mode]);
        assert(sched_policy_parse(text,4,&p));
        for(int i=0;i<5;i++)missing[i]=(bits&(1<<i))?(i+1)*4096:0;
        assert(sched_policy_order(&p,ids,missing,5,out));
        int seen=0;for(int i=0;i<5;i++){assert(out[i]>=0&&out[i]<5);assert(!(seen&(1<<out[i])));seen|=1<<out[i];}
        assert(seen==31);
    }
    assert(sched_policy_parse("maskch:2:0:0",2,&p));
    assert(!sched_policy_order(&p,ids,missing,5,out));
    return 0;
}
