import subprocess,sys,json,statistics as st
f=sys.argv[1]
def run(K,W,pct,seed,reps):
    out=subprocess.run(["./repro_sched",f,str(K),str(W),str(pct),str(seed),str(reps)],capture_output=True,text=True).stdout
    return [json.loads(l) for l in out.splitlines()]
def cell(K,W,pct,seeds,reps):
    acc={}
    for seed in seeds:
        for r in run(K,W,pct,seed,reps):
            acc.setdefault(r["policy"],[]).append(r["ms"])
    return {k:st.mean(v) for k,v in acc.items()}
seeds=(1,7,42,123,999,2024,31337)
print("K  W pct | random  cold   hot   alt   | alt vs cold")
for K,W in ((8,4),(25,4),(25,8)):
  for pct in (50,80):
    m=cell(K,W,pct,seeds,4)
    alt_vs_cold=(m["cold"]-m["alternating"])/m["cold"]*100
    print("%2d %d %3d | %6.3f %6.3f %6.3f %6.3f | %+5.1f%%"%(
      K,W,pct,m["random"],m["cold"],m["hot"],m["alternating"],alt_vs_cold))
