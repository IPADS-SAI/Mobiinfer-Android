#!/system/bin/sh
# on-device helper, run as uid shell.  Provides the aggregates the app UID itself
# cannot read (Android 12+ SURFACING / uid_change view strips /proc/<pid>/{io,status}),
# by going through system services that are readable from adb:
#   sp_helper.sh net  <pkg>            -> "rxCum txCum" bytes since boot for that package
#   sp_helper.sh rss  <pkg>            -> total RSS KiB over the package's processes
#   sp_helper.sh sample <pkg> <secs> <csv>   per-second rows: rss, cumulative rx/tx, app CPU jiffies
# NOTE: FILE*/fgets/fscanf on kernel & binder-backed files can yield zeros under the
# uid_change view -> we use read(2) (awk/sed are fine) and `pidof` (not uid scans).
export PATH=/system/bin:$PATH
D=${D:-$(cd "$(dirname "$0")" && pwd)}

uid_of() { dumpsys package "$1" 2>/dev/null | awk '/appId=/{sub(/.*=/, ""); print; exit}'; }
pids_of() { pidof $(a=; for s in "" :push :remote :sandbox :support :msf :service :background :mini:0; do a="$a $1$s"; done; echo $a) 2>/dev/null; }

net_of() {   # netstats detail: per-uid history, take the latest bucket, sum over sets/networks
  uid=$(uid_of "$1")
  dumpsys netstats detail 2>/dev/null | tr -d '\000' > $D/.net.$$
  cat > $D/.net.awk <<'EOS'
$0 ~ /ident=\[/ {
    s2 = $0; sub(/.*\] /, "", s2)          # strip "... }] " -> "uid=N set=... tag=..."
    if (index(s2, "uid=" want " ") == 1) { inhdr = 1; h++; next }
}
inhdr && /st=/ {
    nst++; rb = 0; tb = 0; st = 0
    for (i = 1; i <= NF; i++) { split($i, kv, "=")
        if (kv[1] == "st") st = kv[2] + 0; else if (kv[1] == "rb") rb = kv[2] + 0; else if (kv[1] == "tb") tb = kv[2] + 0 }
    if (!last || st >= last) { if (st > last) { cr = 0; ct = 0 } last = st; cr += rb; ct += tb }
    next
}
inhdr { rx += cr; tx += ct; cr = ct = 0; last = 0; inhdr = 0; nend++; }
END { if (ENVIRON["HELPER_DBG"]) printf "hdr=%d st=%d end=%d\n", h, nst, nend > "/dev/stderr"; printf "%d %d\n", rx + cr, tx + ct }
EOS
  awk -v want="$uid" -f $D/.net.awk $D/.net.$$
  rm -f $D/.net.$$ $D/.net.awk
}

rss_of() {   # dumpsys meminfo "Total RSS by process" block + pidof list of the package
  dumpsys meminfo 2>/dev/null > $D/.mem.$$
  mine=" $(pids_of "$1") "
  awk -v mine="$mine" '
    /^Total RSS by process:/ { f = 1; next }
    f && /^[^ ]/ { exit }
    f && match($0, "[(]pid [0-9]+") {
        pid = substr($0, RSTART + 5, RLENGTH - 5) + 0
        if (index(mine, " " pid " ") == 0) next
        v = $0; sub(/:.*/, "", v); gsub(/,/, "", v); s += v }
    END { printf "%d\n", s }' $D/.mem.$$
  rm -f $D/.mem.$$
}

cpu_of() {   # sum utime+stime of the package's processes (read(2) via cat; fields 14,15 of /proc/pid/stat)
  t=0
  for p in $(pids_of "$1"); do
    s=$(cat /proc/$p/stat 2>/dev/null) || continue
    s=${s##*) }
    set -- $s
    [ $# -ge 16 ] || continue
    t=$(( t + ${12} + ${13} ))
  done
  echo $t
}

sample_case() {
    local pkg out secs i r c rx tx
    pkg="$1"; secs="$2"; out="$3"
    echo "ts,elapsed,rssKiB,rxCumBytes,txCumBytes,cpuJiffies,memAvailKiB,swapUsedKiB" > "$out"
    t0=$(date +%s); i=0
    while :; do
      now=$(date +%s); el=$(( now - t0 ))
      [ "$el" -ge "$secs" ] && break
      i=$((i+1))
      r=$(rss_of "$pkg")
      set -- $(net_of "$pkg"); rx=$1; tx=$2
      c=$(cpu_of "$pkg")
      # elapsed 记真实秒数（不是迭代次数）：dumpsys 慢时也不会把时间轴算错
      ma=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null); ma=${ma:-0}
      sf=$(awk '/^SwapFree:/{print $2}' /proc/meminfo 2>/dev/null); sf=${sf:-0}
      st=$(awk '/^SwapTotal:/{print $2}' /proc/meminfo 2>/dev/null); st=${st:-0}
      su=$(( st > sf ? st - sf : 0 ))   # KiB
      echo "$(date +%s%3N),$(( $(date +%s) - t0 )),$r,$rx,$tx,$c,$ma,$su" >> "$out"
      sleep 1
    done
}

case "$1" in
  net) net_of "$2" ;;
  rss) rss_of "$2" ;;
  cpu) cpu_of "$2" ;;
  sample) sample_case "$2" "$3" "$4" ;;
esac
