#!/bin/bash
# genshin/launch.sh [--enter] [--name 名字]
#
# 拉起原神并（可选）自动进入游戏世界。
#
# ⚠️ 关键：原神是**横屏**运行，逻辑坐标是 3168x1440（不是竖屏 1440x3168）。
#    用 wm size 得到的 1440x3168 是物理尺寸，方向不同；给 input 的坐标必须是
#    当前方向的逻辑尺寸。见 `dumpsys display | grep logicalFrame`。
#
# ⚠️ 原神整个游戏渲染在**单个 SurfaceView** 里，uiautomator 只看到
#    SurfaceView(desc="Game view")，看不到任何按钮/文字。所以：
#      - 游戏内 UI 无法"解析"，只能按坐标点；
#      - 例外：**名字输入框**被点时 Unity 会调起原生 EditText + 确定按钮，
#        那层是能 dump 的（本脚本就靠这个自动取名）。
#
# 坐标（横屏 3168x1440，实测于 2026-11-06）：
#   "点击进入" 横幅      (1584, 1330)
#   名字输入框           (1479, 1256)   ← 对应原生 EditText [158,1166][2800,1346]
#   原生"确定"           (2976, 1256)   ← 对应原生 Button  [2800,1160][3152,1352]
#   游戏内"确认"         (2820, 1300)
source "$(cd "$(dirname "$0")/../shared" && pwd)/lib.sh"
PKG=com.miHoYo.Yuanshen
ENTER=0; NAME="荧"
while [ $# -gt 0 ]; do case "$1" in
  --enter) ENTER=1; shift ;;
  --name) NAME=$2; shift 2 ;;
  *) echo "未知参数 $1"; exit 2 ;;
esac; done

fa_install_tools
fa_wake

# 屏幕方向检查：给 input 的坐标必须是"逻辑"尺寸
logical=$(sh_ "dumpsys display 2>/dev/null | grep -o 'logicalFrame=Rect(0, 0 - [0-9]*, [0-9]*)' | head -1" | tr -d '\r')
case "$logical" in
  *3168,\ 1440*) LAND=1; W=3168; H=1440 ;;
  *1440,\ 3168*) LAND=0; W=1440; H=3168 ;;
  *) LAND=1; W=3168; H=1440 ;;   # 兜底按横屏
esac
log "当前逻辑分辨率: ${W}x${H}（原神应始终横屏）"

sh_ "export PATH=/system/bin:\$PATH; am force-stop $PKG" >/dev/null 2>&1
sleep 2
dump_native() {
  sh_ 'export PATH=/system/bin:$PATH; uiautomator dump /data/local/tmp/gs.xml >/dev/null 2>&1'
  sh_ 'cat /data/local/tmp/gs.xml 2>/dev/null'
}

log "拉起原神 ..."
sh_ "export PATH=/system/bin:\$PATH; monkey -p $PKG -c android.intent.category.LAUNCHER 1" >/dev/null 2>&1

# 等 SurfaceView 出现（游戏窗口起来了）
for i in $(seq 1 90); do
  sleep 2
  sh_ 'export PATH=/system/bin:$PATH; uiautomator dump /data/local/tmp/gs.xml >/dev/null 2>&1'
  sh_ 'cat /data/local/tmp/gs.xml 2>/dev/null' | grep -q 'Game view' && { log "游戏窗口已就绪（$((i*2))s）"; break; }
  [ $i -eq 90 ] && { log "ERROR: 游戏窗口未出现"; exit 1; }
done

if [ "$ENTER" != 1 ]; then
  log "停在标题页。要自动进入世界请加 --enter"
  exit 0
fi

# 进世界流程（进度存在服务器，重进只需点"点击进入"，不再走选角/取名）
#
# 时序（实测）：
#   冷启动 -> 标题页 "点击进入"      约 40-55 s
#   点进入 -> 角色出现在世界里        约 40-60 s
log "等待到达标题页（固定 50s）..."
sleep 50
log "点击'点击进入' @ (1584,1330)"
sh_ 'input tap 1584 1330' >/dev/null 2>&1
sleep 6

# 抛锚一次：万一弹了原生覆盖层（取名/公告），把它点掉
XML=$(dump_native 2>/dev/null || true)
if echo "$XML" | grep -q 'EditText'; then
  log "检测到原生输入框（可能是取名），跳过自动填写"
fi

# 有时会弹出 miHoYo 账号登录页（SdkActivity，含"进入游戏"按钮）。
# 实测：该页的按钮对 input tap 无响应（连按下态都没有），但 **BACK 键有效**，
# 按 BACK 会回到游戏主 Activity 并继续走标题页 -> 进世界。
for k in 1 2 3; do
  sleep 15
  cur=$(sh_ 'dumpsys activity activities 2>/dev/null | grep -m1 topResumedActivity' | tr -d '\r')
  case "$cur" in
    *SdkActivity*)
      log "检测到 miHoYo 登录页(SdkActivity)，按 BACK 返回游戏"
      sh_ 'input keyevent KEYCODE_BACK' >/dev/null 2>&1
      sleep 10
      # 回标题页后再点一次"点击进入"
      sh_ 'input tap 1584 1330' >/dev/null 2>&1
      ;;
    *) break ;;
  esac
done

log "等待进入游戏世界（固定 55s）..."
sleep 55
log "已发出进世界指令；用截图确认角色是否已落地"
log "完成（是否真正进入世界请用截图确认；Unity 无 UI 可判定）"
