#!/bin/bash
# sync_data_th-ex.sh —— 从 TH-eX 拉算例数据回本机（改写自你的 sync_data.sh）
#
# 与你原脚本的区别：
#   远端 Callisto:/mnt/sdc/Zhixuan/athena_works  →  TH-eX 登录节点:/fs2/home/saturn_lizx/athena_works
#   include 规则与原来一致（out1/out2 的 athdf + xdmf、rst、athinput）
#
# 用法（在 hydro_scripts 目录下）：
#   ./sync_data_th-ex.sh 583 EPR_SF           # 拉某个快照号
#   ./sync_data_th-ex.sh latest EPR_SF        # 先找最新快照号再拉
#   MODE=scp ./sync_data_th-ex.sh 583 EPR_SF  # 直接用 scp（见下）
#   DRY=1 ./sync_data_th-ex.sh latest EPR_SF  # 只打印将执行的命令
#
# ⚠️ 关于 "protocol version mismatch -- is your shell clean?"：
#   这是 **远端登录 shell 在 rsync 握手前往 stdout 打印了东西**（banner/模块加载提示等）导致协议流被污染。
#   本机与远端 rsync 都是 3.2.7，所以不是版本问题。两个办法：
#     ① 根治（在集群侧）：
#          ssh th0 /bin/true | od -c | head                       # 有输出 = 确认有噪声
#          ssh th0 'bash --noprofile --norc -c /bin/true' | od -c | head   # 若这条干净 → 噪声来自 rc 文件
#          ssh th0 'grep -nE "echo|printf|banner|motd" ~/.bashrc ~/.bash_profile ~/.profile 2>/dev/null'
#        然后在 ~/.bashrc 最上面加一行（让非交互 shell 直接返回），或把打印的那行重定向到 /dev/null：
#          case $- in *i*) ;; *) return;; esac
#     ② 绕过（本脚本已内置）：rsync 失败时自动改用 scp —— scp 走 SFTP 子系统、不经过登录 shell，免疫该噪声。
set -uo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <num|latest> <case>"
  echo "  e.g.  $0 583 EPR_SF        |   $0 latest EPR_SF"
  exit 1
fi

REMOTE="${REMOTE:-th0}"                       # ~/.ssh/config 里已有的别名
RUSER="${RUSER:-saturn_lizx}"
RBASE="${RBASE:-/fs2/home/${RUSER}/athena_works}"

SPEC="$1"; DIR="$2"
REMOTE_DIR="${RBASE}/${DIR}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DIR="$(dirname "$SCRIPT_DIR")/${DIR}"    # = athena_works/<case>
[ -d "$LOCAL_DIR" ] || { echo "ERROR: 本地目录不存在 $LOCAL_DIR" >&2; exit 2; }

# 复用一条 ssh 连接（与原脚本一致：密码只输一次）
SOCKET="/tmp/ssh_mux_${USER}_${REMOTE}"
if [ "${DRY:-0}" != "1" ]; then
  ssh -M -S "$SOCKET" -f -N -o ControlPersist=600 "$REMOTE" 2>/dev/null \
    || ssh -M -S "$SOCKET" -f -N -o ControlPersist=600 "$REMOTE"
  echo "SSH master connection established (password entered once)."
fi

if [ "$SPEC" = "latest" ]; then
  echo "在 ${REMOTE}:${REMOTE_DIR} 找最新快照号…"
  # 客户端再过滤一道：远端 shell 可能往 stdout 打 banner，这里只接受"纯 5 位数字"的行
  NUM="$(ssh -S "$SOCKET" "$REMOTE" "ls -1t '${REMOTE_DIR}'/iceline.*.rst 2>/dev/null | head -1 | sed -n 's#.*iceline\.\([0-9]\{5\}\)\.rst#\1#p'" \
        | grep -oE '^[0-9]{5}$' | head -1)"
  [ -n "$NUM" ] || { echo "ERROR: 没找到 *.rst（或连不上 ${REMOTE}）" >&2; [ "${DRY:-0}" = "1" ] || ssh -S "$SOCKET" -O exit "$REMOTE" 2>/dev/null; exit 3; }
  echo "最新快照号: $NUM"
else
  # 10# 强制十进制：printf %d 会把 00583 当八进制（0583₈ = 387）——静默拉错文件
  NUM="$(printf '%05d' "$((10#$1))")"
fi

# ── scp 兜底：SFTP 子系统不经过登录 shell，绕开 shell 噪声 ──
pull_scp() {
  local list n=0 f
  list="$(ssh -S "$SOCKET" "$REMOTE" "cd '${REMOTE_DIR}' && ls -1 \
      *out1.*${NUM}*.athdf *out1.*${NUM}*.athdf.xdmf \
      *out2.*${NUM}*.athdf *out2.*${NUM}*.athdf.xdmf \
      *iceline.*${NUM}*.rst athinput.iceline 2>/dev/null" | grep -E '^(iceline|athinput)')"
  if [ -z "$list" ]; then echo "scp 兜底：远端没有匹配快照 ${NUM} 的文件"; return 1; fi
  while read -r f; do
    [ -n "$f" ] || continue
    if scp -p "${REMOTE}:${REMOTE_DIR}/${f}" "$LOCAL_DIR/"; then n=$((n+1)); fi
  done <<< "$list"
  echo "scp 兜底完成：$n 个文件 → $LOCAL_DIR"
  [ "$n" -gt 0 ]
}

RSYNC=(rsync -r -u --progress -e "ssh -S $SOCKET"
  "${REMOTE}:${REMOTE_DIR}/" "$LOCAL_DIR/"
  --include='*/'
  --include="*out1.*${NUM}*.athdf"
  --include="*out1.*${NUM}*.athdf.xdmf"
  --include="*out2.*${NUM}*.athdf"
  --include="*out2.*${NUM}*.athdf.xdmf"
  --include="*iceline.*${NUM}*.rst"
  --include="athinput.iceline"
  --exclude='*')

if [ "${DRY:-0}" = "1" ]; then
  echo "（DRY=1）将执行:"; printf '  %s\n' "${RSYNC[*]}"; exit 0
fi

if [ "${MODE:-rsync}" = "scp" ]; then
  echo "MODE=scp：跳过 rsync，直接用 scp"
  pull_scp; rc=$?
else
  "${RSYNC[@]}"; rc=$?
  if [ $rc -ne 0 ]; then
    cat <<'EOF'
rsync 失败 —— 若报 "protocol version mismatch -- is your shell clean?"，说明远端登录 shell 往 stdout
打印了东西污染了协议流。改用 scp 兜底（走 SFTP 子系统，不经过登录 shell）；根治方法见本脚本头部注释。
EOF
    pull_scp; rc=$?
  fi
fi

ssh -S "$SOCKET" -O exit "$REMOTE" 2>/dev/null
echo "Done（快照 ${NUM} → ${LOCAL_DIR}）rc=$rc"
exit $rc
