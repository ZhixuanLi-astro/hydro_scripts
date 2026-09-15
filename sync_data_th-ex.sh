#!/bin/bash
# sync_data_th-ex.sh —— 从 TH-eX 拉算例数据回本机（改写自你的 sync_data.sh）
#
# 与你原脚本的区别：
#   远端 Callisto:/mnt/sdc/Zhixuan/athena_works  →  TH-eX 登录节点:/fs2/home/saturn_lizx/athena_works
#   include 规则与你原来完全一致（out1/out2 的 athdf + xdmf、rst、athinput）
#
# 用法（在 hydro_scripts 目录下，和原来一致）：
#   ./sync_data_th-ex.sh 744 DAS            # 拉某个快照号
#   ./sync_data_th-ex.sh latest DAS         # 先在集群上找最新快照号再拉
#   DRY=1 ./sync_data_th-ex.sh latest DAS   # 只打印将要执行的命令
#
# 更省事的替代：`ssh-copy-id th0` 一次之后，ssh/rsync 不再要密码。
set -uo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <num|latest> <case>"
  echo "  e.g.  $0 744 DAS        |   $0 latest DAS"
  exit 1
fi

REMOTE="${REMOTE:-th0}"                       # ~/.ssh/config 里已有的别名（= sh-ex-ln0）
RUSER="${RUSER:-saturn_lizx}"
RBASE="${RBASE:-/fs2/home/${RUSER}/athena_works}"

SPEC="$1"; DIR="$2"
REMOTE_DIR="${RBASE}/${DIR}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DIR="$(dirname "$SCRIPT_DIR")/${DIR}"    # = athena_works/<case>
[ -d "$LOCAL_DIR" ] || { echo "ERROR: 本地目录不存在 $LOCAL_DIR" >&2; exit 2; }

# 复用一条 ssh 连接（与你原脚本同样的做法：密码只输一次）
SOCKET="/tmp/ssh_mux_${USER}_${REMOTE}"
if [ "${DRY:-0}" != "1" ]; then
  ssh -M -S "$SOCKET" -f -N -o ControlPersist=600 "$REMOTE" 2>/dev/null \
    || ssh -M -S "$SOCKET" -f -N -o ControlPersist=600 "$REMOTE"
  echo "SSH master connection established (password entered once)."
fi

if [ "$SPEC" = "latest" ]; then
  echo "在 ${REMOTE}:${REMOTE_DIR} 找最新快照号…"
  NUM="$(ssh -S "$SOCKET" "$REMOTE" "ls -1t '${REMOTE_DIR}'/iceline.*.rst 2>/dev/null | head -1 | sed -n 's#.*iceline\.\([0-9]\{5\}\)\.rst#\1#p'")"
  [ -n "$NUM" ] || { echo "ERROR: 没找到 *.rst（或连不上 ${REMOTE}）" >&2; [ "${DRY:-0}" = "1" ] || ssh -S "$SOCKET" -O exit "$REMOTE" 2>/dev/null; exit 3; }
  echo "最新快照号: $NUM"
else
  NUM="$(printf '%05d' "$1")"
fi

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

"${RSYNC[@]}"
rc=$?
ssh -S "$SOCKET" -O exit "$REMOTE" 2>/dev/null
echo "Done（快照 ${NUM} → ${LOCAL_DIR}）rc=$rc"
exit $rc
