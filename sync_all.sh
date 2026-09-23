#!/bin/bash
# sync_all.sh — poll-and-sync: every INTERVAL seconds, pull new files from the remote site.
#
# Two sites are supported (default keeps the original Callisto behaviour):
#   SITE=callisto (default) : Callisto:/mnt/sdc/Zhixuan/athena_works   (uses sshpass + ~/.ssh/callisto_pass)
#   SITE=cluster            : th0:/fs2/home/saturn_lizx/athena_works   (uses SSH keys — run `ssh-copy-id th0` once)
#
# Usage: ./sync_all.sh <dir1> [dir2] [dir3] ...
#   e.g.  ./sync_all.sh passive_test single_pop                    # from Callisto (unchanged)
#         SITE=cluster ./sync_all.sh EPR_SF DAS                    # from the TH-eX cluster
#         SITE=cluster ONCE=1 ./sync_all.sh EPR_SF                 # run a single cycle then exit
#         SITE=cluster INTERVAL=600 ./sync_all.sh EPR_SF           # poll every 10 min
#         DRY=1 SITE=cluster ONCE=1 ./sync_all.sh EPR_SF           # print commands, do nothing
#
# Notes about the cluster:
#   * If rsync fails with "protocol version mismatch -- is your shell clean?", the remote login shell is
#     printing something on stdout. This script then falls back to scp (SFTP subsystem, immune to that),
#     and only transfers files newer than your newest local copy (incremental) instead of everything.
#     Permanent fix (on the cluster): put `case $- in *i*) ;; *) return;; esac` at the top of ~/.bashrc.
#   * Override with REMOTE= / REMOTE_BASE= / SSH_PASS_FILE= if your paths differ.

if [ $# -eq 0 ]; then
    echo "Usage: $0 <dir1> [dir2] [dir3] ..."
    echo "  e.g.  $0 passive_test single_pop              # Callisto (default)"
    echo "        SITE=cluster $0 EPR_SF DAS              # TH-eX cluster"
    exit 1
fi

SITE="${SITE:-callisto}"
case "$SITE" in
  callisto)
    REMOTE="${REMOTE:-Callisto}"
    REMOTE_BASE="${REMOTE_BASE:-/mnt/sdc/Zhixuan/athena_works}"
    SSH_PASS_FILE="${SSH_PASS_FILE:-$HOME/.ssh/callisto_pass}"
    ;;
  cluster|th-ex|thex|tj)
    REMOTE="${REMOTE:-th0}"                                   # ~/.ssh/config alias -> 192.168.10.50
    REMOTE_BASE="${REMOTE_BASE:-/fs2/home/saturn_lizx/athena_works}"
    SSH_PASS_FILE="${SSH_PASS_FILE:-}"                         # empty = use SSH keys (recommended)
    ;;
  *)
    echo "Unknown SITE='$SITE' (use callisto or cluster)" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_BASE="$(dirname "$SCRIPT_DIR")"                          # = athena_works/
SOCKET="/tmp/ssh_mux_${USER}_${SITE}_syncall"
INTERVAL="${INTERVAL:-1800}"   # 30 minutes

# ssh wrapper: use sshpass only when a password file is configured and readable
SSH() {
    if [ -n "$SSH_PASS_FILE" ] && [ -r "$SSH_PASS_FILE" ] && command -v sshpass >/dev/null 2>&1; then
        sshpass -f "$SSH_PASS_FILE" ssh "$@"
    else
        ssh "$@"
    fi
}
# scp 复用同一条 ssh 主连接：注意 `scp -S` 是"指定 ssh 程序"，不是控制套接字 —— 要用 -o ControlPath
SCP() {
    if [ -n "$SSH_PASS_FILE" ] && [ -r "$SSH_PASS_FILE" ] && command -v sshpass >/dev/null 2>&1; then
        sshpass -f "$SSH_PASS_FILE" scp -o ControlPath="$SOCKET" -o ControlMaster=auto "$@"
    else
        scp -o ControlPath="$SOCKET" -o ControlMaster=auto "$@"
    fi
}
if [ -n "$SSH_PASS_FILE" ]; then
    RSH="sshpass -f $SSH_PASS_FILE ssh -S $SOCKET"
else
    RSH="ssh -S $SOCKET"
fi

echo "=== sync_all.sh started at $(date) ==="
echo "Site      : $SITE  ($REMOTE:$REMOTE_BASE)"
echo "Local base: $LOCAL_BASE"
echo "Watching  : $*"
if [ "$INTERVAL" -ge 60 ] 2>/dev/null; then
    echo "Polling every ${INTERVAL}s ($(( INTERVAL / 60 )) min). Press Ctrl+C to stop."
else
    echo "Polling every ${INTERVAL}s. Press Ctrl+C to stop."
fi
[ "${DRY:-0}" = "1" ] && echo "DRY=1: only printing commands."
echo ""

# ── Establish persistent SSH master connection ───────────────────────────────
setup_ssh_master() {
    if ! ssh -S "$SOCKET" -O check "$REMOTE" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] Opening SSH master to $REMOTE ..."
        SSH -M -S "$SOCKET" -f -N -o ControlPersist=yes -o ServerAliveInterval=60 "$REMOTE" 2>/dev/null
        if [ $? -ne 0 ]; then
            # retry once on stale socket
            rm -f "$SOCKET"
            SSH -M -S "$SOCKET" -f -N -o ControlPersist=yes -o ServerAliveInterval=60 "$REMOTE"
        fi
        echo "[$(date '+%H:%M:%S')] SSH master ready."
    fi
}

# ── Incremental scp fallback (used when rsync's protocol handshake is polluted) ──
#   1) find the newest mtime we already have locally
#   2) ask the remote for files newer than that (one ssh call)
#   3) scp them in batches (SFTP subsystem -> not affected by a noisy login shell)
sync_dir_by_scp() {
    local DIR="$1" REMOTE_DIR="${REMOTE_BASE}/${1}" LOCAL_DIR="${LOCAL_BASE}/${1}"
    local newest remote_list n=0 batch=() f
    newest="$(find "$LOCAL_DIR" -maxdepth 1 -type f \( -name 'iceline.*' -o -name 'athinput.iceline' \) \
              -printf '%T@\n' 2>/dev/null | sort -n | tail -1)"
    if [ -n "$newest" ]; then
        newest="${newest%.*}"
        remote_list="$(SSH -S "$SOCKET" "$REMOTE" "find '${REMOTE_DIR}' -maxdepth 1 -type f \
            -newermt @${newest} \( -name 'iceline.*' -o -name 'athinput.iceline' \) -printf '%f\n' 2>/dev/null" \
            | grep -E '^(iceline|athinput)')"
    else
        remote_list="$(SSH -S "$SOCKET" "$REMOTE" "cd '${REMOTE_DIR}' 2>/dev/null && ls -1 \
            *iceline.* athinput.iceline 2>/dev/null" | grep -E '^(iceline|athinput)')"
    fi
    if [ -z "$remote_list" ]; then
        echo "[$(date '+%H:%M:%S')]   scp fallback: nothing new."
        return 0
    fi
    echo "[$(date '+%H:%M:%S')]   scp fallback: $(printf '%s\n' "$remote_list" | grep -c .) file(s) to fetch."
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        batch+=("${REMOTE}:${REMOTE_DIR}/${f}")
        if [ ${#batch[@]} -ge 40 ]; then
            if [ "${DRY:-0}" = "1" ]; then
                echo "    [DRY] scp -p <${#batch[@]} files> $LOCAL_DIR/"
            else
                SCP -p "${batch[@]}" "$LOCAL_DIR/" && n=$((n + ${#batch[@]}))
            fi
            batch=()
        fi
    done <<< "$remote_list"
    if [ ${#batch[@]} -gt 0 ]; then
        if [ "${DRY:-0}" = "1" ]; then
            echo "    [DRY] scp -p <${#batch[@]} files> $LOCAL_DIR/"
        else
            SCP -p "${batch[@]}" "$LOCAL_DIR/" && n=$((n + ${#batch[@]}))
        fi
    fi
    echo "[$(date '+%H:%M:%S')]   scp fallback done: ${n:-0} file(s)."
}

# ── Sync one directory ──────────────────────────────────────────────────────
sync_dir() {
    local DIR="$1"
    local REMOTE_DIR="${REMOTE_BASE}/${DIR}"
    local LOCAL_DIR="${LOCAL_BASE}/${DIR}"

    echo "[$(date '+%H:%M:%S')] Syncing ${DIR} -> ${LOCAL_DIR} ..."

    if [ "${DRY:-0}" = "1" ]; then
        echo "    [DRY] mkdir -p ${LOCAL_DIR}"
        echo "    [DRY] rsync -r -u --progress -e \"$RSH\" ${REMOTE}:${REMOTE_DIR}/ ${LOCAL_DIR}/ \\"
        echo "          --include='*/' --include='*out1.*.athdf' --include='*out1.*.athdf.xdmf' \\"
        echo "          --include='*out2.*.athdf' --include='*out2.*.athdf.xdmf' \\"
        echo "          --include='*iceline.*.rst' --include='athinput.iceline' --exclude='*'"
        echo "[$(date '+%H:%M:%S')] ${DIR} done (dry run)."
        return 0
    fi

    if rsync -r -u --progress \
        -e "$RSH" \
        "${REMOTE}:${REMOTE_DIR}/" \
        "$LOCAL_DIR/" \
        --include='*/' \
        --include='*out1.*.athdf' \
        --include='*out1.*.athdf.xdmf' \
        --include='*out2.*.athdf' \
        --include='*out2.*.athdf.xdmf' \
        --include='*iceline.*.rst' \
        --include='athinput.iceline' \
        --exclude='*'
    then
        echo "[$(date '+%H:%M:%S')] ${DIR} done."
    else
        echo "[$(date '+%H:%M:%S')] rsync failed — falling back to scp (SFTP, immune to shell noise)."
        sync_dir_by_scp "$DIR"
        echo "[$(date '+%H:%M:%S')] ${DIR} done (scp fallback)."
    fi
}

# ── Cleanup on exit ─────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[$(date '+%H:%M:%S')] Shutting down ..."
    ssh -S "$SOCKET" -O exit "$REMOTE" 2>/dev/null
    echo "SSH connection closed. Goodbye."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Main loop ───────────────────────────────────────────────────────────────
ITER=0
while true; do
    ITER=$((ITER + 1))
    echo ""
    echo "─── Cycle #${ITER} at $(date) ───"

    if [ "${DRY:-0}" != "1" ]; then
        setup_ssh_master
    fi

    for DIR in "$@"; do
        sync_dir "$DIR"
    done

    if [ "${ONCE:-0}" = "1" ]; then
        echo "[$(date '+%H:%M:%S')] ONCE=1 — exiting after one cycle."
        cleanup
    fi

    echo "[$(date '+%H:%M:%S')] Sleeping ${INTERVAL}s ..."
    sleep "$INTERVAL"
done
