#!/bin/bash
# sync_all.sh — poll-and-sync: every 30 minutes, pull new files from Callisto.
# Usage: ./sync_all.sh <dir1> [dir2] [dir3] ...
#   e.g.  ./sync_all.sh passive_test single_pop

PASSFILE="$HOME/.ssh/callisto_pass"
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dir1> [dir2] [dir3] ..."
    echo "  e.g.  $0 passive_test single_pop"
    exit 1
fi

REMOTE="Callisto"
REMOTE_BASE="/mnt/sdc/Zhixuan/athena_works"
# Resolve LOCAL_BASE to the parent of this script's directory (absolute path)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_BASE="$(dirname "$SCRIPT_DIR")"
SOCKET="/tmp/ssh_mux_${USER}_callisto_syncall"
INTERVAL=1800  # 30 minutes

echo "=== sync_all.sh started at $(date) ==="
echo "Watching directories: $@"
echo "Polling every ${INTERVAL}s (30 min). Press Ctrl+C to stop."
echo ""

# ── Establish persistent SSH master connection ───────────────────────────────
setup_ssh_master() {
    if ! sshpass -f "$PASSFILE" ssh -S "$SOCKET" -O check "$REMOTE" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] Opening SSH master to $REMOTE ..."
        sshpass -f "$PASSFILE" ssh -M -S "$SOCKET" -f -N -o ControlPersist=yes -o ServerAliveInterval=60 "$REMOTE" 2>/dev/null
        if [ $? -ne 0 ]; then
            # retry once on stale socket
            rm -f "$SOCKET"
            sshpass -f "$PASSFILE" ssh -M -S "$SOCKET" -f -N -o ControlPersist=yes -o ServerAliveInterval=60 "$REMOTE"
        fi
        echo "[$(date '+%H:%M:%S')] SSH master ready."
    fi
}

# ── Sync one directory ──────────────────────────────────────────────────────
sync_dir() {
    local DIR="$1"
    local REMOTE_DIR="${REMOTE_BASE}/${DIR}"
    local LOCAL_DIR="${LOCAL_BASE}/${DIR}"

    echo "[$(date '+%H:%M:%S')] Syncing ${DIR} -> ${LOCAL_DIR} ..."

    mkdir -p "$LOCAL_DIR"

    rsync -r -u --progress \
        -e "sshpass -f $PASSFILE ssh -S $SOCKET" \
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

    echo "[$(date '+%H:%M:%S')] ${DIR} done."
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

    setup_ssh_master

    for DIR in "$@"; do
        sync_dir "$DIR"
    done

    echo "[$(date '+%H:%M:%S')] Sleeping ${INTERVAL}s ..."
    sleep "$INTERVAL"
done
