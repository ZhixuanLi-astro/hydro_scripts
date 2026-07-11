#!/bin/bash
# Sync data from Callisto using SSH multiplexing — password only needed once.
# Usage: ./sync_data.sh <num> <dir>
#   e.g.  ./sync_data.sh 744 passive_test
#   -> syncs *00744* files from /mnt/sdc/Zhixuan/athena_works/passive_test/

if [ $# -ne 2 ]; then
	echo "Usage: $0 <num> <dir>"
	echo "  e.g.  $0 744 passive_test"
	exit 1
fi

NUM=$(printf "%05d" "$1")
DIR="$2"

REMOTE="Callisto"
REMOTE_DIR="/mnt/sdc/Zhixuan/athena_works/${DIR}"
LOCAL_DIR="../${DIR}"

echo "Syncing *${NUM}* from ${REMOTE}:${REMOTE_DIR}/"

# --- SSH multiplexing setup: reuse one connection for all rsyncs ---
SOCKET="/tmp/ssh_mux_${USER}_callisto"
ssh -M -S "$SOCKET" -f -N -o ControlPersist=600 "$REMOTE" 2>/dev/null

# If the master connection failed (e.g. socket stale), force a fresh one
if [ $? -ne 0 ]; then
	ssh -M -S "$SOCKET" -f -N -o ControlPersist=600 "$REMOTE"
fi

echo "SSH master connection established (password entered once)."

# --- Rsync the data files ---
rsync -r -u --progress \
	-e "ssh -S $SOCKET" \
	"${REMOTE}:${REMOTE_DIR}/" \
	"$LOCAL_DIR/" \
	--include='*/' \
	--include="*out1.*${NUM}*.athdf" \
	--include="*out1.*${NUM}*.athdf.xdmf" \
	--include="*out2.*${NUM}*.athdf" \
	--include="*out2.*${NUM}*.athdf.xdmf" \
	--include="*iceline.*${NUM}*.rst" \
	--include="athinput.iceline" \
	--exclude='*'

# --- Clean up the SSH master connection ---
ssh -S "$SOCKET" -O exit "$REMOTE" 2>/dev/null
echo "Done. SSH connection closed."
