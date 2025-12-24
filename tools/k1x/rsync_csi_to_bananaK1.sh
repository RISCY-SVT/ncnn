#!/usr/bin/env bash
set -euo pipefail

# Sync a local directory to a remote K1X/board using rsync over SSH.
# Usage:
#   BANANA_HOST=host BANANA_USER=user [BANANA_PORT=22] [DEST_DIR=/remote/path] \
#     ./rsync_csi_to_bananaK1.sh [SRC_DIR]
# Example:
#   BANANA_HOST=board.local BANANA_USER=user ./rsync_csi_to_bananaK1.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,40p' "$0"
  exit 0
fi

SRC_DIR="${1:-$SCRIPT_DIR}"
BANANA_HOST="${BANANA_HOST:-}"
BANANA_USER="${BANANA_USER:-}"
BANANA_PORT="${BANANA_PORT:-22}"

if [[ -z "$BANANA_HOST" || -z "$BANANA_USER" ]]; then
  echo "ERROR: BANANA_HOST and BANANA_USER must be set." >&2
  echo "Hint: BANANA_HOST=host BANANA_USER=user $0 [SRC_DIR]" >&2
  exit 2
fi

DEST_DIR="${DEST_DIR:-/home/${BANANA_USER}/${SRC_DIR##*/}}"

rsync -avz --no-perms --delete -e "ssh -p ${BANANA_PORT}" "${SRC_DIR%/}/" \
  "${BANANA_USER}@${BANANA_HOST}:${DEST_DIR}"

echo
echo "Files successfully synced. Current time: $(date +'%F %T %Z')"
date +%Y-%m-%d_%H-%M-%S
