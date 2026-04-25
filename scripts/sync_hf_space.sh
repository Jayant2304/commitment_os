#!/usr/bin/env bash
# Sync this repo into the Hugging Face Space (same pattern as incident_response_env/deploy.sh).
# HF rejects binary PNGs on git push — we rsync the tree omitting *.png; training curves stay on GitHub.
set -euo pipefail

HF_USERNAME="${HF_USERNAME:-Jayant2304}"
SPACE_NAME="${SPACE_NAME:-commitment-os}"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN for git push to Hugging Face (see https://huggingface.co/settings/tokens)}"
COMMIT_MSG="${1:-Sync from GitHub (exclude *.png per HF policy)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLONE_DIR="${HF_SYNC_CLONE_DIR:-/tmp/hf-sync-${SPACE_NAME}}"

echo "==> Cloning Space into ${CLONE_DIR}"
rm -rf "${CLONE_DIR}"
git clone "https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}" "${CLONE_DIR}"

echo "==> Rsync from ${REPO_ROOT} (excluding PNGs and local-only paths)"
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '*.egg-info/' \
  --exclude '.env' \
  --exclude 'training_output/' \
  --exclude '*.png' \
  --exclude '.DS_Store' \
  "${REPO_ROOT}/" "${CLONE_DIR}/"

echo "==> Space README: HF_README.md -> README.md"
cp "${REPO_ROOT}/HF_README.md" "${CLONE_DIR}/README.md"

cd "${CLONE_DIR}"
git add -A
if git diff --cached --quiet; then
  echo "==> No changes to commit (tree already matched)"
else
  git commit -m "${COMMIT_MSG}"
fi

echo "==> Pushing to Hugging Face"
git push

echo "==> Done. Build: https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
