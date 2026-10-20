#!/bin/bash
# Download the pack at the pinned revision with hf_xet, then verify every file
# against scripts/SHA256SUMS (the Hub's own LFS sha256 for the 17 shards and the
# index, plus sha256 of the three small files).
#
#   HF_TOKEN=hf_... bash scripts/download_and_verify.sh
#
# The repo is public but gated: the token must belong to an account that has
# accepted the model's terms. The token is never printed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$HERE/env.sh"

: "${HF_TOKEN:?Set HF_TOKEN to a token with access to $MODEL_REPO}"

# Runs on the host, not in the image (the image is built after the download).
if ! python3 -c "import huggingface_hub, hf_xet" 2>/dev/null; then
  echo "host python3 lacks huggingface_hub and/or hf_xet. Install them first:" >&2
  echo "  python3 -m pip install --user -U 'huggingface_hub[hf_xet]'" >&2
  exit 1
fi

export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=0   # xet path, not the legacy transfer

mkdir -p "$MODEL_DIR"

python3 - "$MODEL_REPO" "$MODEL_REV" "$MODEL_DIR" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

from huggingface_hub import HfApi
from huggingface_hub.utils import RevisionNotFoundError

repo, rev, dest = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi(token=os.environ["HF_TOKEN"])
try:
    api.model_info(repo, revision=rev)
except RevisionNotFoundError:
    main = api.model_info(repo).sha
    print(f"WARNING: pinned revision {rev[:12]} no longer exists on the Hub "
          f"(history replaced); downloading main = {main[:12]} instead. "
          f"scripts/SHA256SUMS decides whether it is the measured pack.")
    rev = main
print(f"downloading {repo}@{rev[:12]} -> {dest}")
snapshot_download(
    repo_id=repo,
    revision=rev,
    local_dir=dest,
    token=os.environ["HF_TOKEN"],
    max_workers=2,          # measured: >2 workers grows swap on a 128 GB UMA part
    etag_timeout=60,
)
PY

echo "verifying against scripts/SHA256SUMS"
cd "$MODEL_DIR"
sha256sum -c "$HERE/SHA256SUMS"
echo "VERIFY PASS: every file matches the pinned manifest"
echo "total bytes on disk: $(du -sb "$MODEL_DIR" | cut -f1)"