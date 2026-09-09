#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

if [[ ! -x .venv/bin/v2a || ! -x server/.venv/bin/v2a-inspect-server ]]; then
    echo "Local environments are missing. Run: uv sync --extra ui && uv sync --project server" >&2
    exit 1
fi
if [[ ! -f .env ]]; then
    echo ".env is missing. Copy .env.example to .env and configure it." >&2
    exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

pids=()
cleanup() {
    trap - EXIT INT TERM
    if ((${#pids[@]})); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

server/.venv/bin/v2a-inspect-server serve &
pids+=("$!")
.venv/bin/v2a ui &
pids+=("$!")

echo "Inference: http://127.0.0.1:${V2A_SERVER_PORT:-8080}"
echo "UI:        http://127.0.0.1:${V2A_INSPECT_UI_PORT:-8501}"
wait -n "${pids[@]}"
