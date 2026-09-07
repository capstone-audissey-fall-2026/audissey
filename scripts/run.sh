#!/usr/bin/env bash
set -euo pipefail

input_dir=$(realpath "${1:-.}")
server_url=${2:-http://127.0.0.1:8080}
output_dir=$(realpath -m "${3:-$input_dir/out}")
work_dir=$(realpath -m "${4:-$input_dir/work}")

mkdir -p "$output_dir" "$work_dir"
shopt -s nullglob
videos=("$input_dir"/*.mp4)
if ((${#videos[@]} == 0)); then
    echo "No MP4 files found in $input_dir" >&2
    exit 1
fi

for file in "${videos[@]}"; do
    filename=$(basename "${file%.*}")
    item_work_dir="$work_dir/$filename"
    mkdir -p "$item_work_dir"
    uv run v2a run \
        --output "$output_dir/$filename.json" \
        --work-dir "$item_work_dir" \
        --server-url "$server_url" \
        "$file"
done
