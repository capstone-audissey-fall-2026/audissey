#!/usr/bin/env bash
# Download every model Audyssey needs into $HF_HOME (a GPU server cache or a RunPod network volume).
# Run once per cache after `huggingface-cli login` with a read token that has the access listed in
# docs/running.md (SAM licenses, sogang-audyssey org). Set WITH_VLLM=1 to add the Compose vLLM planner.
set -euo pipefail
: "${HF_HOME:?set HF_HOME to the weight cache directory}"

dl() { huggingface-cli download "$@" > /dev/null && echo "ok  $1"; }

# Separation (SAM Audio) and its text encoder
dl facebook/sam-audio-large-tv
dl t5-base
# Filtering (SAM Audio Judge): weights on the sam_audio revision, config/tokenizer on main
dl facebook/sam-audio-judge --revision sam_audio
dl facebook/sam-audio-judge --include "*.json"
# Grounding (SAM3.1, used only with AUDYSSEY_VISUAL_PROMPT=true)
dl facebook/sam3.1 config.json sam3.1_multiplex.pt
# CLAP: generation scoring and separation reranking
dl lukewys/laion_clap 630k-best.pt
dl roberta-base
dl bert-base-uncased
dl facebook/bart-base
# Generation (internal model): public base weights, private weights (sogang-audyssey members only),
# encoders, vocoder
dl jnwnlee/SelVA weights/video_enc_sup_5.pth ext_weights/v1-44.pth ext_weights/synchformer_state_dict.pth
dl sogang-audyssey/internal-model generator/S_O_s43/model_ema_step1000.pth encoder/S_O/best.pt
dl apple/DFN5B-CLIP-ViT-H-14-384
dl google/flan-t5-base
dl nvidia/bigvgan_v2_44khz_128band_512x

if [ "${WITH_VLLM:-0}" = 1 ]; then
  dl Qwen/Qwen2.5-VL-3B-Instruct
fi
echo "done: $HF_HOME"
