# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

AI Toolkit is a diffusion model training suite supporting image, video, and audio models (FLUX, Chroma, HiDream, Wan, ACE Step, and more). It runs as a CLI or full-featured web UI and targets consumer-grade GPUs.

## Common Commands

### Python Backend

```bash
# Run a training job
python run.py config/examples/train_lora_flux_24gb.yaml

# Run with name substitution (replaces [name] in config)
python run.py config/examples/train_lora_flux_24gb.yaml --name my_subject

# Run multiple jobs sequentially, continuing past failures
python run.py config1.yaml config2.yaml --recover

# Log output to file
python run.py config/examples/train_lora_flux_24gb.yaml --log output.log

# Run tests (no pytest runner; run test scripts directly)
python testing/test_bucket_dataloader.py
python testing/test_vae.py

# Gradio simplified UI
python flux_train_ui.py

# Cloud training on Modal
python run_modal.py config/my_config.yaml
```

### Web UI (from `ui/` directory)

```bash
cd ui
npm install
npm run update_db        # Run Prisma migrations + generate client
npm run dev              # Development server (port 3000 + worker)
npm run build_and_start  # Production: install, migrate, build, start (port 8675)
npm run lint             # ESLint via Next.js
npm run format           # Prettier
```

### Environment Variables

- `SEED` — global random seed (integer)
- `DEBUG_TOOLKIT=1` — enables PyTorch anomaly detection
- `AI_TOOLKIT_AUTH` — auth token for the web UI
- `HF_HUB_ENABLE_HF_TRANSFER` — defaults to `1` (fast HF downloads)

### Docker

```bash
docker compose up        # GPU-enabled container (requires nvidia runtime)
```

## Architecture

### Job Execution Pipeline

`run.py` → `toolkit/job.py:get_job()` → one of several `jobs/` classes → one or more Process objects.

The top-level `job` key in a YAML config determines which job class runs:
- `train` → `TrainJob` — the primary use case; delegates to an extension's trainer
- `generate` → `GenerateJob`
- `extract` → `ExtractJob`
- `extension` → `ExtensionJob` — runs arbitrary extension logic
- `mod` → `ModJob` — model weight manipulation

### Extension System

All model support lives in `extensions_built_in/`. Each subdirectory exposes an `AI_TOOLKIT_EXTENSIONS` list of extension classes. Extensions are discovered dynamically by `toolkit/extension.py:get_all_extensions()`.

Key extension groups:
- `diffusion_models/` — one subdirectory per model family (flux2, chroma, hidream, wan, ltx2, anima, etc.). Each typically contains a `*_model.py` trainer, a `pipeline.py`, and a `src/` with custom architecture code.
- `audio_models/` — ACE Step audio model
- `captioner/` — dataset captioning tools (Qwen3VL, ACE Step)
- `dataset_tools/` — data preparation utilities
- `concept_slider/`, `concept_replacer/`, `advanced_generator/` — specialized training/generation modes

User extensions go in `extensions/` (not tracked by git).

### Config System

Configs are YAML/JSON. `toolkit/config.py:get_config()` handles loading, env variable substitution, and `[name]` placeholder replacement. Example configs live in `config/examples/`.

Key config sections: `job`, `config.name`, `config.process[]` (list of process configs, each with a `type` key that maps to an extension's process class).

### Toolkit Core (`toolkit/`)

Heavy lifting lives here:
- `stable_diffusion_model.py` — base diffusion model wrapper
- `lora_special.py`, `lycoris_utils.py` — LoRA/LyCORIS application
- `data_loader.py`, `buckets.py` — aspect-ratio-bucketed dataset loading
- `optimizer.py`, `schedulers.py` — optimizer and LR scheduler wrappers
- `saving.py` — checkpoint save/load and format conversion (SafeTensors ↔ LDM)
- `accelerator.py` — thin wrapper around Hugging Face Accelerate
- `config_modules.py` — Pydantic config dataclasses shared across the codebase

### Web UI (`ui/`)

Next.js 15 + React 19 + Tailwind. Prisma + SQLite for job state. A background `cron/worker.ts` process manages job execution. Monaco editor for YAML config editing in-browser. Real-time training metrics displayed via Recharts. Runs on port 8675 in production.

## Anima Model

Anima is a flow-matching DiT (MiniTrainDIT architecture, Cosmos Predict2-style) with a Qwen3-0.6B text encoder bridged to the DiT via an LLM adapter. Config key: `arch: "anima"`.

### Required model files
- **Transformer** (`name_or_path`): `anima-preview3-base.safetensors` or a HF repo ID. Keys are stored with a `net.` prefix that is stripped on load.
- **Text encoder** (`extras_name_or_path`): `"Qwen/Qwen3-0.6B"` (HF hub, recommended) or path to `qwen_3_06b_base.safetensors`.
- **VAE** (`vae_path`): `qwen_image_vae.safetensors`. Omit to auto-download from HF. Keys require remapping via `_remap_vae_keys()`.

### Architecture notes
- Uses a T5 tokenizer (`google/t5-v1_1-base`, 32128-vocab) to produce token IDs fed to the LLM adapter, **not** for embeddings.
- The LLM adapter cross-attends T5 token IDs to Qwen3 hidden states → 1024-dim context vectors.
- **Context must be padded to 512 tokens** (zeros) before being passed to the DiT. The model was trained with 512-length context; feeding shorter sequences corrupts cross-attention weights. This is done in `get_prompt_embeds` after the `llm_adapter` call.
- `concat_padding_mask=True` — the `PatchEmbed` expects 17 input channels (16 latent + 1 padding mask). When `padding_mask=None` a zero mask is created automatically inside `MiniTrainDIT.forward`.
- Timesteps are passed in [0, 1] range (scheduler values [0, 1000] divided by 1000).
- Loss target is the flow-matching velocity: `noise - x_0` (returned by `get_loss_target`).
- VAE: `AutoencoderKLQwenImage`, 16× spatial downsampling, 16 latent channels. Latents normalised as `(raw - mean) / std` on encode; reversed on decode.

### Example config
`config/examples/train_lora_anima.yaml`

## Adding a New Model

1. Create `extensions_built_in/diffusion_models/<model_name>/`.
2. Implement a trainer class extending the appropriate base (see nearby models for patterns).
3. Expose it in `__init__.py` via `AI_TOOLKIT_EXTENSIONS`.
4. Add an example config to `config/examples/`.

The trainer class must implement at minimum: `get_process()` returning a process instance that has `run()` and `cleanup()`.
