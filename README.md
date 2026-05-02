# Skripsi: Thesis Encoder Implementation (Research-Focused)

This repository contains the implementation used in my thesis work on encoder modeling and training.

## Important Disclaimer

- This project is **for research and thesis demonstration purposes**.
- This repository **does not claim to beat SOTA encoder models**.
- The main contribution is the **rigorous explanation and calculation examples behind the encoder design** in the thesis manuscript.
- The code here serves as a **program implementation reference** and a record of **experimental results**.

## Project Scope

This codebase includes:

- Pretraining pipeline (ELECTRA-style with generator + discriminator)
- Finetuning pipeline for GLUE tasks
- Baselines (absolute positional encoding, MLM-only, RTD-only)
- Data download + preprocessing scripts
- Smoke tests for pretrain and finetune flows

## Repository Highlights

- `main_pretrain.py`: main pretraining entrypoint
- `main_finetune.py`: main finetuning entrypoint
- `scripts/`: dataset download, tokenizer training, preprocessing
- `configs/`: model/training configs
- `baselines/`: baseline variants and configs
- `test/`: smoke tests and sample outputs

## Requirements

- Python 3.10+ recommended
- PyTorch-compatible environment (CPU or GPU)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download Pretraining Data

```bash
python scripts/download_pretrain.py
```

Optional debug mode (small sample):

```bash
python scripts/download_pretrain.py --debug
```

### 2. Train Tokenizer

```bash
python scripts/train_tokenizer.py
```

### 3. Preprocess Pretraining Data

```bash
python scripts/preprocess.py --task pretrain --max_length 256
```

### 4. Run Pretraining

```bash
python main_pretrain.py
```

Outputs will be saved under `outputs/pretrain`.

## Finetuning on GLUE

### 1. Download GLUE

```bash
python scripts/download_glue.py
```

Optional debug mode:

```bash
python scripts/download_glue.py --debug
```

### 2. Preprocess GLUE

```bash
python scripts/preprocess.py --task finetune --max_length 256
```

### 3. Configure Task

Edit `configs/finetune.yaml`:

- Set `task.name` to a single task (for example `SST2`),
- Or set `task.name: ALL` to run all configured tasks.

Also ensure `checkpoint.pretrained_path` points to your desired pretrained checkpoint.

### 4. Run Finetuning

```bash
python main_finetune.py
```

Outputs will be saved under `outputs/finetune/<TASK_NAME>`.

## Baseline Experiments

Pretraining baselines:

```bash
python baselines/main_absolute_pos.py
python baselines/main_rtd_only.py
python baselines/main_mlm_only.py
```

Finetuning baselines:

```bash
python baselines/main_finetune_absolute_pos.py
python baselines/main_finetune_rtd_only.py
python baselines/main_finetune_mlm_only.py
```

## Smoke Tests

Run quick functional tests:

```bash
python test/test_pretrain.py
python test/test_finetune.py
```

Note: test scripts are lightweight functional checks, not full benchmark runs.

## Notes on Results and Interpretation

- Reported metrics in this repository should be interpreted as **thesis experiment results**, not leaderboard-oriented optimization.
- Comparison runs are intended to support analysis and discussion in the thesis.
- Reproducibility may depend on hardware, random seeds, and exact dataset snapshot/version.

## License

This project is distributed under the terms in `LICENSE`.
