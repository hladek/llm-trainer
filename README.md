# llm-trainer

A flexible toolkit for preparing datasets and training transformer-based language models (BERT, T5, etc.) using Hugging Face Transformers and Accelerate. This repository includes scripts for both dataset preprocessing and efficient, distributed training.

---

## Features

- **Dataset Preparation**: Tokenize, chunk, and save datasets for efficient ML training (`prepare_dataset.py`).
- **Accelerated Training**: Distributed, mixed precision, and gradient accumulation enabled training (`run_accel.py`).
- **Custom Schedulers & Optimizers**: Support for Adafactor, AdamW, Muon, SGD-SAI, cosine decay, trapezoid LR schedules, etc.
- **Logging**: Integrated logging (stdout, WandB, JSON logs).
- **Flexible Model & Data Support**: Easily switch between Masked Language Modeling (MLM) and T5-style training.
- **Evaluation & Checkpointing**: Periodic evaluation, logging, and checkpoint saving.
- **Debugging/Profiling**: Profiling of training steps and layer freezing/unfreezing for transfer learning.

Advanced features:

- **Resume Training**:  
  `--resume_path` and `--resume_step` to load previous checkpoints.
- **Mixed Precision**:  
  `--mixed_precision fp16` or `bf16` for faster training.
- **Layer Freezing/Unfreezing**:  
  `--unfreeze layer1,layer2` for transfer learning.

---

## Getting Started


### 1. Install dependencies

- [transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets)
- [accelerate](https://github.com/huggingface/accelerate)
- [torch](https://pytorch.org/)
- [tqdm](https://github.com/tqdm/tqdm)

Install everything:
```bash
pip install transformers datasets accelerate torch tqdm
```

### 2. Prepare tokenizer

You need a tokenizer available via Hugging Face's `AutoTokenizer`, either from a model name (`bert-base-uncased`, etc.) or from a path to a local directory containing the tokenizer files.

An example script to train a tokenizer is in the tokenizers folder.

---

### 3. Prepare Your Dataset

Use `prepare_dataset.py` to tokenize and chunk your raw text data.

```bash
python prepare_dataset.py \
  --output_dir ./output_dataset \
  --dataset_path ./my_data \
  --tokenizer_path bert-base-uncased \
  --max_seq_length 512 \
  --preprocessing_num_workers 8
```

**Dataset Format**:  
Your dataset should be organized as JSON files in a directory, with filenames matching:  
- `train_*` for training  
- `valid_*` for validation  
- `test_*` for testing  
Each line must have a `"text"` field.

---

### 4. Train Your Model

Use `run_accel.py` for distributed training.

```bash
python run_accel.py \
  --output_dir ./training_results \
  --dataset_path ./output_dataset \
  --model_config bert-base-uncased \
  --tokenizer_path bert-base-uncased \
  --per_device_batch_size 8 \
  --optimizer adamw \
  --scheduler cosine \
  --learning_rate 1e-4 \
  --num_warmup_steps 10000 \
  --train_steps 100000 \
  --collator mlm
```

**Main Arguments**:
- `--output_dir`: Save directory for checkpoints/logs.
- `--dataset_path`: Path to processed dataset (`prepare_dataset.py` output).
- `--model_config` / `--tokenizer_path`: Hugging Face model/tokenizer configs or local paths.
- `--per_device_batch_size`: Batch size per GPU/CPU.
- `--optimizer`: Optimizer type (`adamw`, `adafactor`, etc.).
- `--scheduler`: LR scheduler (`cosine`, `constant`, `adafactor`).
- `--collator`: Data collator (`mlm` for BERT/MLM, `t5` for T5).

For a full list of options, run:
```bash
python run_accel.py --help
```


# References

We used these libraries and ideas:

- [Accelerate](https://github.com/huggingface/accelerate) for scaling the training.
- [Datasets](https://github.com/huggingface/datasets) for data preprocessing.
- [Nano T5](https://github.com/PiotrNawrot/nanoT5) training a T5 model from scratch.
- [Moonlight](https://github.com/MoonshotAI/Moonlight) Muon optimizer implementation.
- [SGD Sai](https://github.com/AnonymousAlethiometer/SGD_SaI/) optimizer
- [Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://arxiv.org/abs/2405.18392) trapezoid learning rate scheduler.


