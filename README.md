# Use LoRA to Fine-Tune Qwen2.5-7B for ICD-11 Code Lookup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LiweiWang-Serena/Use-LoRA-to-fine-tune/blob/main/ICD11_LoRA_finetune.ipynb)

Parameter-efficient fine-tuning (QLoRA) of **Qwen2.5-7B-Instruct** to answer questions of the form *"What is the ICD-11 code of {disease}?"*, with a controlled comparison of four key hyper-parameters: **batch size, learning rate, training steps, and LoRA rank**.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Method](#method)
- [Dataset](#dataset)
- [Experimental Design](#experimental-design)
- [Results](#results)
- [Key Findings](#key-findings)
- [Best Configuration](#best-configuration)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Environment](#environment)
- [Limitations](#limitations)
- [Acknowledgments](#acknowledgments)

---

## Overview

General-purpose LLMs know *about* the ICD-11 classification but rarely return the exact code in a usable format. This project fine-tunes a 4-bit quantized Qwen2.5-7B-Instruct model with LoRA adapters on 200 ICD-11 code–title pairs, and evaluates how different hyper-parameter settings affect the model's ability to return the correct code for three test diseases:

| Disease | Expected ICD-11 code |
|---|---|
| Severe dengue | `1D22` |
| Acute appendicitis | `DB10.0` |
| Chronic kidney disease | `GB61` |

The whole pipeline runs on a single free **Google Colab T4 GPU (15 GB)**.

## Key Results

- **Best setting:** batch size **4**, learning rate **4e-4**, steps **200**, rank **32** → **3/3 correct**, lowest training loss (**0.1583**).
- **Training amount (epochs) mattered most.** Only settings trained for 4 epochs produced correct codes; all 1–2 epoch settings scored 0/3.
- **Rank was not the bottleneck.** Rank 8, 32, and 64 gave identical predictions at the same training budget.

## Method

| Component | Choice |
|---|---|
| Base model | [`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct-bnb-4bit) (4-bit, bitsandbytes) |
| Fine-tuning | QLoRA via [Unsloth](https://github.com/unslothai/unsloth) + TRL `SFTTrainer` (supervised instruction tuning) |
| LoRA target modules | All linear layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Prompt format | Qwen chat template applied to each `user` / `assistant` message pair |
| Decoding | Greedy (`do_sample=False`, `max_new_tokens=20`) for deterministic, reproducible evaluation |
| Evaluation prompt | `what is the ICD11 code of {disease}?` |
| Scoring | ICD-11 code extracted from the response with a regular expression and compared to the expected code (exact match) |

The code-extraction pattern follows the ICD-11 stem-code structure (e.g., `1D22`, `DB10.0`, `9C61.01`) and uses word boundaries so the token `ICD11` itself is never mistaken for a code:

```python
CODE_RE = re.compile(r"\b[0-9A-Z][A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,2})?\b")
```

## Dataset

`fine_tuning_data.jsonl` contains **200** examples in chat format, one JSON object per line:

```json
{"messages": [
  {"role": "user", "content": "What is the ICD11 code of Allergic asthma?"},
  {"role": "assistant", "content": "ICD11:CA23.0"}
]}
```

All three test diseases appear in the training data, so this experiment measures how well each setting **memorizes** the target codes (see [Limitations](#limitations)).

## Experimental Design

A one-factor-at-a-time design: starting from a baseline, each setting changes exactly **one** hyper-parameter so its individual effect can be isolated.

| Setting | Batch size | Learning rate | Steps | Rank | Changed |
|---|---|---|---|---|---|
| 1 | 4 | 4e-4 | 100 | 32 | Baseline |
| 2 | **2** | 4e-4 | 100 | 32 | Batch size ↓ |
| 3 | **8** | 4e-4 | 100 | 32 | Batch size ↑ |
| 4 | 4 | **1e-4** | 100 | 32 | Learning rate ↓ |
| 5 | 4 | **8e-4** | 100 | 32 | Learning rate ↑ |
| 6 | 4 | 4e-4 | **50** | 32 | Steps ↓ |
| 7 | 4 | 4e-4 | **200** | 32 | Steps ↑ |
| 8 | 4 | 4e-4 | 100 | **8** | Rank ↓ |
| 9 | 4 | 4e-4 | 100 | **64** | Rank ↑ |

Epochs are computed as `batch_size × steps ÷ 200`. Each setting starts from a freshly loaded base model so all runs share the same starting point.

## Results

### Base model (before fine-tuning)

The base model answered with long explanatory sentences (e.g., *"The International Classification of Diseases (ICD) 11th Revision…"*) and returned **no ICD-11 code** within the generation limit: **0/3**.

### Fine-tuned models

| Setting | Batch | LR | Steps | Rank | Epochs | Final loss | Severe dengue | Acute appendicitis | Chronic kidney disease | Correct |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 4e-4 | 100 | 32 | 2 | 0.2611 | 1F02.2 | DB90.0 | GB60 | 0/3 |
| 2 | 2 | 4e-4 | 100 | 32 | 1 | 0.3759 | 1C02.3 | DD90 | GB80 | 0/3 |
| 3 | 8 | 4e-4 | 100 | 32 | 4 | 0.1747 | 1C84.0 | **DB10.0** | **GB61** | 2/3 |
| 4 | 4 | 1e-4 | 100 | 32 | 2 | 0.3498 | 1A02.1 | DB80.0 | DB80 | 0/3 |
| 5 | 4 | 8e-4 | 100 | 32 | 2 | 0.2488 | 1F02.3 | DB90.0 | GB60 | 0/3 |
| 6 | 4 | 4e-4 | 50 | 32 | 1 | 0.4075 | 1A02.3 | DA00 | BD00 | 0/3 |
| **7** | **4** | **4e-4** | **200** | **32** | **4** | **0.1583** | **1D22** | **DB10.0** | **GB61** | **3/3** |
| 8 | 4 | 4e-4 | 100 | 8 | 2 | 0.2554 | 1F02.3 | DB90.0 | GB60 | 0/3 |
| 9 | 4 | 4e-4 | 100 | 64 | 2 | 0.2658 | 1F02.3 | DB90.0 | GB60 | 0/3 |

Trainable parameters by rank: **r = 8** → 20.2M (0.26%), **r = 32** → 80.7M (1.05%), **r = 64** → 161.5M (2.08%).

## Key Findings

1. **Epochs are the deciding factor.** Only the two 4-epoch settings (3 and 7) produced correct codes; every 1–2 epoch setting scored 0/3.
2. **Training loss tracks accuracy.** The two lowest losses (0.1583, 0.1747) are exactly the two settings with correct answers; every setting with loss above 0.24 scored 0/3.
3. **More updates beat bigger batches.** With the same 800 examples seen (4 epochs), 200 steps × batch 4 (3/3) outperformed 100 steps × batch 8 (2/3).
4. **Rank is not the bottleneck.** Ranks 8, 32, and 64 yielded identical predictions and similar losses (~0.26). Even rank 8 has enough capacity for 200 codes; the limiting factor was training amount.
5. **Too small a learning rate learns too slowly.** At 1e-4 the loss stayed high (0.3498) and predictions drifted to the wrong ICD chapter (e.g., `DB80` for chronic kidney disease). Raising it to 8e-4 did not compensate for limited training.
6. **Learning is progressive.** Every fine-tuned model adopted the `ICD11:XXXX` format, and many wrong answers already had the correct chapter prefix (`DB…` for digestive, `GB…` for genitourinary) or were off by one digit (`GB60` vs. `GB61`): format first, then chapter, then the exact code.

## Best Configuration

```python
batch_size = 4
lr = 4e-4
steps = 200   # = 4 epochs on 200 examples
rank = 32
```

## Getting Started

### Option 1: Google Colab (recommended)

1. Click the **Open in Colab** badge at the top of this page.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Upload `fine_tuning_data.jsonl` via the **Files** panel on the left.
4. Select **Runtime → Run all**. All nine settings finish in roughly 20–30 minutes on a T4.

### Option 2: Local machine

Requires an NVIDIA GPU with at least ~15 GB of VRAM.

```bash
git clone https://github.com/LiweiWang-Serena/Use-LoRA-to-fine-tune.git
cd Use-LoRA-to-fine-tune
pip install unsloth jupyter
jupyter notebook ICD11_LoRA_finetune.ipynb
```

Place `fine_tuning_data.jsonl` in the same folder as the notebook before running.

## Project Structure

```
Use-LoRA-to-fine-tune/
├── ICD11_LoRA_finetune.ipynb   # Full pipeline: setup, training, evaluation, results, report
├── README.md                   # Project documentation
└── fine_tuning_data.jsonl      # 200 ICD-11 code–title pairs (provided separately)
```

The notebook writes `results.csv` after every setting so progress is preserved if a run is interrupted.

## Environment

| Package | Version |
|---|---|
| GPU | NVIDIA Tesla T4 (15 GB), Google Colab |
| Unsloth | 2026.10.7 |
| Transformers | 5.17.0 |
| TRL | 1.15.0 |
| PyTorch | 2.11.0 (CUDA 13.0) |
| Precision | FP16 (T4 does not support BF16) |

## Limitations

- **Memorization, not generalization.** The three test diseases are part of the training data, so the results measure recall of trained codes rather than performance on unseen diseases. In real applications, evaluation should use a held-out test set, and heavy training (4+ epochs) may cause overfitting.
- **Small evaluation.** Only three test cases and a single run per setting; results may vary with different random seeds.
- **Not for clinical use.** This is an educational experiment. Model outputs must not be used for medical coding, diagnosis, or billing without expert verification.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for fast, memory-efficient LoRA fine-tuning
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) by the Qwen team, Alibaba Cloud
- [Hugging Face TRL](https://github.com/huggingface/trl) and [Transformers](https://github.com/huggingface/transformers)
- [ICD-11](https://icd.who.int/) by the World Health Organization
