# DPO-Safety-Alignment-Phi3

A reproducible safety fine-tuning pipeline for Phi-3-mini using Direct Preference Optimisation (DPO) with QLoRA, including training health monitoring, data validation, and a toxicity/reasoning evaluation harness — all runnable on a single free-tier 16GB T4 GPU (Kaggle or Colab).

---

## What this demonstrates

- **DPO from scratch** — preference dataset construction, reward-free alignment without a separate reward model
- **QLoRA efficiency** — 4-bit quantisation + LoRA adapters, fits a 3.8B model into 16GB VRAM
- **Training health monitoring** — custom callback that halts and checkpoints automatically on NaN/loss spikes
- **Data validation** — `DataValidator` class filters malformed and short samples before training
- **Eval harness** — toxicity scoring + reasoning benchmarks before and after alignment
- **Honest debugging log** — `NOTES.md` documents six real, unplanned failures (dataset schema changes, library API breaks, import shadowing) and how each was fixed

---

## Results summary

| Metric | Base Phi-3-mini | After DPO |
|---|---|---|
| Toxicity rate (mean) | 0.00326 | 0.000186 |
| Reward accuracy (chosen vs rejected) | — | 0.579 |
| MMLU accuracy | 68.1 | 67.8 |

Training time: 5h 03m on a single Kaggle T4. Full numbers and honest commentary in [RESULTS.md](RESULTS.md).

---

## Repository structure

```
DPO-Safety-Alignment-Phi3/
├── train.py              — DPO training with QLoRA, health monitoring, data validation
├── eval_harness.py       — Toxicity + reasoning evaluation before/after alignment
├── requirements.txt      — Exact dependencies for Kaggle/Colab T4 environment
├── RESULTS.md            — Full training metrics, eval results, and honest limitations
├── NOTES.md              — Design decisions, what broke, and how it was debugged
└── results/
    └── eval_results.json — Raw evaluation output
```

---

## Quick start (Kaggle or Colab with T4 GPU)

```bash
git clone https://github.com/Nawfal-AI/DPO-Safety-Alignment-Phi3
cd DPO-Safety-Alignment-Phi3
pip install -r requirements.txt

# 1. Run DPO training (requires 16GB GPU, ~5 hours on T4)
python train.py

# 2. Evaluate toxicity + reasoning
python eval_harness.py
```

On Kaggle: create a notebook with GPU T4 accelerator, clone the repo into `/kaggle/working/`, and run the two commands above.

---

## Pipeline

```
PKU-SafeRLHF preference pairs (chosen / rejected)
    │
    ▼
DataValidator (filter empty / too-short samples)
    │
    ▼
4-bit QLoRA adapter init (Phi-3-mini-4k-instruct)
    │
    ▼
DPO training loop
    ├── TrainingHealthCallback (halt + checkpoint on NaN/spike)
    └── W&B logging (loss, grad norm, reward margin)
    │
    ▼
Merged checkpoint
    │
    ▼
eval_harness.py
    ├── Toxicity classifier (5-prompt red-team set)
    └── Reasoning sanity check (logic + language prompts)
```

---

## Stack

`PyTorch` · `Transformers` · `PEFT` · `TRL` · `bitsandbytes` · `Weights & Biases`

---

## Engineering notes

[NOTES.md](NOTES.md) covers the design decisions (why DPO not PPO, why 4-bit QLoRA, why this sample size) and six real failures hit during development — dataset schema changes, torch runtime corruption, a `trl` API break, and a self-shadowing import — with the fix for each. Worth reading if you want to understand how the project was actually built.

---

## Honest limitations

This demonstrates a working DPO/QLoRA pipeline, correct preference-label handling, and a real (if small) shift toward safer responses on a safety-tuned base model. It does not demonstrate distributed systems experience (single GPU throughout) or robustness against adversarial jailbreak prompts — the eval set is five direct requests, not a red-team suite. See `RESULTS.md` for the full honest take.

---

## License

MIT
