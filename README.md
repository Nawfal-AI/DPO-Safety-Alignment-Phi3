# 🧠 DPO-Safety-Alignment-Phi3

> **DPO + QLoRA safety alignment of Phi-3-mini, end-to-end on a single 16GB T4 GPU**

A reproducible safety fine-tuning pipeline for Phi-3-mini using Direct Preference Optimisation (DPO) with QLoRA, including training health monitoring, data validation, and a toxicity/reasoning evaluation harness — all runnable on free-tier Colab hardware.

---

## ✨ What this demonstrates

- **DPO from scratch** — preference dataset construction, reward-free alignment without RLHF
- **QLoRA efficiency** — 4-bit quantisation + LoRA adapters, fits in 16GB VRAM
- **Training observability** — loss curves, gradient norms, learning-rate schedule logged to W&B
- **Eval harness** — toxicity scoring (Perspective API / local classifier) + reasoning benchmarks before/after

---

## 📊 Results (summary)

| Metric | Base Phi-3-mini | After DPO |
|---|---|---|
| Toxicity rate (%) | ~4.2 | ~0.9 |
| Helpfulness (human eval) | baseline | +12% |
| MMLU accuracy | 68.1 | 67.8 |

> Full training logs and W&B report linked in the notebook.

---

## 🚀 Quick start

```bash
git clone https://github.com/Nawfal-AI/DPO-Safety-Alignment-Phi3
cd DPO-Safety-Alignment-Phi3
pip install -r requirements.txt

# 1. Build the preference dataset
python scripts/build_dataset.py

# 2. Run DPO training (requires 16GB GPU)
python train_dpo.py --config configs/phi3_qlora.yaml

# 3. Evaluate
python eval/run_eval.py --model outputs/phi3-dpo
```

Or open `notebooks/DPO_Training_Walkthrough.ipynb` in Google Colab.

---

## 🏗️ Pipeline

```
Raw preference pairs (chosen / rejected)
    │
    ▼
Data validation + quality filtering
    │
    ▼
QLoRA adapter init (4-bit Phi-3-mini)
    │
    ▼
DPO training loop
├── Health monitoring (loss, grad norm)
└── W&B logging
    │
    ▼
Merged checkpoint
    │
    ▼
Eval harness
├── Toxicity classifier
└── Reasoning benchmarks
```

---

## 🔧 Stack

`PyTorch` · `Transformers` · `PEFT` · `TRL` · `bitsandbytes` · `Weights & Biases`

---

## 📄 License

MIT
