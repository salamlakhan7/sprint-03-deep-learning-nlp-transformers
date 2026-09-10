# Sprint 03 — Deep Learning, NLP & Transformers

Part of the Nolyth Software House AI Bootcamp (Associate AI Engineer track).
Follows directly from [Sprint 02 — Classical ML (Loan Default Prediction)](#).

## Description

A two-week practical sprint moving from classical machine learning into
neural networks, sequence models, NLP, attention, and Transformer
fundamentals. Every sub-topic is implemented hands-on, not just studied —
each folder under `src/` is a working, runnable exercise tied to a specific
day-range in the official sprint roadmap.

Flagship deliverable: a **text classification comparison** —
Classical ML → LSTM/GRU → fine-tuned DistilBERT — on the same dataset,
demonstrating the full progression from hand-crafted features to
contextual Transformer representations.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Deep learning framework | PyTorch |
| Transformers / pretrained models | Hugging Face `transformers`, `datasets` |
| Classical ML baseline | scikit-learn |
| Data handling | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Experiment tracking | manual logs (`docs/progress.md`) — TensorBoard optional |
| Environment | pip + `requirements.txt` |

**Framework choice rationale:** PyTorch, chosen over TensorFlow/Keras — its
manual training loop (forward → loss → backward → optimizer step) reinforces
the Input → Model → Prediction → Loss → Learning cycle this sprint is meant
to teach, and it's Hugging Face's native framework for the DistilBERT
fine-tuning work in later days.

## Repo Structure

```
sprint-03-deep-learning-nlp-transformers/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── day01_02_ann/              # Neural network foundations — ANN on MNIST
│   ├── day03_04_cnn/              # CNNs — convolution, pooling, transfer learning
│   ├── day05_06_sequence/         # RNN, LSTM, GRU
│   ├── day07_08_nlp_embeddings/   # Tokenization, embeddings
│   ├── day09_10_attention/        # Self-attention, Query/Key/Value
│   └── day11_12_transformers/     # Transformer architecture, DistilBERT fine-tune
├── notebooks/                     # Exploratory notebooks
├── data/                          # Datasets (gitignored)
├── models/                        # Saved model weights (gitignored)
└── docs/
    └── progress.md                # Daily progress log / Discord update history
```

## Sprint Roadmap

| Days | Focus | Output |
|---|---|---|
| 1–2 | Neural Network Foundations | Simple ANN, trained + validated |
| 3–4 | CNN | Basic image classification model |
| 5–6 | Sequence Models | Simple RNN/LSTM/GRU sequence model |
| 7–8 | NLP + Embeddings | Text preprocessing + embedding pipeline |
| 9–10 | Attention | Attention concept demo/notebook |
| 11–12 | Transformers | Small Transformer / DistilBERT fine-tune experiment |
| 13–14 | Project + Demo | Final submission package |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Status

🟡 In progress — currently on **Days 1–2: Neural Network Foundations**

See `docs/progress.md` for the running log.
