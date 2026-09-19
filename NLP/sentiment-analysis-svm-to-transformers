# 🎬 NLP Zero to Hero: Sentiment Analysis — From Bag-of-Words to Transformers

> A complete, hands-on journey through IMDB movie review sentiment classification — from classical machine learning (SVM) through recurrent neural networks (RNN/LSTM/GRU/BiLSTM) to pre-trained transformers (DistilBERT). Every model is trained, checkpointed, evaluated, and benchmarked against the original reference article.

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-HuggingFace-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-DistilBERT-yellow)](https://huggingface.co/docs/transformers/index)
[![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [The Business Problem](#-the-business-problem)
- [Dataset](#-dataset)
- [Reference Articles](#-reference-articles)
- [Methodology: The Full Pipeline](#-methodology-the-full-pipeline)
  - [Part 1 — Classical ML: SVM with Different Vectorizers](#part-1--classical-ml-svm-with-different-vectorizers)
  - [Part 2 — Recurrent Neural Networks](#part-2--recurrent-neural-networks)
  - [Part 3 — Transformer-Based Models](#part-3--transformer-based-models)
- [Results: My Numbers vs. The Article's Numbers](#-results-my-numbers-vs-the-articles-numbers)
- [Key Insights & Analysis](#-key-insights--analysis)
- [Engineering Practices Applied](#-engineering-practices-applied)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Sample Predictions](#-sample-predictions)
- [What's Next](#-whats-next)
- [Author](#-author)

---

## 🎯 Project Overview

This project is a **from-scratch, end-to-end exploration of sentiment analysis** on the IMDB movie review dataset, systematically comparing **three generations of NLP technique**:

1. **Classical Machine Learning** — Support Vector Machines (SVM) paired with four different text vectorization strategies
2. **Deep Learning (Sequential)** — four flavors of Recurrent Neural Networks, from the simplest to the most sophisticated
3. **Transfer Learning (Transformers)** — a pre-trained DistilBERT model, fine-tuned on the task

Rather than just running code, this project treats it as a **controlled experiment**: every single model is trained on the *same* dataset, evaluated on the *same* test split, and its accuracy is directly compared against a **published reference benchmark** (the source articles). This makes it possible to answer a genuinely interesting question — **does more architectural complexity actually buy you better sentiment classification?**

Spoiler: not always, and the "why" is more interesting than the "what."

---

## 💼 The Business Problem

Sentiment analysis — automatically determining whether a piece of text expresses a positive or negative opinion — is one of the most commercially important NLP tasks in the world. Real applications include:

- **E-commerce**: mining product reviews to surface satisfaction trends at scale
- **Media & entertainment**: understanding audience reception of films, shows, and content
- **Social listening**: tracking brand sentiment across social media in real time
- **Customer support**: auto-routing angry vs. neutral tickets for prioritized handling

The IMDB dataset used here is the "MNIST of sentiment analysis" — a well-known, clean, binary-labeled benchmark that makes it possible to fairly compare wildly different modeling approaches on equal footing.

---

## 🗂 Dataset

**IMDB Dataset of Movie Reviews** (Sentiment Analysis, CSV format)
📦 Source: Kaggle — `columbine/imdb-dataset-sentiment-analysis-in-csv-format`

| Split | Purpose |
|---|---|
| `Train.csv` | Model training |
| `Valid.csv` | Validation / early stopping during training |
| `Test.csv` | Final, held-out evaluation (never seen during training) |

- **Task**: Binary classification — `Positive` (1) or `Negative` (0)
- **Format**: Raw review text + a sentiment label
- **Preprocessing applied**: HTML tag stripping, non-alphabetic character removal, lowercasing, tokenization, stopword removal, and lemmatization (via NLTK)

---

## 📚 Reference Articles

This project directly follows and benchmarks against Prateek Gaurav's **"NLP Zero to Hero"** series on Medium:

| Part | Title | Covers |
|---|---|---|
| **Part 1** | *Introduction, BoW, TF-IDF, Word2Vec* | Text preprocessing, feature extraction, SVM classifiers |
| **Part 2** | *Vanilla RNN, LSTM, GRU & Bi-Directional LSTM* | Sequential deep learning architectures |
| **Part 3** | *Transformer-Based Models* | Pre-trained DistilBERT & RoBERTa |

Every model built in this repository mirrors the article's own approach — same architecture choices, same hyperparameters where specified — so that the accuracy numbers are a fair apples-to-apples comparison, not a different experiment wearing the same name.

---

## 🔬 Methodology: The Full Pipeline

### Part 1 — Classical ML: SVM with Different Vectorizers

The foundational idea of Part 1 is that **before a machine can classify text, text has to become numbers** — and *how* you turn words into numbers has a huge effect on downstream performance. Four vectorization strategies were tested, all feeding into the same classifier (Support Vector Machine with an RBF kernel):

#### 1️⃣ Bag of Words (BoW)
Represents each review as a vector of raw word counts — no notion of grammar, word order, or importance. Simple and surprisingly strong as a baseline.

#### 2️⃣ TF-IDF (Term Frequency–Inverse Document Frequency)
Improves on BoW by **down-weighting** common words that appear in nearly every review (like "movie," "the," "film") and **up-weighting** words that are distinctive to a particular review — capturing which words actually *discriminate* sentiment.

#### 3️⃣ Word2Vec (Custom-Trained)
Instead of counting words, this learns a **dense vector representation** for each word based on the words that tend to surround it (its "context") — trained from scratch on this dataset alone. Each review becomes the *average* of its words' vectors.

#### 4️⃣ Word2Vec (Google News, Pre-Trained)
Same idea as above, but using vectors pre-trained on ~100 billion words of Google News text — bringing in *external* semantic knowledge the model never had to learn itself.

**Why SVM?** SVMs are a strong, well-understood classical baseline for text classification — they find the optimal boundary that separates positive from negative reviews in the vectorized feature space, and they're much cheaper to train than a neural network.

---

### Part 2 — Recurrent Neural Networks

Where Part 1 treats a review as an unordered "bag" of words (even Word2Vec just *averages* word vectors, losing word order), Part 2 introduces architectures that read text **sequentially, word by word**, maintaining a memory of what came before.

#### 🔹 Vanilla RNN
The simplest recurrent architecture. At each time step, it combines the current word with a "hidden state" carrying information from all previous words. Its core weakness: the **vanishing gradient problem** — over long sequences, the influence of early words fades almost to nothing by the time the model reaches the end of a review, making it hard to learn long-range dependencies (e.g., a "but" that flips the sentiment 40 words later).

#### 🔹 LSTM (Long Short-Term Memory)
Purpose-built to fix the vanishing gradient problem. Introduces a dedicated **memory cell** and three gates — **input, forget, and output** — that let the network explicitly decide what to remember, what to discard, and what to output at each step. This lets LSTMs capture much longer-range dependencies than vanilla RNNs.

#### 🔹 GRU (Gated Recurrent Unit)
A streamlined cousin of the LSTM — merges the gating mechanism down to just two gates (**reset** and **update**), resulting in fewer parameters and faster training, while still substantially solving the vanishing gradient problem. The trade-off is a theoretically smaller representational capacity than LSTM, though in practice the gap is often negligible.

#### 🔹 Bi-Directional LSTM
Runs **two LSTMs simultaneously** — one reading the review forward, one reading it backward — then combines both outputs at every time step. This means the model has access to *both* past and future context when interpreting any given word, which is especially useful in language where meaning often depends on what comes later in a sentence.

**Common pipeline for all four:** Tokenize → pad to fixed length → learnable Embedding layer (converts word IDs into dense vectors, trained jointly with the model) → recurrent layer → Dropout (regularization) → Dense + softmax output.

---

### Part 3 — Transformer-Based Models

Introduced in the landmark 2017 paper *["Attention Is All You Need"](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)* by Vaswani et al., the **Transformer** architecture abandoned recurrence entirely in favor of a mechanism called **self-attention**.

**Why this matters:** RNNs (even LSTMs) process text one token at a time, in order — which is both slow (no parallelization across the sequence) and still imperfect at capturing very long-range relationships. Self-attention instead lets the model directly weigh the relationship between **every pair of words in a sentence simultaneously**, regardless of distance — no more fading signal over long sequences, and far better parallelization during training.

#### DistilBERT
A **distilled** (compressed) version of BERT — trained to mimic the behavior of the full BERT model using ~40% fewer parameters, while retaining ~97% of its language understanding capability. Pre-trained on a massive unlabeled text corpus using self-supervised objectives (predicting masked words), then **fine-tuned** here on the IMDB sentiment task with a classification head on top.

This is the key conceptual shift from Parts 1–2: instead of learning language patterns *from zero* using only this dataset, DistilBERT starts with a rich, general-purpose understanding of English **already baked in**, and only needs to learn the relatively small adjustment of "how does this general understanding map to positive/negative movie sentiment?"

> 📝 *Note: The reference article's Part 3 also covers RoBERTa (a more heavily-trained, larger BERT variant). RoBERTa fine-tuning was initiated in this project on Google Colab's free-tier GPU but was not completed to convergence within a single session due to its significantly higher compute cost at the article's specified learning rate (`2e-6`) — full training was estimated to require 5–10+ hours. This remains a documented next step (see [What's Next](#-whats-next)).*

---

## 📊 Results: My Numbers vs. The Article's Numbers

### Part 1 — SVM + Vectorization Methods

| Method | Article's Accuracy | My Accuracy | Δ |
|---|:---:|:---:|:---:|
| SVM + Bag of Words | 88.04% | ~88.0% | ≈ match |
| SVM + TF-IDF | 90.04% | **90.04%** | ✅ exact match |
| SVM + Word2Vec (custom-trained) | 49.92% | **49.90%** | ≈ match |
| SVM + Word2Vec (Google News, pre-trained) | 85.74% | **85.78%** | +0.04 |

### Part 2 — RNN Family (pre-processed text)

| Model | Article's Accuracy (pre-processed) | My Accuracy | Δ |
|---|:---:|:---:|:---:|
| Vanilla RNN | 82.34% | **86.20%** | +3.86 |
| LSTM | 85.44% | **88.20%** | +2.76 |
| GRU | 86.06% | **87.60%** | +1.54 |
| Bi-Directional LSTM | 85.18% | **88.46%** | +3.28 |

### Part 3 — Transformer-Based Models

| Model | My Accuracy | Notes |
|---|:---:|---|
| DistilBERT (fine-tuned) | **92.80%** | Best-performing model overall; converged in 5 epochs with early stopping |
| RoBERTa | *Not completed* | Training interrupted before convergence — see notes above |

### 🏆 Full Leaderboard (My Results, All Models)

| Rank | Model | Accuracy |
|:---:|---|:---:|
| 🥇 | **DistilBERT** | **92.80%** |
| 🥈 | Bi-Directional LSTM | 88.46% |
| 🥉 | LSTM | 88.20% |
| 4 | GRU | 87.60% |
| 5 | Vanilla RNN | 86.20% |
| 6 | SVM + Word2Vec (Google, pre-trained) | 85.78% |
| 7 | SVM + TF-IDF | 90.04%* |
| 8 | SVM + Bag of Words | ~88.0% |
| 9 | SVM + Word2Vec (custom-trained) | 49.90% |

*\*TF-IDF is genuinely the strongest classical baseline — high enough that it actually outperforms every RNN-family model, a pattern also seen in the original article.*

---

## 🧠 Key Insights & Analysis

**1. My RNN-family models consistently outperformed the article's.**
Across all four recurrent architectures, my results beat the article's pre-processed-text numbers by 1.5–3.9 percentage points. This kind of gap is well within normal run-to-run variance — differences in random weight initialization, the exact train/validation split, and minor library version differences between when the article was written (2023) and now can all shift results by a few points. It's a reminder that **reported benchmarks are a range, not a fixed truth** — reproducing someone else's exact number is genuinely hard, even with "the same" code.

**2. Complexity ≠ automatic performance gains — but only up to a point.**
The article's own headline conclusion was that *none* of its RNN models beat its SVM + BoW/TF-IDF baselines. In my run, TF-IDF (90.04%) still edges out every RNN model, **but every RNN model comfortably beat the article's own RNN numbers**, and DistilBERT beat *everything* — including TF-IDF — by a wide margin (+2.76 points over the best classical model). This paints a more nuanced picture than "simple is always better" or "complex is always better": **within a technique family, small implementation and tuning details matter a lot; across technique families, transfer learning from a pre-trained transformer is a genuine step-change, not just incremental improvement.**

**3. Custom Word2Vec is a cautionary tale about training data size.**
At just ~50% accuracy (essentially a coin flip for a binary task), the custom-trained Word2Vec model is a clear illustration that **word embeddings need a lot of data to learn meaningful semantic structure**. Word2Vec trained from scratch on one dataset's vocabulary — especially combined with the article's approach of training *separate* Word2Vec models for train and test text — simply doesn't have enough signal to produce useful vectors. Swapping in the Google News pre-trained vectors (trained on ~100 billion words) immediately jumped accuracy to ~86%, underscoring how much value pre-trained, externally-learned semantic knowledge can add when your own dataset is comparatively small.

**4. Transformers justify their reputation — with major caveats.**
DistilBERT's 92.8% is the clear best result in this entire project, achieved in just 5 epochs before early stopping. But this came at a real engineering cost: multi-hour training time even on a GPU (vs. minutes for SVM, tens of minutes for RNNs), a much heavier dependency footprint, and — critically — required troubleshooting a **major breaking library change** (Hugging Face's `transformers` v5 dropped TensorFlow/Keras support entirely, requiring a full rewrite of the article's TF-based training code into PyTorch). This is a realistic reflection of production ML work: state-of-the-art results often come with state-of-the-art infrastructure headaches.

**5. Binary classifiers struggle with genuinely neutral text — as they should.**
Testing the trained DistilBERT model on a deliberately ambiguous custom input ("It was okay, nothing special but not bad either.") returned **Negative at 91% confidence** — a reasonable outcome given the model was only ever trained to choose between two classes. This is a useful, honest limitation to understand: a binary sentiment model isn't "wrong" here so much as **forced to make a call it was never designed to make gracefully**. A production system built on this would benefit from either a 3-class (positive/neutral/negative) label scheme or a confidence threshold below which predictions are flagged as uncertain rather than trusted outright.

---

## 🛠 Engineering Practices Applied

Beyond the modeling itself, this project deliberately incorporated production-style engineering discipline, since long-running GPU training in an ephemeral, disconnect-prone environment (Colab) creates real operational risk:

- ✅ **Checkpoint-and-resume pattern** applied to *every* model: each training cell checks for an existing saved model before training, and skips straight to loading if found — turning multi-hour retraining into a multi-second reload on any subsequent run
- ✅ **Persistent storage via Google Drive** so trained artifacts survive runtime disconnects, session timeouts, and even full account switches
- ✅ **Preprocessing pipeline checkpointed separately** from individual models (tokenizer, padded sequences, label encodings) — avoiding redundant NLTK/tokenization work across four separate RNN model cells
- ✅ **Diagnosed and resolved a major breaking dependency change** (TensorFlow → PyTorch migration forced by `transformers` v5) rather than pinning to an outdated library version
- ✅ **Root-caused a silent data pipeline failure** (a malformed Kaggle API authentication call was producing a near-empty 5-row dataset instead of failing loudly) by systematically tracing the notebook's data lineage back to its source
- ✅ **Cross-platform portability** — the same logical pipeline was successfully adapted to run across Google Colab (GPU/TPU) and Kaggle Notebooks (GPU) to work around usage quota limits

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.13 |
| **Classical ML** | scikit-learn (SVM, CountVectorizer, TfidfVectorizer) |
| **Word Embeddings** | Gensim (Word2Vec — custom & pre-trained Google News vectors) |
| **Deep Learning (RNNs)** | TensorFlow / Keras (SimpleRNN, LSTM, GRU, Bidirectional) |
| **Transformers** | PyTorch, 🤗 Hugging Face `transformers` (DistilBERT, Trainer API) |
| **Text Preprocessing** | NLTK (tokenization, stopwords, lemmatization) |
| **Data Handling** | pandas, NumPy |
| **Environments** | Google Colab (GPU/TPU), Kaggle Notebooks (GPU) |
| **Persistence** | Google Drive, `joblib`, native Keras `.save()`, Hugging Face `save_pretrained()` |
| **Dataset Source** | Kaggle API |

---

## 📁 Repository Structure

```
├── nlp-sentiment-analysis-bow-to-transformers.ipynb   # Full end-to-end notebook
├── README.md                                          # You are here
└── (trained model checkpoints tracked separately —
     see "How to Run" for regenerating them locally)
```

> **Note:** Trained model artifacts (`.pkl`, `.keras`, and Hugging Face model folders) are stored on Google Drive rather than committed to this repository, due to file size — checkpoint paths are documented inline in the notebook so they can be regenerated by re-running each cell.

---

## ▶️ How to Run

1. **Clone this repository**
   ```bash
   git clone https://github.com/salamlakhan7/nlp-sentiment-analysis-bow-to-transformers.git
   ```
2. **Open the notebook in Google Colab or Kaggle Notebooks**
3. **Set up Kaggle API credentials** to download the dataset (`KAGGLE_USERNAME` + `KAGGLE_KEY` environment variables, from your Kaggle account → Settings → API)
4. **Run cells sequentially, top to bottom** — each stage is self-contained and checkpointed, so interrupted runs can safely be resumed by re-running from the top (already-completed steps will load instantly instead of retraining)
5. **GPU strongly recommended** for Part 2 (RNN family) and required in practice for Part 3 (DistilBERT) — enable via `Runtime → Change runtime type → GPU`

---

## 💬 Sample Predictions

Using the fine-tuned DistilBERT model:

| Input | Prediction | Confidence |
|---|---|:---:|
| *"This movie was absolutely fantastic, one of the best I've seen this year!"* | Positive | 99.51% |
| *"Complete waste of time, terrible acting and a boring plot."* | Negative | 99.93% |
| *"It was okay, nothing special but not bad either."* | Negative | 91.11% |

---

## 🚧 What's Next

- [ ] Complete RoBERTa fine-tuning (with `fp16=True` mixed-precision to reduce training time on free-tier GPUs)
- [ ] Add a confidence-threshold layer for ambiguous/neutral predictions
- [ ] Experiment with a 3-class (Positive / Neutral / Negative) relabeling for more nuanced real-world use
- [ ] Deploy the best model (DistilBERT) behind a lightweight FastAPI inference endpoint
- [ ] Build a small Gradio/Streamlit demo for interactive testing

---

## 👤 Author

**Abdul Salam**
Backend & AI Systems Engineer — Django · FastAPI · RAG · LLMs

- 🔗 GitHub: [github.com/salamlakhan7](https://github.com/salamlakhan7)
- 💼 LinkedIn: [linkedin.com/in/abdul-salam-501b2025b](https://linkedin.com/in/abdul-salam-501b2025b)
- 📧 Email: salamlakhan7@gmail.com

---

<p align="center">
  <i>Built as part of the Nolyth Software House AI Bootcamp — Associate AI Engineer Track (Sprint 03: Deep Learning, NLP & Transformers)</i>
</p>
