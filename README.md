# Sprint 03 : Deep Learning, NLP & Transformers

Part of the Nolyth Software House AI Bootcamp (Associate AI Engineer track).
Follows directly from [Sprint 02 — Classical ML (Loan Default Prediction)](https://github.com/salamlakhan7/nolyth-ai-bootcamp-sprint02).

## Overview

A practical sprint that moves from classical machine learning into neural networks, convolutional and sequence
models, NLP, attention and Transformers. Everything was implemented hands-on and evaluated honestly: data audits,
leakage checks, unit-tested losses and metrics, controlled experiments, error analysis, and a second training run to
measure how much results move by chance.

The sprint produced three assignments:

| # | Assignment | Task | Headline result | Folder |
|---|---|---|---|---|
| 1 | CNN image classification | ISIC-2019 skin lesions: SVM/KNN → ANN → CNN → CNN + augmentation → transfer learning | Completed. Full results and report in the folder. | [`Assignments/CNN_Assignment_ISIC2019_Classification`](Assignments/CNN_Assignment_ISIC2019_Classification) |
| 2 | Progressive text classification | 20 Newsgroups: ANN, 1D CNN, RNN family, pretrained embeddings, BERT fine-tuning, LLM zero/one/few-shot | Completed. Full results and comparison in the folder. | [`Assignments/nlp-progressive-text-classification2`](Assignments/nlp-progressive-text-classification2) |
| 3a | Candidate assessment, Part A | Binary segmentation, Oxford-IIIT Pet, U-Net from scratch | **Test Dice 0.9028, IoU 0.8320** | [`Assignments/segmentation-unet-oxford-pets`](Assignments/segmentation-unet-oxford-pets) |
| 3b | Candidate assessment, Part B | Multi-task retinal classification, IDRiD (DR grade + DME risk) | **Test avg macro-F1 0.568** (pretrained ResNet-18 + augmentation) vs 0.264 for a CNN from scratch | [`Assignments/classification-idrid-multitask`](Assignments/classification-idrid-multitask) |

Each assignment folder contains its notebook (with saved outputs), a written report where applicable, and a `results/`
folder (metrics as JSON, figures as PNG).

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 (Kaggle and Colab runtimes) |
| Deep learning | PyTorch (segmentation, retinal classification); **TODO:** confirm the framework used for the NLP notebook (TensorFlow/Keras appeared in its GPU memory logs) |
| Pretrained models | torchvision ResNet-18 (ImageNet weights), Hugging Face `transformers` (BERT / DistilBERT) |
| Classical ML baseline | scikit-learn (SVM, TF-IDF, metrics) |
| Data handling | pandas, NumPy, Pillow |
| Visualization | Matplotlib |
| Compute | Kaggle notebooks (Tesla T4) and Google Colab |
| Experiment tracking | Manual logs, JSON result files, saved confusion matrices and curves |

**Why PyTorch for the vision work:** the explicit training loop (forward → loss → backward → optimizer step) makes every
part of the learning cycle visible and easy to unit-test, and it is the native framework for pretrained vision models and
Hugging Face.

## Assignment 3 in detail : Segmentation + multi-task classification

### Part A: U-Net segmentation (Oxford-IIIT Pet)
- **Model:** U-Net trained from scratch, 7,765,985 parameters.
- **Data audit:** 23 images with empty trimaps (broken labels) excluded; 12 non-RGB images converted to RGB. A second
  corrupted label (`Maine_Coon_268`) was caught during error analysis.
- **Split:** 70/15/15, seed 42 → 5,156 / 1,105 / 1,106.
- **Trustworthy metrics:** Dice, IoU and BCE implemented by hand and unit-tested on cases with known answers before any training.
- **Experiments (one factor at a time):** BCE → BCE + Dice → BCE + Dice + augmentation. Best model: the augmented one,
  **test Dice 0.9028, IoU 0.8320**.
- **Error analysis:** four documented failure modes with likely causes.

### Part B: Multi-task retinal classification (IDRiD)
One fundus image gets two labels: diabetic retinopathy (DR) grade (5 classes, 0–4) and diabetic macular edema (DME) risk
(3 classes, 0–2).

**Dataset and split**
- 455 images, 455 label rows; 0 unpaired, 0 duplicate ids, 0 corrupt images; all images 4288×2848.
- The 80 `*test` images are the dataset's **official test split** and were kept untouched. The remaining 375 were split
  85/15 stratified on DR grade (seed 42): **train 318 / val 57 / test 80**.
- Strong class imbalance (DR grade 1: 22 images in total; DME class 1: 48 images). Class weights were computed from the
  training set only.

**Label formulation.** Two independent softmax heads on one shared encoder, loss = CE(DR) + CE(DME). A single 8-unit
sigmoid output would treat mutually exclusive classes as independent labels and could predict two grades at once.

**Experiments (one factor changed at a time)**

| Run | Description | Val avg macro-F1 | Test avg macro-F1 | Test DR macro-F1 | Test DME macro-F1 |
|---|---|---|---|---|---|
| A | CNN from scratch, unweighted loss | 0.3226 | 0.2640 | 0.1449 | 0.3831 |
| B | A + inverse-frequency class weights | 0.3321 | 0.2643 | 0.1640 | 0.3645 |
| C1 | ResNet-18 (ImageNet), frozen backbone, new heads | 0.5214 | 0.5140 | 0.4155 | 0.6125 |
| C2 | C1 + unfreeze `layer4`, lr 1e-4 | 0.5663 | 0.4731 | 0.3276 | 0.6185 |
| D | C2 + mild augmentation | 0.5723 | 0.5680 | 0.4935 | 0.6425 |

(Run 1; test set n = 80. Full table with balanced accuracy and parameter counts is in the assignment's `results/` folder.)

**Reading the results**
- **Pretrained features were the dominant effect** (about 0.5+ vs about 0.26 test avg macro-F1). The from-scratch CNN did not
  beat a majority-class baseline on DR accuracy.
- **Class weighting alone did not help** (A and B tied on test); it changed which classes were predicted, not how well.
- **Fine-tuning `layer4` (C2) overfit quickly**: validation loss rose from about epoch 3 while training loss kept falling, and
  its validation gain did not carry over to the test set.
- **Augmentation (D) delayed but did not prevent overfitting**, and was the best run on both validation and test.
- **Error analysis (model D):** 37 DR errors, 25 of them adjacent-grade; confident under-calling of moderate disease;
  over-grading of exudate-rich images; grade 1 over-predicted; 14 of 48 high-risk DME cases missed. Causes are hypotheses
  (for example small lesions lost at 256×256), not findings.

**Reproducibility check.** Retraining the same code with the same seed changed the test avg macro-F1 by −0.007 (A),
+0.011 (B), 0.000 (C1) and **+0.083 (D: 0.568 → 0.651)**. So differences below roughly 0.1 between single runs on 80 test
images are not conclusive; the pretrained-vs-scratch gap is. All reported numbers are run 1; run 2 is kept as a replicate.

## What I learned

**Data and evaluation**
1. **Audit the data before modeling.** Checking pairing, duplicates, corruption and label validity caught broken Pet
   trimaps and a corrupted label before they could distort results.
2. **Read the dataset's own structure.** The `test` suffix in IDRiD ids was the official split; honoring it avoided leakage.
   Class weights come from training data only, and the test set is evaluated once per experiment.
3. **Pick metrics that expose the failure.** On imbalanced data, accuracy hid minority-class failures; macro F1 and
   per-class results made them visible.
4. **Checkpoint on the task metric, not the loss.** Best validation F1 (or Dice) decides the saved model.

**Engineering practice**
5. **Unit-test losses and metrics** on hand-computable cases before any real training run.
6. **Change one factor at a time**, so each result has a single explanation.
7. **Measure run-to-run variance.** One retrain moved a headline score by 0.08; without that check I would have
   over-read small gaps between experiments.
8. **Domain-aware preprocessing.** For fundus images I avoided hue/saturation changes because colour carries clinical
   signal, and used only mild geometric and brightness/contrast augmentation.
9. **Formulate the problem correctly.** Multi-task (two softmax heads) vs multi-label (sigmoid outputs) is a modeling
   decision with real consequences.
10. **Operational discipline.** Download checkpoints as soon as they appear, version notebooks at milestones, and label
    every result file with the run it came from.

**Concepts covered this sprint:** feed-forward networks, CNNs and transfer learning, RNN / LSTM / GRU / BiLSTM, tokenization
and embeddings (Word2Vec, FastText, BERT), attention and Query/Key/Value, Transformer architecture, BERT and DistilBERT
fine-tuning, LLM zero/one/few-shot evaluation, and segmentation architectures (U-Net).
**TODO:** keep only the items you implemented or want to claim as understood.

## Repo Structure

```
sprint-03-deep-learning-nlp-transformers/
├── README.md
├── requirements.txt
├── .gitignore
├── Assignments/
│   ├── segmentation-unet-oxford-pets/      # Assignment 3a: notebook, report, results/
│   ├── classification-idrid-multitask/     # Assignment 3b: notebook, README, results/
│   ├── TODO-cnn-isic-2019/                 # Assignment 1
│   └── TODO-nlp-progressive-text-classification/   # Assignment 2
├── src/                           # Day-by-day exercises (ANN, CNN, sequence models, embeddings, attention, Transformers)
├── notebooks/                     # Exploratory notebooks
├── data/                          # Datasets (gitignored)
├── models/                        # Saved model weights (gitignored)
└── docs/
    └── progress.md                # Daily progress log / Discord update history
```

## Sprint Roadmap

| Days | Focus | Output |
|---|---|---|
| 1–2 | Neural network foundations | Simple ANN, trained + validated |
| 3–4 | CNN | Image classification model, transfer learning |
| 5–6 | Sequence models | RNN / LSTM / GRU models |
| 7–8 | NLP + embeddings | Text preprocessing + embedding pipeline |
| 9–10 | Attention | Attention concept notebook |
| 11–12 | Transformers | DistilBERT / BERT fine-tuning experiments |
| 13–14 | Project + demo | Final submission package |

## Limitations and Next Steps

- Small test sets (80 images in Part B) and a single seed per run; results should be read with the variance above in mind.
- Retinal images were resized to 256×256 without preserving aspect ratio, which may erase small lesions. Next: higher
  resolution (384/512) and aspect-preserving padding.
- Not yet tried: softer or no class weights, an ordinal loss for DR grades, Grad-CAM to inspect what drives the grade 2 → 4
  errors, and an 8-unit sigmoid head as a controlled comparison with the two-head design.
- These are coursework models and have no clinical validation.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The assignment notebooks were run on Kaggle (GPU enabled); dataset paths inside each notebook point to the Kaggle input
directory.

## Status

✅ Sprint 03 assignments complete: CNN image classification (ISIC-2019), progressive text classification (20 Newsgroups), and the segmentation + multi-task classification assessment (Parts A and B).

See `docs/progress.md` for the running log.

---
Author: Abdul Salam · [GitHub](https://github.com/salamlakhan7) · [Kaggle](https://www.kaggle.com/alexalexandarkhan)
