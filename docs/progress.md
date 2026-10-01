# Sprint 03 — Progress Log

A running record of what was completed. Entries are grouped by assignment; dates are given only where recorded.

## Assignment 1 — CNN image classification (ISIC-2019)
- Completed: SVM/KNN baselines → ANN → CNN → CNN + augmentation → transfer learning.
- Written report: see `Assignments/CNN_Assignment_ISIC2019_Classification`.

## Assignment 2 — Progressive text classification (20 Newsgroups)
- Completed: ANN (TF-IDF), 1D CNN, RNN family, pretrained embeddings (Word2Vec, FastText, BERT), BERT fine-tuning, LLM zero/one/few-shot evaluation.
- Notebook and results: see `Assignments/nlp-progressive-text-classification2`.

## Assignment 3a — Segmentation (Oxford-IIIT Pet)
- U-Net from scratch (7,765,985 parameters); Dice, IoU and BCE unit-tested before training.
- Data audit: 23 broken-label images excluded, 12 non-RGB images converted.
- Three controlled experiments (BCE → BCE + Dice → BCE + Dice + augmentation); best test Dice 0.9028, IoU 0.8320.
- Error analysis with four documented failure modes.
- Folder: `Assignments/segmentation-unet-oxford-pets`.

## Assignment 3b — Multi-task classification (IDRiD)
- Dataset audit and official-split decision (80 `test` images held out; train 318 / val 57 / test 80).
- Multi-task model with a shared encoder and two softmax heads; loss and metrics unit-tested.
- Experiments A–D (scratch CNN, class weighting, frozen ResNet-18, layer4 fine-tuning, augmentation), each evaluated once on the test set.
- Results table, training curves, 16-image qualitative panel, error analysis (completed 2026-09-30).
- Reproducibility check: A, B, C1 and D retrained once; D's test avg macro-F1 moved from 0.568 to 0.651.
- Folder: `Assignments/classification-idrid-multitask`.
