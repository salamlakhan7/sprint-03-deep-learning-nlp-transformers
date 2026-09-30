# Part A: Binary Segmentation with U-Net

Oxford-IIIT Pet dataset — candidate technical assessment.

## Contents
- [`part-a-segmentation-unet-oxford-pets.ipynb`](./part-a-segmentation-unet-oxford-pets.ipynb) — full notebook, all cells run with saved outputs
- [`Part_A_Report.pdf`](./Part_A_Report.pdf) — written report: dataset audit, preprocessing, U-Net architecture, loss/metric unit tests, 3 controlled experiments, qualitative + error analysis, limitations, next steps
- [`results/`](./results) — results table (JSON) and all figures (EDA, preprocessing check, qualitative grid) referenced in the report

## Summary

| Item | Value |
|---|---|
| Dataset | Oxford-IIIT Pet (images + trimap annotations), Kaggle mirror by `julinmaloof` |
| Usable pairs | 7,367 of 7,390 (23 excluded — confirmed broken/empty ground-truth labels) |
| Split | 70/15/15, fixed seed 42 — 5,156 train / 1,105 val / 1,106 test |
| Model | U-Net from scratch, 4 encoder levels + bottleneck, 7,765,985 parameters |
| Best result | **Test Dice 0.9028, Test IoU 0.8320** (Experiment C: BCE+Dice loss + augmentation) |

## Experiments (one factor changed at a time)

| Experiment | Change | Val Dice | Test Dice | Test IoU |
|---|---|---|---|---|
| A — Baseline | BCE loss, no augmentation | 0.8902 | 0.8937 | 0.8181 |
| B — Loss change | BCE+Dice loss | 0.8942 | 0.8969 | 0.8237 |
| C — + Augmentation | BCE+Dice + flip/brightness/contrast | 0.8969 | **0.9028** | **0.8320** |

Full reasoning, unit tests, qualitative grid and error analysis are in `Part_A_Report.pdf`.

## Dataset source
[Oxford-IIIT Pet Dataset With Annotations](https://www.kaggle.com/datasets/julinmaloof/the-oxfordiiit-pet-dataset) (Kaggle), original data from Parkhi, Vedaldi, Zisserman & Jawahar, University of Oxford VGG.

## Notes
- Model checkpoints (`.pt` files) are not included in this repo (large binary files); available on request or re-trainable from the notebook (seed 42, ~58 min/experiment on a T4 GPU).
