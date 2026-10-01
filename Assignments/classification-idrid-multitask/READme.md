# Part B: Multi-task retinal classification on IDRiD (DR grade + DME risk)

Candidate technical assessment, Part B. One fundus image gets **two** labels:
diabetic retinopathy (DR) grade (5 classes, 0-4) and diabetic macular edema (DME) risk (3 classes, 0-2).

Notebook: `part-b-multitask-classification-idrid.ipynb` (Kaggle, T4 GPU). Part A (U-Net segmentation) is in a separate folder.

## Data and split
- Kaggle dataset `mariaherrerot/idrid-dataset`: 455 images, 455 label rows, 0 unpaired, 0 duplicate ids, 0 corrupt images, all 4288x2848.
- The 80 `*test` images are the **official test split** and stay untouched as the held-out test set.
  The other 375 are split 85/15, stratified on DR grade, seed 42: **train 318 / val 57 / test 80**.
  (The assessment's reference split is 413/103; this Kaggle mirror has 455 images, so the split was rebuilt and documented.)
- Class imbalance is strong: DR grade 1 has 22 images in total (14 train, 5 test); DME class 1 has 48.
- Preprocessing: 256x256 bilinear resize (aspect ratio not preserved), ImageNet normalization. Train augmentation is described per experiment below.

## Label formulation
Two independent softmax heads on a shared encoder, loss = CE(DR) + CE(DME) (weights 1:1). No 8-unit sigmoid output:
classes within a task are mutually exclusive. See the notebook's "Label formulation" section.

## Experiments (one factor changed at a time)
| Run | Description |
|---|---|
| A | CNN from scratch, shared encoder + 2 heads, unweighted CE |
| B | A + inverse-frequency class weights (computed on the train set only) |
| C1 | ResNet-18 (ImageNet), backbone frozen, new 2 heads, weighted CE |
| C2 | C1 weights, `layer4` unfrozen, lr 1e-4 |
| D | C2 + mild augmentation (flip, rotation +-15 deg, shift 5%, zoom 0.9-1.1, brightness/contrast +-10%; no hue/saturation change) |

Checkpoint selection: best average of the two tasks' validation macro-F1. Test set evaluated once per experiment.

## Results (run 1; test set n=80)
| Run | Val avg macro-F1 | Test DR macro-F1 | Test DME macro-F1 | Test avg macro-F1 | DR acc | DME acc |
|---|---|---|---|---|---|---|
| A | 0.3226 | 0.1449 | 0.3831 | 0.2640 | 0.350 | 0.600 |
| B | 0.3321 | 0.1640 | 0.3645 | 0.2643 | 0.275 | 0.4375 |
| C1 | 0.5214 | 0.4155 | 0.6125 | 0.5140 | 0.4625 | 0.6875 |
| C2 | 0.5663 | 0.3276 | 0.6185 | 0.4731 | 0.325 | 0.700 |
| D | 0.5723 | 0.4935 | 0.6425 | 0.5680 | 0.5375 | 0.7125 |

Full table (balanced accuracy, parameter counts): `results/results_table.json`.

**What is robust:** pretrained ResNet-18 clearly beats the from-scratch CNN (about 0.5+ vs 0.26 test avg macro-F1).
**What is not:** differences below about 0.1 between single runs (see the replicate below). C2 overfits quickly (val loss rises from about epoch 3).

## Reproducibility: a second training run
A, B, C1 and D were retrained once with the same code and seed 42 (C2 was not rerun):

| Run | Test avg macro-F1, run 1 | run 2 | diff |
|---|---|---|---|
| A | 0.2640 | 0.2568 | -0.007 |
| B | 0.2643 | 0.2751 | +0.011 |
| C1 | 0.5140 | 0.5140 | 0.000 |
| D | 0.5680 | 0.6505 | +0.083 |

C1 (frozen backbone) reproduced exactly; the runs that update convolution weights did not, plausibly because of
non-deterministic GPU kernels (not tested). Evaluating a saved checkpoint is deterministic: re-evaluating the run-1 D
checkpoint reproduced its numbers exactly.

**All tables, figures and the error analysis in this folder use run 1.** Run-2 files are in `results/run2_replicate/`.

> **Important note about the notebook file:** it was saved after the rerun, so the *training logs and test-evaluation cells*
> for A, B and D show **run-2** numbers (for example D: best val 0.5938, test DR macro-F1 0.5943). The results table, curves,
> qualitative panel and error analysis cells show run 1. Run-1 checkpoints are not in this repository (size).

## Error analysis (model D, run 1)
See the notebook section "Step 4: error analysis". Summary: 37 DR errors, 25 of them adjacent-grade; confident under-calling of
moderate disease (hypothesis: small lesions lost at 256x256); over-grading of exudate-rich images; grade 1 over-predicted
(rare-class weight); 14 of 48 high-risk DME images missed. Per-class results for grade 1 rest on 5 test images and are anecdotal.
Lesion descriptions are non-expert visual impressions and the causes are hypotheses, not findings.

## Limitations and next steps
- 80 test images, single seed per run; run-to-run variation up to 0.08 in test avg macro-F1.
- Images are squashed to 256x256, which may erase small lesions: next, higher resolution (384/512), aspect-preserving padding.
- Softer class weights or an ordinal loss; Grad-CAM to check what drives the grade-2-to-4 errors.
- Optional comparison not run: 8-unit sigmoid head vs the two-head design.
- No clinical validation; this is a coursework model, not a diagnostic tool.

## Files
- `part-b-multitask-classification-idrid.ipynb`: full notebook with outputs (see the note above).
- `results/`: `results_table.json`; `test_results_{C1,C2,D}_run1.json` (per-image predictions and probabilities);
  `confusion_matrices_run1.json` and `cm_*_run1.png` (A and B matrices were transcribed from the saved figures, and reproduce
  the logged metrics exactly; the others come from the JSON files); `training_curves_partB.png`; `qualitative_D_run1.png/.csv`
  (16 test images); EDA and preprocessing figures; `run2_replicate/`.
