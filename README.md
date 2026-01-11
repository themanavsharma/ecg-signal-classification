# ECG Signal Classification (PyTorch)

Binary ECG heartbeat classifier trained on the ECG5000 dataset in UCR/UEA `.ts` time-series format.

## Task
This repo maps the original ECG5000 labels to a binary problem:
- Class 0 (normal): original label `1`
- Class 1 (abnormal): original labels `2-5`

## Results (test set)
From `python evaluate.py`:
- AUROC: 0.9156
- Accuracy: 0.8831
- Confusion matrix:
```
[[2369 258]
[ 268 1605]]
```

## Repo layout
- `train.py` trains the model and saves the best checkpoint
- `evaluate.py` reports AUROC, confusion matrix, and a classification report on the test set
- `infer.py` runs inference on a single sample
- `src/dataset.py` loads `.ts` files, applies the binary label mapping, and standardizes inputs
- `src/model.py` defines a simple 1D CNN

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data
Download the dataset files and place them here:
- `data/ECG5000_TRAIN.ts`
- `data/ECG5000_TEST.ts`

## Train
```bash
python train.py
```
This writes `checkpionts/best.pt`. The checkpoint includes:
- model weights
- test-split mean and std used for standardization

## Evaluate
```bash
python evaluate.py
```

## Inference
```bash
python infer.py
```