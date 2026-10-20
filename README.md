# Wearable Activity Recognition

Classifying human activity (walking, walking upstairs/downstairs, sitting, standing,
laying) from smartphone accelerometer and gyroscope signals, the same problem behind
activity tracking on wearables.

The focus is honest evaluation: models are scored on **people they have never seen**,
because that is what a real product faces.

> Status: in progress. Results, figures, and the product recommendation will be added here.

## Data

[UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones):
30 volunteers, 6 activities, 50 Hz signals cut into 2.56 s windows (128 samples, 50% overlap).

**Setup:** download the dataset zip from the link above and unzip it so this folder exists:

```
data/UCI HAR Dataset/
```

`data/` is git-ignored. If the folder is missing, the loader tries to download it automatically.

## Usage

```bash
pip install -r requirements.txt
```

```python
from src.har.data import load

d = load()
d.features  # (10299, 561) engineered features per window
d.raw       # (10299, 9, 128) raw sensor channels per window
d.y         # activity labels 0-5 (names in d.activity_names)
d.subject   # person ID 1-30, used to split train/test by person
d.split     # official "train" / "test" assignment
```

## Plan

See [THEORY.md](THEORY.md) for the reasoning behind each step.

1. Majority-class baseline
2. Logistic regression and gradient boosting on engineered features
3. 1D convolutional neural network (PyTorch) on raw signals
4. Subject-grouped vs random cross-validation (measuring leakage)
5. Error analysis, per-person performance, learning curve, confidence thresholds
6. Product recommendation

## Results

_Coming soon._

## What I'd recommend the product team do

_Coming soon._
