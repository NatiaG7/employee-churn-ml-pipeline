# Employee churn dataset

The training file is **not** committed to git (size + common bootcamp dataset).

## Dataset

**IBM HR Analytics Employee Attrition** — usually saved as `HR_comma_sep.csv`

| Column | Description |
|--------|-------------|
| `left` | **Target** — 1 = employee left, 0 = stayed |
| `satisfaction_level`, `last_evaluation`, `number_project`, … | Numeric features |
| `sales`, `salary` | Categorical (one-hot encoded in pipeline) |

## Setup

1. Download from [Kaggle: IBM HR Analytics Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)  
   — or use a copy from your bootcamp files.

2. Place the file here:

```
data/raw/HR_comma_sep.csv
```

3. Train (writes `models/churn_model.pkl` locally; not committed to git):

```bash
python scripts/train.py
```

Committed artifacts from a full train include `outputs/metrics.json` and `outputs/figures/*.png` (see root [README.md](../README.md)).

## Local copy (if you already have the file)

```bash
cp ~/Downloads/HR_comma_sep.csv data/raw/HR_comma_sep.csv
```

## Note

`Churn_Modeling.csv` on your Desktop is a **different** dataset (customer churn for a bank). This project uses **HR_comma_sep.csv** with the `left` column.
