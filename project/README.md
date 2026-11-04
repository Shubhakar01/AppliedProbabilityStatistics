# Landslide Susceptibility Classification

A Streamlit machine-learning application that classifies environmental
conditions as **landslide** or **no landslide** using Random Forest and
Logistic Regression.

The application uses `landslide_dataset.csv` as its source dataset. It creates
exactly 1,200 SMOTE samples in memory for training only; synthetic rows are
never added to the test set or written back to a CSV file.

> **Important:** Model results are specific to this dataset and split. They
> should not be interpreted as validated geological or field-deployment
> performance.

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, preprocessing, SMOTE, model training, evaluation, and prediction |
| `landslide_dataset.csv` | Active supplied dataset used by the application |
| `landslide_datasetS.csv` | Separate dataset not used by the application |
| `generate_landslide_dataset.py` | Creates a separate reproducible example dataset |
| `requirements.txt` | Python dependencies |
| `PROJECT_REPORT.txt` | Detailed technical project report |
| `README.md` | Project documentation |

## Dataset

The active `landslide_dataset.csv` contains:

- 3,200 original observations
- 9 input features
- 1 target column
- 2,193 class-0 observations (`No Landslide`)
- 1,007 class-1 observations (`Landslide`)
- 20 missing values in each feature column
- No missing target values

### Input Features

| Feature | Description |
|---|---|
| `Rainfall_mm` | Rainfall measurement |
| `Slope_Angle` | Slope angle |
| `Soil_Saturation` | Soil saturation level |
| `Vegetation_Cover` | Vegetation coverage |
| `Earthquake_Activity` | Earthquake activity level |
| `Proximity_to_Water` | Proximity to water |
| `Soil_Type_Gravel` | Gravel soil indicator |
| `Soil_Type_Sand` | Sand soil indicator |
| `Soil_Type_Silt` | Silt soil indicator |

### Target

`Landslide` is the binary target:

- `0`: No Landslide
- `1`: Landslide

## Preprocessing and Training Workflow

1. Load only `landslide_dataset.csv`.
2. Replace positive and negative infinity with missing values.
3. Remove rows with missing targets.
4. Remove duplicate rows.
5. Convert `Landslide` to integer type.
6. Split the data using a stratified 80/20 train/test split with
   `random_state=42`.
7. Impute missing feature values with the training median.
8. Apply SMOTE only to training data.
9. Train both models.
10. Evaluate predictions on the untouched test set.

### Dataset Split

For the current 3,200-row dataset:

- Original training rows: 2,560
- Held-out test rows: 640
- Original training class 0: 1,754
- Original training class 1: 806
- SMOTE rows added: exactly 1,200
- Effective training rows after SMOTE: 3,760

SMOTE increases the minority class from 806 to 2,006 rows. It is inside the
imbalanced-learn pipeline, so it is applied separately within each
cross-validation training fold and does not leak validation data.

## Models

### Random Forest

```text
n_estimators = 300
max_depth = 8
min_samples_leaf = 5
max_features = "sqrt"
class_weight = "balanced"
random_state = 42
```

Pipeline:

```text
Median imputation -> SMOTE -> Random Forest
```

### Logistic Regression

```text
C = 0.3
max_iter = 1000
random_state = 42
```

Pipeline:

```text
Median imputation -> SMOTE -> RobustScaler -> Logistic Regression
```

## Verified Model Performance

The following results were measured using the active dataset, `random_state=42`,
the 80/20 split, and exactly 1,200 SMOTE rows:

| Model | Test Accuracy | Test ROC-AUC | Mean 5-Fold CV Accuracy |
|---|---:|---:|---:|
| Random Forest | 95.31% | 0.9895 | 95.23% |
| Logistic Regression | 95.16% | 0.9902 | 96.09% |

The application also reports:

- Precision
- Recall
- F1 score
- Cross-validation standard deviation
- Classification reports
- Confusion matrices
- ROC curves
- Random Forest feature importance

## Statistical Evaluation

The application compares both models on the same held-out test rows.

- Bootstrap confidence interval: 5,000 resamples
- Paired permutation test: 10,000 sign permutations
- Random seed: 42

The statistical page reports the Random Forest minus Logistic Regression
accuracy difference, a 95% bootstrap interval, and a paired permutation
p-value.

## Streamlit Pages

### Dashboard

Shows dataset size, feature count, class counts, train/test/SMOTE row counts,
dataset preview, target distribution, correlation matrix, and data-quality
information.

### Model Comparison

Shows metric tables, comparison charts, classification reports, and bootstrap
accuracy intervals.

### Statistical Evaluation

Shows the paired model comparison, confidence interval, and permutation-test
result.

### Confusion Matrices

Displays the confusion matrix for either model.

### ROC Curves

Displays both model ROC curves and AUC values.

### Feature Importance

Displays Random Forest feature importance as a table and chart.

### Predict Susceptibility

Accepts rainfall, slope, soil saturation, vegetation, earthquake activity,
water proximity, and soil type, then returns:

- Predicted class
- Landslide probability
- Illustrative Low, Moderate, or High susceptibility level

The probability categories are illustrative and are not official geological
hazard ratings.

### Dataset

Displays shape, column types, missing values, unique values, IQR outlier
counts, descriptive statistics, duplicate count, and the full active dataset.

## Installation

Clone the repository:

```bash
git clone https://github.com/sangramshitole07/ApsProject.git
cd ApsProject
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Dependencies

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

## Reproducible Example Dataset

`generate_landslide_dataset.py` creates a separate 2,000-row example dataset
using seed `2026`:

```bash
python generate_landslide_dataset.py
```

The generated file is named `generated_landslide_dataset.csv`; the supplied
active `landslide_dataset.csv` is not overwritten.

## Limitations

- The dataset does not include coordinates, GIS raster layers, elevation, or
  validated geological measurements.
- SMOTE creates synthetic training observations; it does not add independent
  field evidence.
- A random split may overestimate geographic generalization.
- Model probabilities are not calibrated geological risk probabilities.
- Real deployment requires independent spatial validation and monitoring.

## Future Improvements

- Add validated georeferenced observations.
- Add elevation, lithology, drainage, land cover, and fault-distance features.
- Use spatial or region-based validation.
- Tune hyperparameters with nested cross-validation.
- Calibrate probabilities.
- Add explainability methods such as permutation importance or SHAP.
- Add model and data versioning.


