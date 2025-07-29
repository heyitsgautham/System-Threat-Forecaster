# 💻 Final Project: SYSTEM THREAT FORECASTER

Developed an end-to-end machine learning pipeline to predict a binary outcome from structured tabular data. The pipeline included robust preprocessing techniques for handling missing values, categorical encoding (One-Hot and Ordinal), and feature scaling. Applied SelectKBest with mutual information to extract the most informative features before model training.

Trained and evaluated a diverse set of classification models including Logistic Regression, Random Forest, XGBoost, LightGBM, and others. Performed hyperparameter tuning using GridSearchCV and RandomizedSearchCV with early stopping and cross-validation to optimize model performance. 



## 📂 Project Structure

- `train_df.csv` – Raw training data
- `submission.csv` – Final prediction file for submission
- `notebooks/` – Jupyter Notebooks for experimentation
- `models/` – Trained models and performance comparisons

---

## 📌 Project Goals

- Clean and preprocess real-world structured data
- Engineer and select informative features
- Build and evaluate a wide range of classification models
- Optimize model performance via hyperparameter tuning
- Prepare results for submission (e.g., on Kaggle)

---

## 🧪 Dataset Summary

- **Problem Type:** Binary Classification
- **Target Variable:** (Not specified here; assumed binary)
- **Feature Types:** Numerical, Categorical (Nominal + Ordinal), Missing Values

---

## 📘 Topics Covered

### 🧹 1. Data Preprocessing
- Handled missing values using `SimpleImputer`
- Categorical encoding:
  - `OneHotEncoder` (for nominal features)
  - `OrdinalEncoder` (for ordinal features)
- Converted mixed-type columns (`str` + `float`) to consistent `str` type
- Applied `StandardScaler` to scale numerical and ordinal data
- Combined all transformations using `ColumnTransformer`
- Exported preprocessed data to `.csv` for reproducibility

---

### 🧠 2. Feature Selection
- Used `SelectKBest` with `mutual_info_classif` to rank and select top 100 features
- Selected features from raw `train_df` for interpretation
- Transformed training and validation sets using the same selector

---

### 🧪 3. Model Training
Trained and evaluated the following models:
- ✅ Logistic Regression  
- ✅ Decision Tree  
- ✅ Random Forest  
- ✅ XGBoost  
- ✅ LightGBM  
- ✅ AdaBoost  
- ✅ Gradient Boosting  
- ✅ K-Nearest Neighbors  
- ✅ Extra Trees  
- ✅ Ridge Classifier  
- ✅ SGD Classifier

Handled class imbalance using `class_weight='balanced'` in supported models.

---

### 🎯 4. Model Evaluation
- Evaluated on hold-out validation set
- Used **accuracy** as primary scoring metric
- Exported submission predictions in `submission.csv`

---

### ⚙️ 5. Hyperparameter Tuning
- Applied `RandomizedSearchCV` and `GridSearchCV` with:
  - Parameters: `learning_rate`, `num_leaves`, `max_depth`, `min_data_in_leaf`, etc.
  - LightGBM callbacks: `early_stopping`, `log_evaluation`
- Used validation set as `eval_set` during model fitting
- Optimized model generalization by tuning `n_estimators` and tree complexity

---

## 📈 Results

- Best model: **LightGBM with tuned hyperparameters**
- Validation Accuracy: *0.63290*
- Final submission generated using selected top-performing model

---

## 🚀 How to Run

```bash
# Install requirements
pip install -r requirements.txt

# Run training script or open the notebook
jupyter notebook notebook.ipynb

# Generate submission
python generate_submission.py

```



## ✅ Requirements

- Python 3.7+
- Jupyter Notebook
- `pandas`, `numpy`, `matplotlib`, `seaborn`

## 🧾 Notes

- Author: GAUTHAM KRISHNA S `23f2000466`
- Term: T1 2025
- Dataset source: *(You can mention the source if available, or write "proprietary/in-class" if applicable)*

## 📬 Contact

For any queries, contact: heyitsgautham@gmail.com
