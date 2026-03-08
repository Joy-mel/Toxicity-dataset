# Toxic Comment Classification

## Project Overview

This project builds a **machine learning pipeline to classify toxic and non-toxic content** from a dataset. The goal is to automatically identify harmful or toxic data instances using supervised learning techniques.

The workflow includes:

* Data loading and inspection
* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature selection
* Handling class imbalance
* Model training and evaluation

The project is implemented in **Python using Jupyter Notebook**.

---

## Project Structure

```
Toxic-Classification/
│
├── Toxic_classification_by_joymelvine.ipynb   # Main notebook
├── data.csv                                   # Dataset used for training
├── README.md                                  # Project documentation
```

---

## Technologies Used

### Programming Language

* Python

### Libraries

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* imbalanced-learn

---

## Methodology

### 1. Data Loading

The dataset is loaded using **pandas** and basic sanity checks are performed to inspect:

* Dataset shape
* Missing values
* Candidate target columns

### 2. Data Preprocessing

The preprocessing pipeline includes:

* Handling missing values using **SimpleImputer**
* Feature scaling using **StandardScaler**
* Feature selection using **Mutual Information (SelectKBest)**

### 3. Exploratory Data Analysis (EDA)

Visualizations are used to understand the dataset:

* Class balance plots
* Correlation heatmaps
* Feature distribution plots (box + swarm plots)

### 4. Handling Class Imbalance

Since toxic datasets are often imbalanced, **SMOTE (Synthetic Minority Oversampling Technique)** is used to balance the dataset.

### 5. Model Training

Several machine learning models are evaluated:

* Logistic Regression
* Random Forest
* Extra Trees Classifier
* HistGradientBoosting Classifier

Cross-validation (**StratifiedKFold**) is used to ensure reliable performance evaluation.

### 6. Evaluation Metrics

Models are evaluated using:

* **F1 Score**
* **Precision**
* **Recall**
* **ROC-AUC**
* **Classification Report**
* **ROC Curve**

---

## Results

The models are compared based on their classification performance metrics. The best-performing model can then be used for detecting toxic content automatically.

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/toxic-classification.git
cd toxic-classification
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

### 3. Run the notebook

Open Jupyter Notebook and run:

```bash
jupyter notebook
```

Then open:

```
Toxic_classification_by_joymelvine.ipynb
```

---

## Future Improvements

* Hyperparameter tuning for improved model performance
* Testing additional models such as XGBoost or LightGBM
* Deploying the model as a web API
* Integrating the model into a moderation system

---

## Author

**Joy Melvine**

---


(which makes the project look **far more professional for portfolios or job applications**).

