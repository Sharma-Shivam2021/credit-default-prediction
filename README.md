# Credit Default Prediction

A machine learning project for predicting whether a credit card customer is likely to default on their next payment.

The project uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository and follows an end-to-end machine learning workflow including exploratory data analysis, preprocessing, baseline modelling, cross-validation, hyperparameter tuning, final model evaluation, and model interpretation.

## Problem Statement

Credit default prediction is a binary classification problem in which the objective is to predict whether a customer will default on their next credit card payment.

The target variable is defined as:

- `0` — Non-default
- `1` — Default

Because the dataset is imbalanced, with approximately **22.12%** of customers belonging to the default class, model performance cannot be evaluated using accuracy alone. Therefore, the project considers precision, recall, F1-score, ROC-AUC, and average precision.

## Dataset

The project uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.

The dataset contains:

- **30,000 customer records**
- **23 predictor variables**
- **1 binary target variable**

The predictors contain information about:

- Credit limit
- Demographic characteristics
- Repayment status across previous months
- Monthly bill amounts
- Previous payment amounts

The raw dataset is not stored in the repository. It can be downloaded automatically using:

```bash
python src/download_data.py
```

## Methodology

The project was developed in three main stages.

### 1. Exploratory Data Analysis

The dataset was examined to understand its structure, feature distributions, class imbalance, and relationships between the predictors and credit default.

The analysis included:

- Target class distribution
- Distribution of numerical and categorical features
- Default rates across categorical variables
- Repayment-status analysis
- Bill amount and payment amount distributions
- Correlation analysis
- Identification of undocumented categorical values

The exploratory analysis showed that repayment-status variables, particularly recent repayment behaviour, were strongly associated with credit default.

### 2. Data Preprocessing and Baseline Model

The preprocessing stage included:

- Mapping undocumented `EDUCATION` categories (`0`, `5`, and `6`) to the `Others` category
- Mapping undocumented `MARRIAGE = 0` to the `Others` category
- Removing `ID` from the predictor variables
- One-hot encoding categorical and repayment-status features
- Standardizing continuous numerical features
- Performing an 80/20 stratified train-test split

All fitted preprocessing operations were performed through Scikit-learn pipelines to prevent data leakage.

Logistic Regression was used as the initial baseline model. The baseline achieved an accuracy of approximately **81.68%**, but recall for the default class was only **35.12%**, demonstrating why accuracy alone was insufficient for this imbalanced classification problem.

### 3. Model Comparison, Tuning, and Evaluation

Five classification algorithms were compared using **5-fold stratified cross-validation**:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- RBF Support Vector Machine

The models were evaluated using precision, recall, F1-score, ROC-AUC, and average precision.

After the initial comparison, Logistic Regression and Random Forest were selected for hyperparameter tuning. F1-score was defined as the primary model-selection metric to balance precision and recall for the default class.

The tuned Random Forest achieved the highest cross-validation F1-score and was therefore selected as the final model.

## Results

### Baseline Model

The initial Logistic Regression baseline produced the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 0.8168 |
| Precision | 0.6619 |
| Recall | 0.3512 |
| F1-score | 0.4589 |
| ROC-AUC | 0.7580 |
| Average Precision | 0.5275 |

Although the baseline achieved relatively high accuracy, its low recall showed that many actual defaulters were not identified.

### Final Model

After model comparison and hyperparameter tuning, a Random Forest classifier was selected using cross-validation F1-score as the primary selection criterion.

The selected configuration was:

- `n_estimators = 200`
- `max_depth = None`
- `min_samples_leaf = 10`
- `class_weight = "balanced"`

The final model achieved the following performance on the previously untouched test set:

| Metric | Score |
|---|---:|
| Accuracy | 0.7890 |
| Precision | 0.5211 |
| Recall | 0.5667 |
| F1-score | 0.5430 |
| ROC-AUC | 0.7761 |
| Average Precision | 0.5529 |

The final confusion matrix was:

| | Predicted Non-Default | Predicted Default |
|---|---:|---:|
| **Actual Non-Default** | 3982 | 691 |
| **Actual Default** | 575 | 752 |

Compared with the initial Logistic Regression baseline, the selected Random Forest substantially improved recall for the default class from **35.12% to 56.67%** and increased the F1-score from **0.4589 to 0.5430**.

This improvement came at the cost of lower overall accuracy and precision, illustrating the trade-off involved when improving minority-class detection in an imbalanced classification problem.

## Model Interpretation

The final Random Forest model was interpreted using two complementary techniques:

1. Random Forest impurity-based feature importance
2. Permutation importance measured using F1-score

Both methods identified the most recent repayment status, `PAY_0`, as a particularly influential predictor.

Permutation importance showed that shuffling `PAY_0` resulted in an average F1-score decrease of approximately **0.0848**, substantially larger than the decrease produced by any other individual feature.

Other repayment-status, previous-payment, and billing variables also contributed to prediction.

The interpretation also demonstrated that impurity-based and permutation importance can produce different feature rankings. For example, `LIMIT_BAL` received relatively high impurity-based importance but much lower permutation importance.

Feature importance in this project represents the model's predictive reliance on a feature and should not be interpreted as evidence of a causal relationship with credit default.

## Repository Structure

```text
credit-default-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_credit_default_eda.ipynb
│   ├── 02_preprocessing_and_baseline.ipynb
│   └── 03_model_comparison.ipynb
│
├── src/
│   └── download_data.py
│
├── .gitignore
├── README.md
└── requirements.txt
```
### Notebook Description

The project is organized into three notebooks, each representing a different stage of the machine learning workflow.

#### `01_credit_default_eda.ipynb`

Performs exploratory data analysis to understand the dataset before modelling.

The notebook includes:

- Dataset structure and feature inspection
- Target class distribution
- Analysis of numerical and categorical variables
- Default rates across demographic categories
- Repayment-status analysis
- Bill amount and payment amount analysis
- Correlation analysis
- Identification of undocumented categorical values

#### `02_preprocessing_and_baseline.ipynb`

Builds the preprocessing workflow and establishes the initial machine learning baseline.

The notebook includes:

- Cleaning undocumented categorical values
- Feature and target separation
- Stratified train-test splitting
- One-hot encoding of categorical features
- Standardization of continuous numerical features
- Logistic Regression baseline
- Class-weight experiment
- Classification-threshold experiment
- ROC-AUC and average precision evaluation

#### `03_model_comparison.ipynb`

Performs model comparison, hyperparameter tuning, final evaluation, and model interpretation.

The notebook includes:

- 5-fold stratified cross-validation
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- RBF Support Vector Machine
- Hyperparameter tuning using GridSearchCV
- Final model selection using F1-score
- Evaluation on the held-out test set
- Random Forest feature importance
- Permutation feature importance


## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Sharma-Shivam2021/credit-default-prediction
cd credit-default-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

The raw dataset is not stored directly in the repository.

Run:

```bash
python src/download_data.py
```

The script downloads and extracts the dataset into:

```text
data/raw/
```

### 4. Run the Notebooks

The notebooks should be executed in the following order:

```text
1. notebooks/01_credit_default_eda.ipynb
2. notebooks/02_preprocessing_and_baseline.ipynb
3. notebooks/03_model_comparison.ipynb
```

The notebooks can be executed using Jupyter Notebook, JupyterLab, or Google Colab.


## Key Findings

The experiments produced several important observations:

- The dataset is imbalanced, with approximately **22.12%** of customers belonging to the default class.
- Accuracy alone was therefore not sufficient for evaluating model performance.
- Recent repayment behaviour was particularly informative for predicting default.
- `PAY_0`, representing the most recent repayment status, was consistently identified as an influential predictor.
- The unrestricted Decision Tree showed substantial overfitting.
- Hyperparameter tuning and class weighting substantially improved the Random Forest's ability to identify customers belonging to the default class.
- The final Random Forest increased default-class recall from approximately **35.12%** for the initial Logistic Regression baseline to **56.67%**.
- The final F1-score increased from approximately **0.4589** to **0.5430**.
- Cross-validation F1 (**0.5462**) and final test F1 (**0.5430**) were close, indicating consistent performance between validation and the held-out test set.


## Limitations

The results should be interpreted with several limitations in mind:

- The dataset contains credit card customer data from Taiwan collected in 2005. Results may therefore not generalize directly to customers from different populations, financial systems, or time periods.
- The dataset is imbalanced, which makes the classification of the default class more challenging.
- Hyperparameter tuning was performed over a limited search space rather than every possible configuration.
- Several financial variables are strongly correlated, which can affect feature-importance interpretation.
- Impurity-based and permutation feature importance measure predictive reliance and do not establish causal relationships.
- The selected classification threshold was not independently optimized using a dedicated validation set.
- The model was developed as a machine learning experiment and should not be treated as a production-ready credit-risk decision system.


## Future Work

Possible extensions of the project include:

- Engineering features such as the number of delayed-payment months
- Creating credit-utilization-related features
- Performing more extensive hyperparameter optimization
- Evaluating boosting algorithms such as Gradient Boosting or XGBoost
- Optimizing the classification threshold using a dedicated validation strategy
- Evaluating probability calibration
- Investigating additional model-interpretability techniques such as SHAP
- Evaluating the workflow on newer or additional credit-risk datasets


## Conclusion

This project developed an end-to-end machine learning workflow for credit default prediction, beginning with exploratory data analysis and progressing through preprocessing, baseline modelling, cross-validation, hyperparameter tuning, final evaluation, and model interpretation.

The experiments demonstrated that selecting a model based solely on accuracy would be inappropriate for this imbalanced classification problem. Although the initial Logistic Regression model achieved higher overall accuracy, the tuned Random Forest provided substantially better detection of customers in the default class.

Using F1-score as the predefined model-selection criterion, the tuned Random Forest was selected as the final model. It achieved a test F1-score of **0.5430**, recall of **0.5667**, ROC-AUC of **0.7761**, and average precision of **0.5529**.

Feature-importance analysis further indicated that recent repayment behaviour, particularly `PAY_0`, played an important role in the model's predictions.

Overall, the project demonstrates the importance of selecting evaluation metrics and validation procedures according to the characteristics of the machine learning problem rather than relying on accuracy alone.
