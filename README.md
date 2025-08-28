# credit-card-fraud-detection
Kaggle Dataset - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
This project tackles a classic and critical real-world problem in the finance industry using machine learning. The primary goal was to build a predictive model that can accurately identify fraudulent credit card transactions amidst a vast majority of legitimate ones.

The Core Challenge: The dataset is highly imbalanced; only a very small fraction (e.g., 0.17%) of transactions are fraudulent. A naive model that simply predicts "not fraud" for every transaction would be over 99% accurate but completely useless. Therefore, the key challenge was to correctly identify as many fraud cases as possible (maximize recall) while maintaining high precision to avoid too many false alarms.

# My Approach and Steps:

## Data Understanding & Preprocessing:

The dataset contained anonymized features (V1-V28) resulting from a PCA transformation for confidentiality, along with 'Time', 'Amount', and the target 'Class'.

I scaled the 'Time' and 'Amount' features to match the scale of the PCA components using StandardScaler.

Handling Data Imbalance:

I used SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic examples of the minority class (fraud). This created a balanced dataset for training, preventing the model from being biased towards the majority class.

## Model Building and Training:

I split the data into training and testing sets to evaluate performance fairly.

I trained and compared multiple algorithms, including:

Logistic Regression (a strong baseline for classification)

Random Forest Classifier (robust and handles complex relationships well)

XGBoost Classifier (a powerful gradient boosting algorithm often used for winning Kaggle competitions)

## Evaluation and Results:

Since accuracy is misleading, I focused on other metrics:

Precision: Of all transactions predicted as fraud, how many were actually fraud? (We want this high to avoid annoying customers with false declines).

Recall (Sensitivity): Of all the actual fraud transactions, how many did we correctly catch? (This is the most important metric to minimize financial loss).

F1-Score: A harmonic mean of Precision and Recall.

The XGBoost model performed the best, achieving a high recall score, meaning it was excellent at correctly identifying the vast majority of fraudulent transactions.

## Key Skills Demonstrated:
Python (Pandas, NumPy, Scikit-learn, XGBoost), Data Preprocessing, Handling Imbalanced Data (SMOTE), Machine Learning Model Development, Hyperparameter Tuning, Model Evaluation.
