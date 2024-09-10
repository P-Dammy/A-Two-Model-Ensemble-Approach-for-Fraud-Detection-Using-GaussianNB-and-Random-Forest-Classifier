# A-Two-Model-Ensemble-Approach-for-Fraud-Detection-Using-GaussianNB-and-Random-Forest-Classifier

## Overview

This project demonstrates a novel two-model ensemble approach to improve classification accuracy in detecting fraudulent transactions. The method employs two distinct machine learning models to classify transactions: 

1. A **Gaussian Naive Bayes (GaussianNB)** model, which handles the initial classification.
2. A **Random Forest Classifier**, which is trained on the misclassified data points from the first model to enhance prediction accuracy.

The predictions from both models are linearly combined, significantly boosting the overall classification performance, especially in identifying fraudulent transactions. This approach is particularly useful in handling imbalanced datasets commonly encountered in fraud detection scenarios.

---

## Project Structure

- **Exploratory Data Analysis (EDA):** 
  - Preprocessing and understanding the dataset, which contains details of online transactions.
  - Data cleaning, including handling missing values and generating descriptive statistics.
  - Visualizing the distribution of transaction types and correlation analysis.

- **Machine Learning Pipeline:**
  - Training a **Gaussian Naive Bayes model** on the balanced dataset.
  - Feature engineering to improve model performance, including generating new features like `amount_relative_to_balance_orig`, `newbalanceDest_rank`, and `average_transaction_amount_dest`.
  - Training a **Random Forest Classifier** on the misclassified data points from the first model, followed by hyperparameter tuning using **GridSearchCV**.

- **Ensemble Modeling:**
  - Combining the predictions of both models using a custom function that prioritizes the "Fraud" label when both models agree.
  - Performance evaluation using accuracy, confusion matrix, and classification report.

- **Deployment:**
  - Models are serialized using **joblib** for future predictions.
  - A function to load the models and preprocess new transaction data for fraud prediction.

---

## Key Features

1. **Balanced Dataset:** 
   - The original dataset was highly imbalanced, with significantly more non-fraudulent transactions. The project includes a random undersampling technique to create a balanced dataset, ensuring fair model training.

2. **Feature Engineering:**
   - New features were engineered to capture critical aspects of the transactions, such as the relation between transaction amounts and account balances.

3. **Two-Model Approach:**
   - A two-phase modeling strategy: the first model classifies the data, while the second model addresses misclassifications to refine the prediction.

4. **Ensemble Model:**
   - Linear combination of both models’ predictions, enhancing the detection of fraud.

---
## Data Loading & Exploratory Data Analysis

- The dataset (`onlinefraud.csv`) gotten from **https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset** is loaded, and initial data exploration is conducted, such as:
  - Checking column names, data types, and missing values.
  - Analyzing the distribution of transaction types.
- Descriptive statistics and visualizations are used to gain insights, including:
  - Pie charts and bar plots to show the distribution of transaction types.
  - Correlation heatmaps to highlight relationships between features.

---

## Data Preprocessing

- The `type` column (categorical) is converted into numerical values for model compatibility.
- **Handling imbalanced data**: Since fraudulent and non-fraudulent transactions are significantly imbalanced, non-fraudulent cases are randomly sampled to match the number of fraudulent cases, creating a balanced dataset for model training.
- **Feature engineering**: New columns are introduced to potentially improve model performance.

---

## Model Training

### First Model: GaussianNB
- A **Gaussian Naive Bayes** model is trained initially on the raw dataset, and later on the dataset with engineered features.
- The model’s performance is evaluated using metrics like:
  - **Accuracy**
  - **Confusion matrix**
  - **Classification report**
- After feature engineering, the model's accuracy shows improvement.

---

## Misclassification Handling

- Misclassified data points from the first model (where the GaussianNB model failed) are extracted.
- These data points are used to train a **Random Forest Classifier** as the second model.
- **Hyperparameter tuning** is performed on the Random Forest model using **GridSearchCV** to optimize model parameters.

---

## Model Combination

- Predictions from both models (GaussianNB and Random Forest) are combined using a custom function `combine_predictions()`.
- The combined model’s performance is evaluated, showing:
  - **Higher accuracy** than either individual model.
  - Improved confusion matrix results.

---

## Saving and Loading Models

- Both models, along with the combination function, are saved as `.pkl` files using the **joblib** library for future use.
- These models can later be reloaded to predict new data without the need for retraining.

---

## Prediction for New Data

- A function is provided to preprocess new input data and make predictions using the saved models.
- The prediction output is displayed as either "Fraud" or "No Fraud."

---

## Key Takeaways

- A two-model approach is employed, combining predictions from both the GaussianNB and Random Forest models to improve classification accuracy.
- **Feature engineering** and **hyperparameter tuning** significantly boost model performance.
- The approach is applied to a real-world fraud detection problem, effectively addressing the class imbalance of fraudulent transactions.
- The final model is capable of making predictions on new input data, providing real-time fraud detection functionality.



## Usage

### 1. Data Preparation

- Download the dataset and place it in the project directory.
- Run the Jupyter notebook to preprocess the data and conduct exploratory data analysis (EDA).

### 2. Model Training

- Run the notebook to train the **Gaussian Naive Bayes (GaussianNB)** and **Random Forest Classifier** models.
- The notebook includes feature engineering to optimize the performance of both models.

### 3. Prediction

- The trained models can be used to predict whether a transaction is fraudulent by utilizing the provided `make_predictions()` function.
- A sample input format is demonstrated within the notebook for ease of use.

### 4. Deployment

- After training, the models are saved in `.pkl` files for future predictions.
- These saved models can be loaded to predict fraud in new transaction data without needing to retrain the models.

---

## Results

- The combined two-model approach significantly improves classification accuracy over either individual model.
- Key performance metrics such as **accuracy**, **precision**, **recall**, and **F1 score** are calculated and provided in the notebook for evaluation.

---

## Conclusion

This project illustrates the power of ensemble modeling in fraud detection, particularly for handling imbalanced datasets. By leveraging the strengths of different machine learning models, the two-model approach achieves higher accuracy and produces more robust predictions.


