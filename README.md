# IntelliSpam Classifier

## Overview

IntelliSpam Classifier is a robust SMS spam classification project leveraging Natural Language Processing (NLP) and machine learning techniques. The primary aim is to identify whether a given SMS is spam or ham (legitimate). This project achieves an accuracy of **97.09%** and is powered by the **Multinomial Naive Bayes algorithm**, which outperformed various other classification models tested.

## Features

- **High Accuracy:** Achieves an impressive accuracy of 97.09%.
- **Robust Model:** Multinomial Naive Bayes algorithm provided the best results.
- **Advanced Vectorization:** Utilized TF-IDF vectorization with max features set to 3000 for optimal performance.
- **Confusion Matrix:**
  - **True Positives (TP):** 896
  - **True Negatives (TN):** 108
  - **False Positives (FP):** 0
  - **False Negatives (FN):** 30

## Dataset

The project uses the [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), a collection of 5,574 English SMS messages tagged as spam or ham. This dataset is widely used for SMS spam research.

## Models Tested

The following models were evaluated, with Multinomial Naive Bayes delivering the best performance:

- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **Multinomial Naive Bayes (Best Performer)**
- **Decision Tree Classifier**
- **K-Nearest Neighbors (KNN) Classifier**
- **Random Forest Classifier**
- **AdaBoost Classifier**
- **Bagging Classifier**
- **Extra Trees Classifier**
- **Gradient Boosting Classifier**
- **XGBoost Classifier**

## How It Works

1. **Preprocessing:** The SMS messages are cleaned, tokenized, and stemmed to reduce noise.
2. **Vectorization:** TF-IDF vectorizer converts text data into numerical format for machine learning models.
3. **Classification:** Multinomial Naive Bayes algorithm is applied to classify the messages.
4. **Prediction Output:** Displays whether the message is spam or ham based on the model's prediction.

## Usage

1. **Clone the repository** and navigate to the project directory.
2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
4. **Enter an SMS** in the text area to classify it as spam or ham.

## Results

- **Accuracy:** 97.09%
- **Precision Score:** 1.0
- **Confusion Matrix:**
  ```
  [[896   0]
   [ 30 108]]
  ```

## Key Insights

- **TF-IDF vectorization** with max features set to 3000 provided better results compared to Bag of Words.
- **Multinomial Naive Bayes** outperformed other models for this dataset.

## Acknowledgments

The dataset used in this project is sourced from the [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). Special thanks to the contributors of this dataset for their efforts in compiling such a valuable resource.