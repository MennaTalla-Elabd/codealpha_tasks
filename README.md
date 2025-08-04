# Credit Risk Classification App

This is an interactive web application built using **Streamlit** that predicts whether a person is eligible for credit based on their financial data. The app uses machine learning models like **Decision Tree** and **Random Forest** to classify the credit risk as either `good` or `bad`.

## 📊 Features

- Data cleaning and preprocessing (missing values, duplicates, outliers)
- Feature engineering (e.g., credit per month, age group, etc.)
- Boxplots visualization before and after handling outliers
- Classification using Decision Tree and Random Forest
- Evaluation metrics: Classification Report, Confusion Matrix, ROC-AUC Score
- ROC Curve comparison
- Interactive user input form to predict credit eligibility

## 📁 Dataset

The app uses the [German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) which includes information about individuals such as:

- Age
- Job
- Credit amount
- Duration
- Housing
- Purpose
- Sex
- Saving and checking accounts
- Risk label (target)

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Seaborn & Matplotlib
- Scikit-learn

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/MennaTalla-Elabd/codealpha_tasks.git
   cd codealpha_tasks

2.Install dependencies:
pip install -r requirements.txt

3.Run the app:
streamlit run app.py

✍️ Author
Menna Talla Elabd
GitHub: @MennaTalla-Elabd

