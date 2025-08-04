import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")

st.title("Credit Risk Classification App")

data = pd.read_csv("C:/Users/Acer/Downloads/archive (10)/german_credit_data.csv")

st.write("### Missing Values")
st.write(data.isnull().sum())
data.drop(columns=['Unnamed: 0'], inplace=True)
data.ffill(inplace=True)
st.write("### Duplicated Rows:", data.duplicated().sum())

numeric_cols = ['Age', 'Job', 'Credit amount', 'Duration']
st.write("#### Boxplots Before Handling Outliers")
for col in numeric_cols:
    fig = plt.figure(figsize=(6, 4))
    sns.boxplot(data=data, x=col)
    plt.title(f'Boxplot of {col}')
    st.pyplot(fig)

# Capping
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data[col] = data[col].apply(lambda x: lower if x < lower else upper if x > upper else x)

st.write("#### Boxplots After Handling Outliers")

for col in numeric_cols:
    fig = plt.figure(figsize=(6, 4))
    sns.boxplot(data=data, x=col)
    plt.title(f'Boxplot of {col} after Capping')
    st.pyplot(fig)

# Target encoding
data['Risk'] = data['Risk'].map({'good': 1, 'bad': 0})
data = data.dropna(subset=['Risk'])

# Feature Engineering
data['Credit_per_month'] = data['Credit amount'] / data['Duration']
data['Long_Duration'] = data['Duration'].apply(lambda x: 1 if x > 24 else 0)
data['Has_Job'] = data['Job'].apply(lambda x: 0 if x == 0 else 1)
data['Age_Group'] = pd.cut(data['Age'], bins=[0, 25, 40, 60, 120], labels=['Young', 'Adult', 'Senior', 'Elder'])
threshold = data['Credit amount'].quantile(0.75)
data['High_Credit'] = (data['Credit amount'] > threshold).astype(int)

# One-hot encoding
cat_cols = data.select_dtypes(include=['object', 'category']).columns
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# Split data
X = data.drop('Risk', axis=1)
y = data['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Decision Tree

model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

st.subheader("Decision Tree Classification Report")
st.text(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)
fig = plt.figure()
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

roc_auc_dt = roc_auc_score(y_test, model_dt.predict_proba(X_test)[:, 1])
fpr_dt, tpr_dt, _ = roc_curve(y_test, model_dt.predict_proba(X_test)[:, 1])
st.write("ROC-AUC Score (Decision Tree):", roc_auc_dt)

# Random Forest

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

st.subheader("Random Forest Classification Report")
st.text(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)
fig = plt.figure()
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

roc_auc_rf = roc_auc_score(y_test, model_rf.predict_proba(X_test)[:, 1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, model_rf.predict_proba(X_test)[:, 1])
st.write("ROC-AUC Score (Random Forest):", roc_auc_rf)


# ROC Curve Comparison

fig = plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {roc_auc_dt:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.2f})", linestyle='--')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
st.pyplot(fig)



# User Input Prediction

st.header("🔍 Predict Credit Eligibility")

# Create input form for user to enter their data
with st.form("credit_form"):
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job (0 = unemployed, 1-3 = employed levels)", options=[0, 1, 2, 3])
    credit_amount = st.number_input("Credit Amount", min_value=250, max_value=20000, value=1000)
    duration = st.slider("Duration (months)", min_value=4, max_value=72, value=24)
    
    housing = st.selectbox("Housing", ['own', 'free', 'rent'])
    purpose = st.selectbox("Purpose", ['radio/TV', 'education', 'furniture/equipment', 'new car',
                                       'used car', 'business', 'domestic appliance', 'repairs',
                                       'vacation/others'])
    sex = st.selectbox("Sex", ['male', 'female'])
    saving_accounts = st.selectbox("Saving Accounts", ['little', 'moderate', 'quite rich', 'rich', 'unknown'])
    checking_account = st.selectbox("Checking Account", ['little', 'moderate', 'rich', 'unknown'])
    
    submit = st.form_submit_button("Predict")

if submit:
    # Build input data dictionary
    input_dict = {
        'Age': age,
        'Job': job,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Credit_per_month': credit_amount / duration,
        'Long_Duration': 1 if duration > 24 else 0,
        'Has_Job': 0 if job == 0 else 1,
        'High_Credit': 1 if credit_amount > data['Credit amount'].quantile(0.75) else 0,
        'Age_Group': pd.cut([age], bins=[0, 25, 40, 60, 120], labels=['Young', 'Adult', 'Senior', 'Elder'])[0]
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Add missing categorical columns with zero
    for col in X.columns:
        if col.startswith(('Housing_', 'Purpose_', 'Sex_', 'Saving accounts_', 'Checking account_')):
            input_df[col] = 0

    # Set the appropriate one-hot encoded columns to 1
    input_df[f'Housing_{housing}'] = 1
    input_df[f'Purpose_{purpose}'] = 1
    input_df[f'Sex_{sex}'] = 1
    input_df[f'Saving accounts_{saving_accounts}'] = 1
    input_df[f'Checking account_{checking_account}'] = 1
    input_df[f'Age_Group_{input_dict["Age_Group"]}'] = 1

    # Fill any missing columns with 0 (just in case)
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Match the column order with training data
    input_df = input_df[X.columns]

    # Predict using Random Forest
    prediction = model_rf.predict(input_df)[0]
    prediction_proba = model_rf.predict_proba(input_df)[0][1]

    # Output the result
    if prediction == 1:
        st.success(f"✅ Eligible for credit (Good Risk) — Probability: {prediction_proba:.2f}")
    else:
        st.error(f"❌ Not eligible for credit (Bad Risk) — Probability: {prediction_proba:.2f}")
