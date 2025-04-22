import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import plotly.express as px
import os

# Load and Preprocess Data
def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"Dataset not found at {file_path}. Please upload creditcard.csv.")
        return None, None, None, None
    
    df = pd.read_csv(file_path)
    expected_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)] + ['Class']
    if not all(col in df.columns for col in expected_cols):
        st.error("Dataset must contain Time, Amount, V1-V28, and Class columns.")
        return None, None, None, None
    
    feature_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    st.write(f"Number of feature columns: {len(feature_cols)}")  # Debug: should be 29
    X = df[feature_cols]
    y = df['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.write(f"Shape of X: {X.shape}")  # Debug: should be (n_rows, 29)
    return X_scaled, y, scaler, feature_cols

# Train and Evaluate Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_pred)
    return model, report, auc, X_test, y_test, y_pred, y_prob

# Streamlit App Layout
st.title("Real-Time Credit Card Fraud Detection System")
st.write("Upload the creditcard.csv dataset to analyze transactions and detect fraud.")

# File Upload Section
uploaded_file = st.file_uploader("Upload transaction data (CSV)", type="csv")

# Process Data Button
if st.button("Process Data") and uploaded_file:
    with open("temp_creditcard.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    X_scaled, y, scaler, feature_names = load_and_preprocess_data("temp_creditcard.csv")
    if X_scaled is not None:
        st.session_state['X_scaled'] = X_scaled
        st.session_state['y'] = y
        st.session_state['scaler'] = scaler
        st.session_state['feature_names'] = feature_names
        st.write(f"Number of feature names: {len(feature_names)}")  # Debug: should be 29
        st.success("Data loaded and preprocessed successfully!")
        os.remove("temp_creditcard.csv")

# Train Model and Predict Button
if 'X_scaled' in st.session_state and st.button("Train Model and Predict"):
    X_scaled = st.session_state['X_scaled']
    y = st.session_state['y']
    scaler = st.session_state['scaler']
    feature_names = st.session_state['feature_names']
    
    model, report, auc, X_test, y_test, y_pred, y_prob = train_model(X_scaled, y)
    st.session_state['model'] = model
    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test
    
    # Display Model Performance
    st.subheader("Model Performance")
    st.write(f"**ROC-AUC Score**: {auc:.4f}")
    st.write("**Classification Report**:")
    st.json(report)
    
    # Display Fraudulent Transactions
    results_df = pd.DataFrame({
        'Transaction_ID': range(len(y_test)),
        'Actual': y_test,
        'Predicted': y_pred,
        'Fraud_Probability': y_prob
    })
    fraud_df = results_df[results_df['Predicted'] == 1]
    
    st.subheader("Fraudulent Transactions")
    if not fraud_df.empty:
        st.dataframe(fraud_df[['Transaction_ID', 'Fraud_Probability']].style.format({'Fraud_Probability': '{:.4f}'}))
    else:
        st.write("No fraudulent transactions detected.")
    
    # Plot Fraud Probability Distribution
    st.subheader("Fraud Probability Distribution")
    fig1 = px.histogram(results_df, x='Fraud_Probability', color='Predicted', nbins=50,
                        title="Fraud Probability Distribution")
    st.plotly_chart(fig1)
    with open("fraud_probability.html", "w") as f:
        f.write(fig1.to_html())
    st.download_button("Download Fraud Probability Plot (HTML)", 
                      data=open("fraud_probability.html", "r").read(),
                      file_name="fraud_probability.html")
    
    # Plot Feature Importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    fig2 = px.bar(importance, x='Feature', y='Importance', title="Feature Importance")
    st.plotly_chart(fig2)
    with open("feature_importance.html", "w") as f:
        f.write(fig2.to_html())
    st.download_button("Download Feature Importance Plot (HTML)", 
                      data=open("feature_importance.html", "r").read(),
                      file_name="feature_importance.html")

# Simulate Real-Time Transaction
st.subheader("Simulate Real-Time Transaction")
with st.form("transaction_form"):
    time = st.number_input("Time (seconds)", min_value=0.0, value=80000.0)
    amount = st.number_input("Amount", min_value=0.0, value=2000.0)
    v_features = []
    for i in range(1, 29):
        if f'V{i}' == 'V14':
            v_features.append(st.number_input('V14', value=-12.0))
        elif f'V{i}' == 'V10':
            v_features.append(st.number_input('V10', value=-10.0))
        elif f'V{i}' == 'V12':
            v_features.append(st.number_input('V12', value=-10.0))
        elif f'V{i}' == 'V17':
            v_features.append(st.number_input('V17', value=-15.0))
        elif f'V{i}' == 'V3':
            v_features.append(st.number_input('V3', value=-7.0))
        elif f'V{i}' == 'V4':
            v_features.append(st.number_input('V4', value=4.0))
        elif f'V{i}' == 'V11':
            v_features.append(st.number_input('V11', value=3.0))
        else:
            v_features.append(st.number_input(f'V{i}', value=0.0))
    submitted = st.form_submit_button("Predict")
    
    if submitted and 'model' in st.session_state and 'feature_names' in st.session_state:
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        feature_names = st.session_state['feature_names']
        transaction = pd.DataFrame(
            [[time, amount] + v_features],
            columns=feature_names
        )
        st.write(f"Transaction features: {len(transaction.columns)}")  # Debug: should be 29
        transaction_scaled = scaler.transform(transaction)
        pred = model.predict(transaction_scaled)[0]
        prob = model.predict_proba(transaction_scaled)[0, 1]
        st.write(f"**Prediction**: {'Fraudulent' if pred == 1 else 'Legitimate'}")
        st.write(f"**Fraud Probability**: {prob:.4f}")
