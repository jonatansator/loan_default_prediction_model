import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.io as pio
import os

# Step 1: Load data from loans.csv
def load_data(fp):
    if not os.path.exists(fp):
        raise FileNotFoundError(f"File {fp} not found.")
    df = pd.read_csv(fp)
    print("Available cols:", df.columns.tolist())
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    return X, y

# Step 2: Preprocess data (scale numerical features and handle missing values)
def preprocess_data(X, y):
    # Fill missing values with median for numerical columns
    med = X.median()
    X = X.fillna(med)
    
    # Scale numerical features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    # Ensure y has no missing values
    y = y.dropna()
    return Xs, y

# Step 3: Train classification model
def train_model(Xs, y):
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.3, random_state=42)
    neg = len(y_tr[y_tr == 0])
    pos = len(y_tr[y_tr == 1])
    wt = {0: 1.0, 1: neg / pos if pos > 0 else 1}
    mdl = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight=wt, random_state=42)
    mdl.fit(X_tr, y_tr)
    y_pr = mdl.predict(X_te)
    y_sc = mdl.predict_proba(X_te)[:, 1]
    return X_te, y_te, y_pr, y_sc

# Step 4: Evaluate model performance
def eval_model(y_te, y_pr, y_sc):
    acc = accuracy_score(y_te, y_pr)
    prec = precision_score(y_te, y_pr)
    rec = recall_score(y_te, y_pr)
    fpr, tpr, _ = roc_curve(y_te, y_sc)
    auc_val = auc(fpr, tpr)
    return acc, prec, rec, fpr, tpr, auc_val

# Step 5: Create combined plot with report
def create_plot(fpr, tpr, auc_val, X_te, y_te, y_pr, acc, prec, rec):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, line=dict(color='#FF6B6B', width=2), name=f'ROC (AUC={auc_val:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color='#4ECDC4', width=2, dash='dash'), name='Random'))
    
    # Add report as annotations
    annotations = [
        dict(text=f"Test Samples: {len(X_te)}", x=0.05, y=0.95, xref="paper", yref="paper", showarrow=False, font=dict(color='white')),
        dict(text=f"Predicted Defaults: {sum(y_pr)}", x=0.05, y=0.90, xref="paper", yref="paper", showarrow=False, font=dict(color='white')),
        dict(text=f"Actual Defaults: {sum(y_te)}", x=0.05, y=0.85, xref="paper", yref="paper", showarrow=False, font=dict(color='white')),
        dict(text=f"Accuracy: {acc:.4f}", x=0.05, y=0.80, xref="paper", yref="paper", showarrow=False, font=dict(color='white')),
        dict(text=f"Precision: {prec:.4f}", x=0.05, y=0.75, xref="paper", yref="paper", showarrow=False, font=dict(color='white')),
        dict(text=f"Recall: {rec:.4f}", x=0.05, y=0.70, xref="paper", yref="paper", showarrow=False, font=dict(color='white')),
        dict(text=f"ROC AUC: {auc_val:.4f}", x=0.05, y=0.65, xref="paper", yref="paper", showarrow=False, font=dict(color='white'))
    ]
    fig.update_layout(
        title='Loan Default Analysis',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        plot_bgcolor='rgb(40, 40, 40)',
        paper_bgcolor='rgb(40, 40, 40)',
        font_color='white',
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_range=[0, 1], yaxis_range=[0, 1.05],
        xaxis_gridcolor='rgba(255, 255, 255, 0.1)',
        xaxis_gridwidth=0.5,
        yaxis_gridcolor='rgba(255, 255, 255, 0.1)',
        yaxis_gridwidth=0.5,
        annotations=annotations
    )
    return fig

# Step 6: Main execution
if __name__ == "__main__":
    fp = "loans.csv"
    try:
        # Step 7: Load data
        X, y = load_data(fp)
        
        # Step 8: Preprocess data
        Xs, y = preprocess_data(X, y)
        
        # Step 9: Train model
        X_te, y_te, y_pr, y_sc = train_model(Xs, y)
        
        # Step 10: Evaluate results
        acc, prec, rec, fpr, tpr, auc_val = eval_model(y_te, y_pr, y_sc)
        
        # Step 11: Create and display plot
        fig = create_plot(fpr, tpr, auc_val, X_te, y_te, y_pr, acc, prec, rec)
        fig.show()
        
    except Exception as err:
        print(f"Error: {err}")