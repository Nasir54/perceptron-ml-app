import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import streamlit as st

# Set page config
st.set_page_config(page_title="Iris Perceptron Classifier", page_icon="🌸", layout="wide")

# Add title and description
st.title("🌸 Iris Flower Classification with Perceptron")
st.markdown("""
This app demonstrates a Perceptron classifier on the famous Iris dataset.
- **Features Used:** Sepal length and Petal length
- **Model:** Perceptron (linear classifier)
""")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['flower_type'] = df['target'].apply(lambda x: iris.target_names[x])
    return df, iris

df, iris = load_data()

# -----------------------------
# Sidebar for user input
# -----------------------------
st.sidebar.header("User Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', float(df['sepal length (cm)'].min()), 
                                    float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
    petal_length = st.sidebar.slider('Petal length (cm)', float(df['petal length (cm)'].min()), 
                                    float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
    data = {'sepal length (cm)': sepal_length,
            'petal length (cm)': petal_length}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# -----------------------------
# Display dataset and user input
# -----------------------------
st.subheader('Dataset Overview')
st.write(df.head())

st.subheader('User Input parameters')
st.write(input_df)

# -----------------------------
# Train Model
# -----------------------------
X = df[['sepal length (cm)', 'petal length (cm)']].values
y = df['target'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Perceptron
p = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
p.fit(X_scaled, y)

# -----------------------------
# Make Prediction
# -----------------------------
input_scaled = scaler.transform(input_df)
prediction = p.predict(input_scaled)
prediction_scores = p.decision_function(input_scaled)

st.subheader('Prediction')
st.write(f"The predicted flower type is: **{iris.target_names[prediction][0]}**")

st.subheader('Prediction Confidence Scores')
st.write("Decision function scores for each class:", prediction_scores[0])
st.write(f"Highest confidence score: {np.max(prediction_scores[0]):.2f}")

# -----------------------------
# Model Performance
# -----------------------------
st.subheader('Model Performance')

# Accuracy
y_pred = p.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)
st.write(f"Training Accuracy: {accuracy:.2%}")

# Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 4))
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(ax=ax)
st.pyplot(fig)

# -----------------------------
# Decision Boundary Visualization
# -----------------------------
st.subheader('Decision Boundary')

x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = p.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, Z, alpha=0.3)
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolor='k', cmap=plt.cm.Set1)

# Plot user input point
input_point = scaler.transform(input_df)
ax.scatter(input_point[:, 0], input_point[:, 1], c='red', marker='X', s=200, label='User Input')

ax.set_xlabel("Sepal length (scaled)")
ax.set_ylabel("Petal length (scaled)")
ax.set_title("Perceptron Decision Boundary")
ax.legend()
st.pyplot(fig)

# -----------------------------
# Model Coefficients
# -----------------------------
st.subheader('Model Details')
st.write("Coefficients:", p.coef_)
st.write("Intercept:", p.intercept_)