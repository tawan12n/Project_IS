import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
from scipy.spatial.distance import cdist

# Load Dataset
file_path = "Global_Mobile_Preference_Prediction_2035_Cleaned.csv"
df = pd.read_csv(file_path)

# Set Streamlit App Title
st.title("📊 K-Means & SVM for Mobile Market Segmentation (2035)")

### 📌 **K-MEANS CLUSTERING SECTION**
st.header("🔹 K-Means Clustering")

# Select Features for Clustering
features = ['GDP_per_capita_2035', 'Population_millions_2035']
X = df[features]

# **Standardize Data**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# **User-Adjustable K Slider**
k_value = st.sidebar.slider("Select Number of Clusters (K)", min_value=2, max_value=10, value=3, step=1)

# **Compute Optimal K Using Elbow Method**
distortions = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    distortions.append(sum(np.min(cdist(X_scaled, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X_scaled.shape[0])

# 📉 **Elbow Method Plot**
st.subheader("📉 Elbow Method to Find Optimal K")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, distortions, marker='o', linestyle='--', color='orange')
ax.set_xlabel('Number of Clusters (K)')
ax.set_ylabel('Distortion (Average Distance to Centroid)')
ax.set_title('Elbow Method for Optimal K')
st.pyplot(fig)

# 📌 **Apply K-Means Clustering with User-Selected K**
kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 📊 **Cluster Visualization**
st.subheader(f"📊 K-Means Clustering with K = {k_value}")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df['GDP_per_capita_2035'], y=df['Population_millions_2035'], hue=df['Cluster'], palette='viridis', s=100, ax=ax)
ax.set_xlabel('GDP per Capita in 2035')
ax.set_ylabel('Population in Millions (2035)')
ax.set_title(f'K-Means Clustering (K = {k_value})')
st.pyplot(fig)

# 📋 **Cluster Summary Table**
st.subheader("📋 Cluster Summary")
cluster_summary = df.groupby('Cluster').agg(
    Count=('Cluster', 'count'),
    Avg_GDP_2035=('GDP_per_capita_2035', 'mean'),
    Avg_Pop_2035=('Population_millions_2035', 'mean')
).reset_index()

st.dataframe(cluster_summary)



### 📌 **SVM CLASSIFICATION SECTION**
st.header("🔹 SVM Classification for Market Segmentation")

# **Use K-Means Cluster Labels as Target for SVM**
df['Cluster_Label'] = df['Cluster']

# **Sidebar - User-Adjustable SVM Parameters**
st.sidebar.header("🔧 SVM Parameters")
kernel_option = st.sidebar.selectbox("Select Kernel:", ['linear', 'rbf', 'poly', 'sigmoid'])
C_value = st.sidebar.slider("Select C (Regularization)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
gamma_value = st.sidebar.slider("Select Gamma (Only for rbf/poly/sigmoid)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

# **Select Features for SVM**
selected_features = st.sidebar.multiselect("Select Features for Classification", features, default=features)

# **Ensure Exactly 2 Features for Decision Boundary Visualization**
if len(selected_features) != 2:
    st.warning("⚠️ Please select exactly 2 features to visualize the decision boundary.")
    st.stop()

# **Split Data into Train/Test Sets**
X = df[selected_features]
y = df['Cluster_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Standardize Data**
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# **Train SVM Model**
svm_model = SVC(kernel=kernel_option, C=C_value, gamma=gamma_value, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# **Predict and Compute Accuracy**
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.subheader(f"📌 Accuracy: {accuracy:.2f}")

# **Plot Decision Boundary**
st.subheader("📊 SVM Decision Boundary")
fig, ax = plt.subplots(figsize=(8, 5))
plot_decision_regions(X_train_scaled, y_train.to_numpy(), clf=svm_model, legend=2)
plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.title(f"SVM Decision Boundary (Kernel = {kernel_option})")
st.pyplot(fig)


### 📌 **Comparison: K-Means vs SVM Predictions**
st.header("🔍 K-Means vs SVM Classification Comparison")

# **Count Matching Predictions**
df['SVM_Prediction'] = svm_model.predict(scaler.transform(X))
df['Match'] = df['SVM_Prediction'] == df['Cluster_Label']

# Count Correct Predictions
match_count = df['Match'].sum()
mismatch_count = len(df) - match_count

# **Plot Bar Chart**
st.subheader("📊 Agreement Between K-Means and SVM")
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Matched", "Mismatched"], [match_count, mismatch_count], color=['green', 'red'])
ax.set_ylabel("Number of Samples")
ax.set_title("Comparison: K-Means vs SVM Predictions")
st.pyplot(fig)

# **Display Percentage Match**
match_percentage = (match_count / len(df)) * 100
st.write(f"✅ **Percentage of Matching Predictions: {match_percentage:.2f}%**")
