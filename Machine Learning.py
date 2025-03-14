import streamlit as st
import pandas as pd

st.title("Machine Learning")

st.write("### DATASET")
st.write("The dataset covers 100 countries from every continent. For forecasting the trend of using Android or iPhone within the next 10 years (2035),(This dataset from chat gpt)")

# โหลดไฟล์ CSV (เปลี่ยนเป็น path ที่ถูกต้อง)
df = pd.read_csv("Global_Mobile_Preference_Prediction_2035__100_Countries_.csv")  

# แสดงตาราง Dataset
st.dataframe(df)

st.write("### FEATURE")
feature_info = pd.DataFrame({
    "Feature Name": [
        "Country", "GDP_per_capita", "Population_millions", "Preferred_Device",
        "GDP_per_capita_2035", "Population_millions_2035", "Predicted_Device_2035"
    ],
    "ประเภทข้อมูล": [
        "Categorical (ข้อความ)", "Numerical (ตัวเลข)", "Numerical (ตัวเลข)", "Categorical (ข้อความ)",
        "Numerical (ตัวเลข)", "Numerical (ตัวเลข)", "Categorical (ข้อความ)"
    ],
    "คำอธิบาย": [
        "ชื่อประเทศ",
        "รายได้เฉลี่ยต่อหัวของประชากรในปัจจุบัน (ดอลลาร์สหรัฐ)",
        "จำนวนประชากรของประเทศในปัจจุบัน (ล้านคน)",
        "ประเภทของสมาร์ทโฟนที่ประชากรส่วนใหญ่ใช้ (Android หรือ iPhone)",
        "รายได้เฉลี่ยต่อหัวของประชากรที่คาดการณ์ในปี 2035",
        "จำนวนประชากรที่คาดการณ์ในปี 2035",
        "ทำนายว่าประเทศนั้นจะใช้ Android หรือ iPhone มากกว่ากันในปี 2035"
    ]
})
# แสดง Feature Table โดยไม่มี Index
st.write("Feature Details Table")
st.table(feature_info.style.hide(axis="index"))  # ซ่อน index ออกจากตาราง

st.write("### Missing Values in Dataset")

# คำนวณ Missing Values ก่อนเติมค่า
missing_values_before = df.isnull().sum().reset_index()
missing_values_before.columns = ["Feature Name", "Missing Values"]
st.write("Before Filling Missing Values")
st.table(missing_values_before)



# เลือกเฉพาะ Feature ที่เป็นตัวเลขและมี Missing Values
numerical_features = ["GDP_per_capita", "Population_millions", "GDP_per_capita_2035", "Population_millions_2035"]
df_filled = df.copy()

# เติมค่าที่หายไปด้วยค่าเฉลี่ยของแต่ละคอลัมน์
for feature in numerical_features:
    if df_filled[feature].isnull().sum() > 0:
        df_filled[feature].fillna(df_filled[feature].mean(), inplace=True)

# ตรวจสอบ Missing Values หลังเติมค่า
missing_values_after = df_filled.isnull().sum().reset_index()
missing_values_after.columns = ["Feature Name", "Missing Values"]

# แสดง Dataset หลังเติมค่าที่หายไป
st.write("### Dataset After Filling Missing Values")
st.dataframe(df_filled)

# แสดงผล Missing Values หลังเติมค่า
st.write("After Filling Missing Values")
st.table(missing_values_after)

# แสดงจำนวน Missing Values ที่ถูกเติม
missing_diff = missing_values_before.set_index("Feature Name") - missing_values_after.set_index("Feature Name")
missing_diff = missing_diff.reset_index()
missing_diff.columns = ["Feature Name", "Missing Values Filled"]


st.title(":star: :blue[K-Means Clustering Algorithm]:star:")
st.write(" K-Means is one of the most popular clustering algorithms used for unsupervised learning. It automatically groups data points into K clusters based on their similarities.")

st.image("Kmean1.png", caption="This image shows an example of segmenting data using the K-Means Clustering Algorithm, divided into multiple segments based on a specified number of clusters (K), which helps to see the impact of choosing different K values.", use_container_width=True)

st.subheader("How K-Means Clustering Work?", divider="red")

st.write("#### **Initial Step:** Select number of Cluster (K) and randomly select center point of each cluster (Centroid).")
st.image("Kmean2.png", use_container_width=True)

st.write("#### **Step 1:** Calculate distance from each data to each Centroid. Then Assign each data to the closest cluster.")
st.image("Kmean3.png", caption="Iteration 1: Step 1 Calculate distance", use_container_width=True)
st.image("Kmean4.png", caption="Iteration 1: Step 1 Calculate distance (conts.)", use_container_width=True)
st.image("Kmean5.png", caption="Iteration 1: Step 1 Calculate distance (conts.)", use_container_width=True)

st.write("#### Step 2: Recalculate the centroid of each cluster.")
st.image("Kmean6.png", caption="Iteration 1: Step 2 Calculate distance", use_container_width=True)
st.write("#### Repeat Step 1 and 2 until the centroids don’t change.")
st.image("Kmean7.png", caption="Iteration 1: Result", use_container_width=True)
st.image("Kmean8.png", caption="Iteration 2: Result", use_container_width=True)
st.image("Kmean9.png", caption="Iteration 3: Result", use_container_width=True)

st.write("### Select K with Elbow method")
st.write("#### Calculate the Within-Cluster-Sum of Squared Errors (WSS) for different values of k")
st.image("Kmean10.png",  use_container_width=True)



st.write("Select Features for Clustering")
code = '''
features = ['GDP_per_capita_2035', 'Population_millions_2035']
X = df[features] 
'''
st.code(code, language="python")
st.write(" Standardize Data")
code = '''
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) '''
st.code(code, language="python")
st.write(" Determine Optimal K Using Elbow Method")
code = '''
k_value = st.sidebar.slider("Select Number of Clusters (K)", min_value=2, max_value=10, value=3, step=1) '''
st.code(code, language="python")
st.image("Kmean12.png", caption="Allows users to select the number of clusters K using a sidebar slider.", use_container_width=True) ##12

st.write(" Iterates over K values from 1 to 10, computing the distortion score using Euclidean distance.")
code = '''
distortions = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    distortions.append(sum(np.min(cdist(X_scaled, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X_scaled.shape[0])
    '''
st.code(code, language="python")

st.write(" Plots the Elbow Method graph to visualize the optimal K.")
code = '''
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, distortions, marker='o', linestyle='--', color='orange')
st.pyplot(fig)
    '''
st.code(code, language="python")
st.image("Kmean13.png",  use_container_width=True) ## 13


st.write(" Apply K-Means Clustering")
code = '''
kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
    '''
st.code(code, language="python")
st.image("Kmean14.png", caption="Runs K-Means with the user-selected K and assigns cluster labels to each data point.", use_container_width=True) ##14

st.write(" Visualize Clustering Results")
code = '''
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df['GDP_per_capita_2035'], y=df['Population_millions_2035'], hue=df['Cluster'], palette='viridis', s=100, ax=ax)
st.pyplot(fig)
    '''
st.code(code, language="python")
st.image("Kmean15.png", caption="Uses Seaborn Scatter Plot to display the clustering results with different colors for each cluster.", use_container_width=True) ##15

st.write(" Displays a Cluster Summary Table, showing the number of countries, average GDP, and average population in each cluster.")
code = '''
cluster_summary = df.groupby('Cluster').agg(
    Count=('Cluster', 'count'),
    Avg_GDP_2035=('GDP_per_capita_2035', 'mean'),
    Avg_Pop_2035=('Population_millions_2035', 'mean')
).reset_index()
st.dataframe(cluster_summary)
    '''
st.code(code, language="python")
st.image("Kmean16.png",  use_container_width=True) ## 16


##SVM

st.title(":dizzy: :blue[Support Vector Machine]:dizzy:")
st.write("The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N the number of features) that distinctly classifies the data points.")
st.image("svm1.png",  use_container_width=True)

st.write("#### Classification Margin")
st.image("svm2.png",  use_container_width=True)

st.write("#### SVM = Maximum Margin Separators")
st.image("svm3.png",  use_container_width=True)
st.write("Support vector machine model is a hyperplane(separator) in an N-dimensional space (features space) that distinctly classifies the dataset with maximum margin.")

st.write("#### Hyperplanes of SVM")
st.image("svm4.png",  use_container_width=True)

st.write("#### Problems with linear")
st.image("svm5.png",  use_container_width=True)
st.write("#### Problems with linear (again!)")
st.image("svm6.png",  use_container_width=True)
st.write("#### Feature spaces Technique")
st.write("Datasets that are linearly separable work out great! (even with some noise)")
st.image("svm7.png",  use_container_width=True)
st.write("But what are we going to do if the dataset is just too hard?")
st.image("svm8.png",  use_container_width=True)
st.write("How about ... mapping data to a higher-dimensional space:")
st.image("svm9.png",  use_container_width=True)

st.write("#### SVM with Feature spaces")
st.write("the original feature space can always be mapped to some higher-dimensional feature space where the training set is separable")
st.image("svm10.png",  use_container_width=True)
st.write("#### Non-linear SVM with Kernel trick")
st.image("svm11.png",  use_container_width=True)

st.title(":hotsprings: :red[Develop SVM Model] :recycle:")


st.write("Use K-Means Cluster Labels as Target for SVM")
code = '''
df['Cluster_Label'] = df['Cluster'] '''
st.code(code, language="python")

st.write(" Allow Users to Set SVM Parameters")
code = '''
kernel_option = st.sidebar.selectbox("Select Kernel:", ['linear', 'rbf', 'poly', 'sigmoid'])
C_value = st.sidebar.slider("Select C (Regularization)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
gamma_value = st.sidebar.slider("Select Gamma", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
 '''
st.code(code, language="python")
st.image("svm14.png", caption="Users can select Kernel Type, Regularization Parameter (C), and Gamma through the sidebar.", use_container_width=True) ##14

st.write(" Select Features and Split Data")
code = '''
selected_features = st.sidebar.multiselect("Select Features for Classification", features, default=features)
if len(selected_features) != 2:
    st.warning("⚠️ Please select exactly 2 features to visualize the decision boundary.")
    st.stop() '''
st.code(code, language="python")
st.image("svm15.png", caption="Users must select exactly two features to visualize the SVM decision boundary.", use_container_width=True) ##15


st.write(" Splits data into 80% training and 20% testing sets.")
code = '''
X = df[selected_features]
y = df['Cluster_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)'''
st.code(code, language="python")

st.write("Trains the SVM model with user-defined parameters.")
code = '''
svm_model = SVC(kernel=kernel_option, C=C_value, gamma=gamma_value, random_state=42)
svm_model.fit(X_train_scaled, y_train)'''
st.code(code, language="python")



st.write("Display SVM Results")
code = '''
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
st.subheader(f"📌 Accuracy: {accuracy:.2f}") '''
st.code(code, language="python")
st.image("svm16.png", caption="Computes SVM accuracy on the test set.", use_container_width=True) ##16


st.write(" Uses plot_decision_regions() to visualize the decision boundary")
code = '''
fig, ax = plt.subplots(figsize=(8, 5))
plot_decision_regions(X_train_scaled, y_train.to_numpy(), clf=svm_model, legend=2)
st.pyplot(fig)
 '''
st.code(code, language="python")
st.image("svm17.png",  use_container_width=True)## 17



st.markdown("<h1 style='color: yellow;'>🐒 Compare K-Means vs SVM Predictions 🍌</h1>", unsafe_allow_html=True)

st.write(" Compares SVM predictions against K-Means cluster labels.")
code = '''
df['SVM_Prediction'] = svm_model.predict(scaler.transform(X))
df['Match'] = df['SVM_Prediction'] == df['Cluster_Label']
match_count = df['Match'].sum()
mismatch_count = len(df) - match_count
 '''
st.code(code, language="python")

st.write(" Displays a bar chart showing the number of matching vs mismatching predictions.")
code = '''
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Matched", "Mismatched"], [match_count, mismatch_count], color=['green', 'red'])
st.pyplot(fig)
'''
st.code(code, language="python")
st.image("svm18.png",  use_container_width=True)

st.write(" Calculates the percentage of matching predictions between K-Means and SVM.")
code = '''
match_percentage = (match_count / len(df)) * 100
st.write(f"✅ **Percentage of Matching Predictions: {match_percentage:.2f}%**")'''
st.code(code, language="python")
st.image("svm19.png",  use_container_width=True)
