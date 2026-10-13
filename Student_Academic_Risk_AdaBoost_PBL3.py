# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
plt.style.use("seaborn-v0_8-whitegrid")
SEED = 250093107004 % (2**32 - 1)
rng = np.random.default_rng(SEED)
print("Libraries imported successfully!")

# %% [markdown]
# # 2. Create a Unique Dataset

# %%
n = 300
attendance = np.clip(rng.normal(77, 13, n), 40, 100)
study_hours = np.clip(rng.normal(12, 4.5, n), 2, 28)
sleep_hours = np.clip(rng.normal(6.8, 1, n), 4, 9.5)
previous_cgpa = np.clip(rng.normal(6.8, 0.9, n), 4, 9.8)
commute = np.clip(rng.normal(38, 21, n), 3, 120)
club_hours = np.clip(rng.normal(4.5, 3.2, n), 0, 16)
internal = np.clip(7 + .16*attendance + .46*study_hours + rng.normal(0,3.2,n), 0, 30)
assignment = np.clip(32 + .46*attendance + 1.4*study_hours + rng.normal(0,9,n), 25, 100)
lab = np.clip(35 + .62*attendance + rng.normal(0,12,n), 20, 100)

# %%
risk_score = (.07*(70-attendance) + .25*(10-study_hours) + .30*(18-internal)
              + .045*(65-assignment) + .90*(6.3-previous_cgpa)
              + .02*(65-lab) + .018*np.maximum(commute-55,0)
              + .22*np.maximum(club_hours-8,0)
              + 1.25*((attendance<65) & (internal<17))
              + 1.00*((study_hours<7) & (assignment<60))
              + rng.normal(0, 1.25, n))
risk_status = np.where(risk_score > .25, "At Risk", "On Track")

# %%
df = pd.DataFrame({
 "Student_ID":[f"ST{i:03d}" for i in range(1,n+1)],
 "Attendance_Percent":attendance.round(1),
 "Study_Hours_Per_Week":study_hours.round(1),
 "Internal_Score":internal.round(1),
 "Assignment_Score":assignment.round(1),
 "Sleep_Hours":sleep_hours.round(1),
 "Previous_CGPA":previous_cgpa.round(2),
 "Lab_Completion_Percent":lab.round(1),
 "Commute_Minutes":commute.round(1),
 "Club_Hours_Per_Week":club_hours.round(1),
 "Risk_Status":risk_status
})
df.to_csv("Data.csv", index=False)
print("Dataset saved. Shape:", df.shape)

# %%
df = pd.read_csv("Data.csv")
print("Dataset loaded successfully!")

# %% [markdown]
# # 3. Dataset Understanding

# %%
df.head()

# %%
df.tail()

# %%
df.sample(5, random_state=42)

# %%
df.info()

# %%
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# %%
print("Column names:")
for col in df.columns:
    print(col)

# %%
print("Missing values:")
print(df.isnull().sum())

# %%
print('Duplicate rows:', df.duplicated().sum())

# %%
print(df['Risk_Status'].value_counts())

# %%
print((df['Risk_Status'].value_counts(normalize=True)*100).round(2))

# %%
df.describe().T

# %%
summary = pd.DataFrame({"Data type":df.dtypes, "Unique":df.nunique(), "Missing":df.isnull().sum()})
summary

# %% [markdown]
# # 4. Data Cleaning

# %%
df.columns = df.columns.str.strip()

# %%
df['Risk_Status'] = df['Risk_Status'].str.strip()

# %%
df = df.drop_duplicates().dropna().reset_index(drop=True)
print("Cleaning completed.")

# %%
print("Shape:", df.shape)
print("Missing:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())

# %% [markdown]
# # 5. Exploratory Data Analysis

# %%
sns.countplot(data=df, x="Risk_Status", hue="Risk_Status", legend=False, palette="Set2")
plt.title("Distribution of Student Risk Status")
plt.show()

# %%
sns.histplot(data=df, x="Attendance_Percent", bins=20, kde=True, color="steelblue")
plt.title("Attendance Distribution")
plt.show()

# %%
sns.histplot(data=df, x="Study_Hours_Per_Week", bins=20, kde=True, color="mediumpurple")
plt.title("Study Hours Distribution")
plt.show()

# %%
sns.boxplot(data=df, x="Risk_Status", y="Attendance_Percent", hue="Risk_Status", legend=False, palette="Set2")
plt.title("Attendance by Risk Status")
plt.show()

# %%
sns.boxplot(data=df, x="Risk_Status", y="Internal_Score", hue="Risk_Status", legend=False, palette="Set2")
plt.title("Internal Score by Risk Status")
plt.show()

# %%
sns.boxplot(data=df, x="Risk_Status", y="Assignment_Score", hue="Risk_Status", legend=False, palette="Set2")
plt.title("Assignment Score by Risk Status")
plt.show()

# %%
sns.scatterplot(data=df, x="Study_Hours_Per_Week", y="Assignment_Score", hue="Risk_Status", palette="Set2")
plt.title("Study Hours vs Assignment Score")
plt.show()

# %%
plt.figure(figsize=(11,8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# %% [markdown]
# # 6. Statistical Analysis

# %%
features = ["Attendance_Percent","Study_Hours_Per_Week","Internal_Score","Assignment_Score","Previous_CGPA","Lab_Completion_Percent"]
print(df.groupby("Risk_Status")[features].mean().round(2))

# %%
print(df.groupby('Risk_Status')[features].median().round(2))

# %%
print(df.groupby('Risk_Status')[features].std().round(2))

# %%
print("Average attendance, At Risk:", round(df[df.Risk_Status=="At Risk"].Attendance_Percent.mean(),2))
print("Average attendance, On Track:", round(df[df.Risk_Status=="On Track"].Attendance_Percent.mean(),2))

# %% [markdown]
# # 7. Visualisation and Feature Association

# %%
risk_percentage = df["Risk_Status"].value_counts(normalize=True)*100
risk_percentage.plot(kind="bar", color=["#66c2a5","#fc8d62"])
plt.title("Risk Status Percentage")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.show()

# %%
academic_means = df.groupby("Risk_Status")[["Internal_Score","Assignment_Score","Lab_Completion_Percent"]].mean()
academic_means.plot(kind="bar", figsize=(9,5))
plt.title("Average Academic Measures")
plt.xticks(rotation=0)
plt.show()

# %%
X_analysis = df.drop(columns=["Student_ID","Risk_Status"])
y_analysis = (df["Risk_Status"]=="At Risk").astype(int)
mi = mutual_info_classif(X_analysis, y_analysis, random_state=42)
association = pd.Series(mi, index=X_analysis.columns).sort_values(ascending=False)
print(association.round(4))

# %%
association.head(8).sort_values().plot(kind="barh", color="teal")
plt.title("Top Features by Mutual Information")
plt.show()

# %% [markdown]
# # 8. Data Preprocessing and Feature Selection

# %%
X = df.drop(columns=["Student_ID","Risk_Status"])
y = (df["Risk_Status"]=="At Risk").astype(int)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("0 = On Track, 1 = At Risk")

# %%
print(X.dtypes)

# %%
selected_features = ["Attendance_Percent","Study_Hours_Per_Week","Internal_Score",
                     "Assignment_Score","Sleep_Hours","Previous_CGPA",
                     "Lab_Completion_Percent","Commute_Minutes","Club_Hours_Per_Week"]
X_selected = X[selected_features]
print("Selected features:", list(X_selected.columns))

# %%
print("Original features:", X.shape[1])
print("Selected features:", X_selected.shape[1])

# %% [markdown]
# # 9. Train-Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=.20, random_state=42, stratify=y)
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# %%
print("Training target distribution:")
print(y_train.value_counts())
print("Testing target distribution:")
print(y_test.value_counts())

# %% [markdown]
# # 10. AdaBoost Model (Beyond Syllabus Topic)

# %%
weak_learner = DecisionTreeClassifier(max_depth=1, random_state=42)
ada_model = AdaBoostClassifier(
    estimator=weak_learner, n_estimators=50, learning_rate=.8, random_state=42)
print("AdaBoost model created successfully!")

# %%
ada_model.fit(X_train, y_train)
print("AdaBoost model trained successfully!")

# %%
y_pred = ada_model.predict(X_test)
print("Predictions generated:", len(y_pred))

# %% [markdown]
# # 11. Model Evaluation

# %%
accuracy = accuracy_score(y_test, y_pred)
print(f"AdaBoost accuracy: {accuracy*100:.2f}%")

# %%
print(classification_report(y_test, y_pred,
      target_names=["On Track","At Risk"], zero_division=0))

# %%
cm = confusion_matrix(y_test, y_pred)
print(cm)

# %%
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["On Track","At Risk"], yticklabels=["On Track","At Risk"])
plt.title("AdaBoost Confusion Matrix")
plt.xlabel("Predicted Status")
plt.ylabel("Actual Status")
plt.show()

# %% [markdown]
# # 12. Feature Importance

# %%
importance = pd.Series(ada_model.feature_importances_, index=X_selected.columns).sort_values(ascending=False)
print(importance.round(4))

# %%
importance.sort_values().plot(kind="barh", figsize=(9,6), color="slateblue")
plt.title("AdaBoost Feature Importance")
plt.xlabel("Importance")
plt.show()

# %%
print('Top 5 important features:')
print(importance.head(5).round(4))

# %% [markdown]
# # 13. Predict One Student

# %%
sample_student = X_test.iloc[[0]]
sample_prediction = ada_model.predict(sample_student)[0]
status = "At Risk" if sample_prediction == 1 else "On Track"
print("Student data:")
print(sample_student.to_string(index=False))
print("Predicted status:", status)

# %%
probability = ada_model.predict_proba(sample_student)[0]
print(f"On Track probability: {probability[0]:.2%}")
print(f"At Risk probability: {probability[1]:.2%}")


actual = "At Risk" if y_test.iloc[0] == 1 else "On Track"
print("Actual status:", actual)
print("Predicted status:", status)
