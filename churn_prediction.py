import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

# 1. Load Dataset

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)


# 2. Data Cleaning

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# 3. EDA

plt.figure(figsize=(5,4))
df["Churn"].value_counts().plot(kind="bar")
plt.title("Churn Distribution")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
df["Contract"].value_counts().plot(kind="bar")
plt.title("Contract Type Distribution")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
plt.hist(df["MonthlyCharges"], bins=30)
plt.title("Monthly Charges Distribution")
plt.tight_layout()
plt.show()


# 4. Data Preprocessing


categorical_cols = df.select_dtypes(include=['object', 'string']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# 5. Features / Target

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Model Comparison

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
}

results = []

best_accuracy = 0
best_model = None
best_model_name = ""

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cv_score = cross_val_score(
        model,
        X,
        y,
        cv=5
    ).mean()

    results.append(
        [name, acc, prec, rec, f1, cv_score]
    )

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print(classification_report(y_test, y_pred))

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "CV Score"
    ]
)

print("\nMODEL COMPARISON")
print(
    results_df.sort_values(
        by="Accuracy",
        ascending=False
    )
)

# 7. Best Model Selection

print("\nBEST MODEL")
print(f"Model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")

print(f"\nUsing {best_model_name} for feature analysis and prediction.")

# 8. Confusion Matrix

ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test,
    y_test
)

plt.title(f"{best_model_name} Confusion Matrix")
plt.show()


# 9. Feature Importance

feature_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

feature_model.fit(X_train, y_train)

importance = pd.Series(
    feature_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print("\nTop Features")

for feature, score in importance.head(10).items():
    print(f"{feature}: {score:.4f}")

plt.figure(figsize=(8,5))
importance.head(10).plot(kind="bar")
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()

# 10. Business Insights

print("\nTOP FEATURES AFFECTING CHURN")

for feature, score in importance.head(5).items():
    print(f"{feature}: {score:.4f}")

# 11. Sample Prediction

sample_customer = pd.DataFrame({
    "gender":[1],
    "SeniorCitizen":[0],
    "Partner":[1],
    "Dependents":[0],
    "tenure":[2],
    "PhoneService":[1],
    "MultipleLines":[0],
    "InternetService":[1],
    "OnlineSecurity":[0],
    "OnlineBackup":[0],
    "DeviceProtection":[0],
    "TechSupport":[0],
    "StreamingTV":[0],
    "StreamingMovies":[0],
    "Contract":[0],
    "PaperlessBilling":[1],
    "PaymentMethod":[2],
    "MonthlyCharges":[90],
    "TotalCharges":[180]
})

prob = best_model.predict_proba(sample_customer)
prediction = best_model.predict(sample_customer)

print("\nCUSTOMER PREDICTION")

if prediction[0] == 1:
    print("Prediction: Customer Likely to Churn")
else:
    print("Prediction: Customer Likely to Stay")

print(f"Stay Probability: {prob[0][0]:.3f}")
print(f"Churn Probability: {prob[0][1]:.3f}")
