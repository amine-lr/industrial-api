import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from  sklearn.ensemble import RandomForestClassifier
import joblib, json

df =pd.read_csv("data/raw.csv")
df["Type"] = df["Type"].map({"L":0, "M":1, "H":2})
FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

X = df[FEATURES]
y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"train {X_train.shape} test {X_test.shape}")
print(y_test.value_counts(normalize=True))

clf = RandomForestClassifier(n_estimators = 100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(classification_report(y_test, pred))

joblib.dump(clf, "model.joblib")

# save metrics for CV/README
report = classification_report(y_test, pred, output_dict=True)
with open("metrics.json", "w") as f:
    json.dump({"precision_1": report["1"]["precision"],
               "recall_1": report["1"]["recall"],
               "f1_1": report["1"]["f1-score"]}, f, indent=2)
print("saved model.joblib + metrics.json")