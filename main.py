import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
data = pd.read_csv('cybersecurity.csv')
print(data)
print(data.head())
print(data.tail())
print(data.info())
print(data.describe())

data=data.drop_duplicates()

print(data.isnull().sum())
print("Before filling:",data['url'].isnull().sum())
data['url']=data['url'].fillna('unknown')
print("After filling:",data['url'].isnull().sum())

target_cols = data.select_dtypes(include=['object', 'bool']).columns

attack_encoder = None

for col in target_cols:
    if data[col].dtype == 'bool':
        data[col] = data[col].astype(int)

    else:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

        # Save only the attack_type encoder
        if col == "attack_type":
            attack_encoder = le

# Save the encoder
with open("attack_encoder.pkl", "wb") as file:
    pickle.dump(attack_encoder, file)

X = data.drop(columns=['attack_type']) 
y = data['attack_type']

# =========================================================
# STEP 5: TRAIN-TEST SPLIT (80% Train, 20% Test)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# STEP 6: PRINT SPLITS IN CLASSIC (X, Y) COORDINATE FORMAT
# =========================================================
print("\n=== TOTAL DATA SAMPLES ===")
print(f"Total Features (x, y): ({data.shape[0]}, {data.shape[1]})") 

print("\n=== TRAINING DATA SAMPLES ===")
print(f"Train Features (x, y): ({X_train.shape[0]}, {X_train.shape[1]})")

print("\n=== TESTING DATA SAMPLES ===")
print(f"Test Features  (x, y): ({X_test.shape[0]}, {X_test.shape[1]})")

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds) * 100:.2f}%")

# 2. Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, svm_preds) * 100:.2f}%")

# 3. K-Nearest Neighbors (KNN)
n_neighbors = 3 if len(X_train) >= 3 else 1
knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)
print(f"KNN Accuracy: {accuracy_score(y_test, knn_preds) * 100:.2f}%")
import pickle
with open("random_forest_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

with open("random_forest_model.pkl", "rb") as file:
    loaded_rf_model = pickle.load(file)
test_preds = loaded_rf_model.predict(X_test)
print("Model loaded successfully and ready for use!")
