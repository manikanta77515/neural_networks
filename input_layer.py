import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the dataset

data = pd.read_csv("students.csv")

print("Missing values before cleaning:")
print(data.isnull().sum())

# 2. Remove rows where the target is missing
data = data.dropna(subset=["result"]).copy()

# 3. Fill missing values in input features using median

feature_columns = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignments_completed",
]

for column in feature_columns:
    data[column] = data[column].fillna(data[column].median())

# 4. Validate the target labels
valid_labels = {0, 1}

if not set(data["result"].unique()).issubset(valid_labels):
    raise ValueError("The result column must contain only 0 and 1.")

# 5. Separate inputs (X) and target (y)
X = data[feature_columns]
y = data["result"].astype("int64")

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# 7. Display the results
print("\nMissing values after cleaning:")
print(data[feature_columns + ["result"]].isnull().sum())

print("\nDataset shape:", data.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

