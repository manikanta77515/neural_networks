
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
data = pd.read_csv("students.csv")

print("First 5 rows:")
print(data.head())

# Step 2: Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Step 3: Remove rows where the correct result is missing
data = data.dropna(subset=["result"]).copy()

# Step 4: Separate input features (X)
X = data[
    [
        "study_hours",
        "attendance",
        "previous_marks",
        "assignments_completed"
    ]
]

# Step 5: Separate the target (y)
y = data["result"]

# Step 6: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Step 7: Fill missing input values using training medians
medians = X_train.median()

X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

# Step 8: Display the results
print("\nTraining input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

print("\nTraining inputs:")
print(X_train)

print("\nTraining targets:")
print(y_train)



