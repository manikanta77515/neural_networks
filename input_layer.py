
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
data = pd.read_csv("students.csv")

# Select the input features
X = data[
    [
        "study_hours",
        "attendance",
        "previous_marks",
        "assignments_completed"
    ]
]

# Select the target
y = data["result"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Display the input data
print("Input features:")
print(X.head())
print("\nNumber of input features:", X.shape[1])
print("Training input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)

