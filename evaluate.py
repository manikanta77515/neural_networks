
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample test data: study hours, attendance
X_test = torch.tensor([
    [2.0, 40.0],
    [3.0, 50.0],
    [6.0, 80.0],
    [8.0, 90.0]
])

# Actual results
y_test = torch.tensor([0, 0, 1, 1])

# Example model architecture
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

# Evaluation mode
model.eval()

# Make predictions without calculating gradients
with torch.no_grad():
    logits = model(X_test)

    # Convert logits into probabilities
    probabilities = torch.sigmoid(logits)

    # Convert probabilities into classes
    predictions = (probabilities >= 0.5).int().squeeze(1)

# Calculate metrics
y_pred = predictions.numpy()

accuracy = accuracy_score(y_test.numpy(), y_pred)
matrix = confusion_matrix(y_test.numpy(), y_pred)

print("Probabilities:")
print(probabilities.squeeze(1))

print("Predicted classes:")
print(y_pred)

print("Actual classes:")
print(y_test.numpy())

print("Accuracy:", accuracy)
print("Confusion matrix:")
print(matrix)

