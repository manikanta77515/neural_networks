
import torch
import torch.nn as nn

# Actual results of two students
actual_results = torch.tensor([
    [1.0],
    [0.0]
])

# Raw predictions (logits) from the neural network
predictions = torch.tensor([
    [2.0],
    [-1.0]
])

# Create the loss function
loss_function = nn.BCEWithLogitsLoss()

# Calculate loss
loss = loss_function(predictions, actual_results)

print("Actual results:")
print(actual_results)

print("Predictions (logits):")
print(predictions)

print("Loss:")
print(loss.item())
