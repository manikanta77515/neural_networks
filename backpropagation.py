
import torch
import torch.nn as nn

# Create a simple neural network layer
layer = nn.Linear(1, 1)

# One input value
x = torch.tensor([[2.0]])

# Actual result
y = torch.tensor([[1.0]])

# Loss function
loss_function = nn.BCEWithLogitsLoss()

# Forward pass
prediction = layer(x)

# Calculate loss
loss = loss_function(prediction, y)

print("Prediction:", prediction.item())
print("Loss:", loss.item())

# Backpropagation
loss.backward()

# Display gradients
print("Weight gradient:", layer.weight.grad)
print("Bias gradient:", layer.bias.grad)

