
import torch
import torch.nn as nn

# Four input features and three hidden neurons
hidden_layer = nn.Linear(
    in_features=4,
    out_features=3
)

# Activation function
activation = nn.ReLU()

# One student's input
student_data = torch.tensor(
    [[6.0, 85.0, 70.0, 8.0]]
)

# Process the inputs through the hidden layer
hidden_output = hidden_layer(student_data)

# Apply the activation function
activated_output = activation(hidden_output)

print("Hidden layer output:")
print(hidden_output)

print("After ReLU:")
print(activated_output)

