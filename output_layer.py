import torch
import torch.nn as nn

# Hidden layer: 4 inputs -> 3 neurons
hidden_layer = nn.Linear(4, 3)

# ReLU activation
relu = nn.ReLU()

# Output layer: 3 values -> 1 output
output_layer = nn.Linear(3, 1)

# One student's input
student_data = torch.tensor([
    [6.0, 85.0, 70.0, 8.0]
])

# Input -> Hidden layer -> ReLU
hidden_output = relu(hidden_layer(student_data))

# Hidden layer -> Output layer
prediction_score = output_layer(hidden_output)

print("Hidden layer output:")
print(hidden_output)

print("\nOutput layer score:")
print(prediction_score)

