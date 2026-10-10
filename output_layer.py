
import torch
import torch.nn as nn

# Example output from the hidden layer
hidden_output = torch.tensor([
    [0.5, 1.2, 0.8]
])

# Create the output layer
output_layer = nn.Linear(3, 1)

# Generate the prediction score
output = output_layer(hidden_output)

print("Prediction score:")
print(output)

