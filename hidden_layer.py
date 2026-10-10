
import torch
import torch.nn as nn

# 1. Create a hidden layer
hidden_layer = nn.Linear(4, 3)

# 2. Create the ReLU activation function
relu = nn.ReLU()

# 3. Give input data for one student
student_data = torch.tensor([
    [6.0, 85.0, 70.0, 8.0]
])

# 4. Pass inputs through the hidden layer
hidden_output = hidden_layer(student_data)

print("Output before ReLU:")
print(hidden_output)

# 5. Apply ReLU
activated_output = relu(hidden_output)

print("Output after ReLU:")
print(activated_output)

