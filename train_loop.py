
import torch
import torch.nn as nn

# Sample student data: study hours, attendance
X = torch.tensor([
    [2.0, 40.0],
    [3.0, 50.0],
    [6.0, 80.0],
    [8.0, 90.0]
])

# Actual results: 0 = Fail, 1 = Pass
y = torch.tensor([
    [0.0],
    [0.0],
    [1.0],
    [1.0]
])

# Define the neural network
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

# Loss function and optimizer
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training loop
epochs = 100

for epoch in range(epochs):

    # Clear old gradients
    optimizer.zero_grad()

    # Forward propagation
    predictions = model(X)

    # Calculate loss
    loss = loss_function(predictions, y)

    # Backpropagation
    loss.backward()

    # Update weights and biases
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss.item():.4f}"
        )
