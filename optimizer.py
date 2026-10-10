
import torch

# Create a trainable weight
weight = torch.tensor(2.0, requires_grad=True)

# Create an optimizer
optimizer = torch.optim.SGD([weight], lr=0.1)

# Actual target
target = torch.tensor(10.0)

# Forward propagation
prediction = weight * 3

# Calculate loss
loss = (prediction - target) ** 2

print("Before update")
print("Weight:", weight.item())
print("Loss:", loss.item())

# Calculate gradients
loss.backward()

print("Gradient:", weight.grad.item())

# Update the weight
optimizer.step()

print("\nAfter update")
print("Weight:", weight.item())

# Check loss after the update
new_loss = (weight * 3 - target) ** 2
print("New loss:", new_loss.item())

