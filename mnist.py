# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

# # Define the CNN model
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
#         self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
#         self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(7*7*64, 1024)
#         self.fc2 = nn.Linear(1024, 10) # MNIST has 10 classes

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.max_pool1(x)
#         x = F.relu(self.conv2(x))
#         x = self.max_pool2(x)
#         x = x.view(-1, 7*7*64) # Flatten the output for the fully connected layer
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# def test(model, device, test_loader):
#     model.eval()  # Set the model to evaluation mode
#     correct = 0
#     total = 0
#     with torch.no_grad():  # Inference mode, gradients not needed
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             _, predicted = torch.max(output.data, 1)
#             total += target.size(0)
#             correct += (predicted == target).sum().item()

#     print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')

# model = CNN()
# model.load_state_dict(torch.load('./mnist/mnist_cnn_adv.pt'))


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10) # MNIST has 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(-1, 7*7*64) # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function for the CNN
def train(model, device, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (data, target) in enumerate(train_loader, 0):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

# Test function for the CNN
def test(model, device, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Inference mode, gradients not needed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')

# Check if model exists, else train and save
def load_or_train_model(path, train_loader, device):
    model = CNN().to(device)
    if os.path.exists(path):
        print("Loading the saved model...")
        model.load_state_dict(torch.load(path))
    else:
        print("Training the model as no saved model found...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        train(model, device, train_loader, optimizer, criterion)
        torch.save(model.state_dict(), path)
        print("Model saved to path:", path)

    return model



