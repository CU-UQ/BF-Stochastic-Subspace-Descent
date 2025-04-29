import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

# Main workflow
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("Training the model...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        train(model, device, train_loader, optimizer, criterion)
        torch.save(model.state_dict(), path)
        print("Model saved to path:", path)

    return model

# Define the updated CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1)   # First conv layer with 2 filters
        self.fc1 = nn.Linear(2 * 13 * 13, 16) # Small fully connected layer
        self.fc2 = nn.Linear(16, 10)          # Output layer

    def forward(self, x):
        x = self.conv1(x)                   
        x = torch.relu(x)                   
        x = torch.max_pool2d(x, 2)          
        x = torch.flatten(x, 1)             
        x = self.fc1(x)                     
        x = torch.relu(x)                   
        x = self.fc2(x)                     
        return torch.log_softmax(x, dim=1)  
    
# Define the distillation loss function
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    # KL Divergence between teacher and student
    kd_loss = nn.KLDivLoss(reduction="batchmean")(
        torch.log_softmax(student_logits / temperature, dim=1),
        torch.softmax(teacher_logits / temperature, dim=1)
    )
    # Cross-Entropy Loss with ground truth
    ce_loss = nn.CrossEntropyLoss()(student_logits, labels)
    # Combine the two losses
    return alpha * kd_loss + (1 - alpha) * ce_loss

# Training loop for knowledge distillation
def train_student_with_distillation(student_model, teacher_model, train_loader, device, temperature=3, alpha=0.5, epochs=10):
    student_model.to(device)
    teacher_model.to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        student_model.train()
        train_loss = 0
        correct = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # Get predictions from the student and teacher
            student_logits = student_model(data)
            with torch.no_grad():
                teacher_logits = teacher_model(data)
            
            # Compute the distillation loss
            loss = distillation_loss(student_logits, teacher_logits, target, temperature, alpha)
            train_loss += loss.item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accuracy
            pred = student_logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}, "
              f"Accuracy: {correct / len(train_loader.dataset):.4f}")

# Evaluation function
def evaluate(model, test_loader, device):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.NLLLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

def main():

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./mnist', train=False, download=True, transform=transform)

    # Select the first 1000 samples
    subset_indices = list(range(1000))  # Indices 0 to 999
    train_dataset = Subset(train_dataset, subset_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Load the pre-trained teacher model
    teacher_model_path = '../model/mnist_cnn_adv.pt'
    teacher_model = load_or_train_model(teacher_model_path, train_loader, device)

    # Initialize the student model
    student_model = SimpleCNN()

    # Train the student model with knowledge distillation
    train_student_with_distillation(student_model, teacher_model, train_loader, device)

    # Evaluate the student model
    evaluate(student_model, test_loader, device)

    save_path = '../model/simple_cnn.pt'
    torch.save(student_model.state_dict(), save_path)
    print(f"Model state dictionary saved to {save_path}")

if __name__ == "__main__":
    main()