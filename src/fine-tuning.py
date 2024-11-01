
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
from torch.utils.data import Subset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import argparse
from wrapper import *

# Set up the device for GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.NLLLoss()

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, 1)   # Reduced number of filters
        self.conv2 = nn.Conv2d(1, 1, 3, 1)   # Reduced number of filters
        self.fc1 = nn.Linear(25, 10)        # Smaller fully connected layer
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return torch.log_softmax(x, dim=1)
    
# Filter function to only keep labels 0 to 7
def filter_mnist(dataset, allowed_labels):
    mask = torch.tensor([label in allowed_labels for label in dataset.targets])
    new_dataset = copy.copy(dataset)
    new_dataset.data = dataset.data[mask]
    new_dataset.targets = dataset.targets[mask]
    return new_dataset

def truncate_to_matrix(array_list):
    # Find the minimum length among the arrays
    min_length = min(arr.shape[0] for arr in array_list)
    
    # Truncate each array to the minimum length and stack them into a matrix
    truncated_matrix = np.vstack([arr[:min_length] for arr in array_list])
    
    return truncated_matrix

# Filter function to only keep labels 0 to 7
def filter_mnist(dataset, allowed_labels):
    mask = torch.tensor([label in allowed_labels for label in dataset.targets])
    new_dataset = copy.copy(dataset)
    new_dataset.data = dataset.data[mask]
    new_dataset.targets = dataset.targets[mask]
    return new_dataset

# Load model
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("MNIST/mnist_cnn.pt", weights_only=False))
    model.to(device)
    return model

def evaluate(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return test_loss

# Assume test_dataset is already loaded as described in previous code
def generate_subset(dataset, percentage=1e-3):
    # Get the total number of samples in the test dataset
    total_samples = len(dataset)
    print(f"Total number of samples in the train dataset: {total_samples}")

    # Calculate the number of samples corresponding to 1%
    num_samples_1_percent = int(percentage * total_samples)

    # Randomly select 1% of the indices from the test dataset
    random_indices = random.sample(range(total_samples), num_samples_1_percent)

    # Create a subset of the test dataset using these random indices
    subset = Subset(dataset, random_indices)
    print(f"Number of samples in the subset: {len(subset)}")

    # Create a new DataLoader for the subsampled data
    subsampled_loader = DataLoader(subset, batch_size=1000, shuffle=False)

    return subsampled_loader

# Assume test_dataset is already loaded as described in previous code
def generate_subset_test(test_dataset, percentage=1e-3):
    # Get the total number of samples in the test dataset
    total_samples = len(test_dataset)
    print(f"Total number of samples in the train dataset: {total_samples}")

    # Calculate the number of samples corresponding to 1%
    num_samples_1_percent = int(percentage * total_samples)

    # Randomly select 1% of the indices from the test dataset
    random_indices = random.sample(range(total_samples), num_samples_1_percent)

    # Create a subset of the test dataset using these random indices
    test_subset = Subset(test_dataset, random_indices)
    print(f"Number of samples in the subset: {len(test_subset)}")

    # Create a new DataLoader for the subsampled data
    subsampled_test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    return subsampled_test_loader

def parse_parameters():
    parser = argparse.ArgumentParser(description='SSD Optimization Parameters')
    parser.add_argument('--d', type=int, default=280, help='Problem Dimension')
    parser.add_argument('--ell', type=int, default=20, help='Subspace Dimension')
    parser.add_argument('--sr', type=float, default=1e-2, help='Subsample Ratio')
    parser.add_argument('--epochs', type=int, default=10, help='Number of Epochs')
    parser.add_argument('--line_iter', type=int, default=20, 
                        help='Maximal Number of Line Search Iterations')
    parser.add_argument('--L0', type=float, default=1.0, 
                        help='Initial Learning Rate for Line Search')
    parser.add_argument('--c', type=float, default=0.99, help='Armijo Shrinking Factor')
    parser.add_argument('--num_trials', type=int, default=3, help='Number of Trials')
    return parser.parse_args()

def main():
    # Parse the parameters
    args = parse_parameters()
    d = args.d
    ell = args.ell
    num_iterations = args.epochs
    linesearch_iter = args.line_iter
    L0 = args.L0
    c = args.c
    num_trials = args.num_trials
    subsample_ratio = args.sr

    # Load the model
    model = CNN()
    model.load_state_dict(torch.load("MNIST/mnist_cnn.pt", weights_only=False))
    model.to(device)

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./MNIST', train=True, download=False, transform=transform)

    # Test model on non-trained labels
    nontrain_labels = list(range(8, 10))
    nontrain_train_dataset = filter_mnist(train_dataset, nontrain_labels)
    nontrain_train_loader = DataLoader(nontrain_train_dataset, batch_size=64, shuffle=True)

    # nontrain_train_loader = DataLoader(nontrain_train_dataset, batch_size=batch_size, shuffle=False)
    subsampled_train_loader = generate_subset_test(nontrain_train_dataset, percentage=subsample_ratio)

    def f_HF(x):
        # Convert the NumPy array to a PyTorch tensor
        new_params = torch.from_numpy(x).float()  # Ensure the type matches

        # After modifying the flattened tensor, update the original parameters
        start = 0
        for param in model.parameters():
            num_params = param.numel()
            param.data = new_params[start:start + num_params].view(param.size())
            start += num_params

        # Evaluate after modification
        test_loss = evaluate(model, nontrain_train_loader, device)

        return test_loss

    def f_LF(x):
        # Convert the NumPy array to a PyTorch tensor
        new_params = torch.from_numpy(x).float()  # Ensure the type matches

        # After modifying the flattened tensor, update the original parameters
        start = 0
        for param in model.parameters():
            num_params = param.numel()
            param.data = new_params[start:start + num_params].view(param.size())
            start += num_params

        # Evaluate after modification
        test_loss = evaluate(model, subsampled_train_loader, device)

        return test_loss

    # Assign function classes
    # High-fidelity objective function
    obj = objectiveFcn(f_HF,label='finetune')
    # Low-fidelity objective function
    obj_lowFi= objectiveFcn(f_LF,label='finetune-LF')
    # Alternative High-fidelity objective function (for oracle SSD line search)
    obj_alt = objectiveFcn(f_HF,label='finetune-alt')

    x0 = 1e-1 * np.random.randn(d)
    learning_rate = 1e-2
    learning_rate_ssd = learning_rate * ell / d

    # Run methods
    methods = ['gd', 'cd', 'ssd', 'spsa', 'rgfm', 'ssd_lf', 'ssd_hf',
            'ssd_bf', 'ssd_oracle', 'ssd_sag']

    res = {m: [] for m in methods}
    for i in tqdm(range(num_trials)):
        # Gradient Descent
        _ = grad_desc(x0,obj,learning_rate=learning_rate,num_iterations=num_iterations)
        res['gd'].append(obj.returnHistory())
        # Coordinate Descent
        _ = coor_desc(x0,obj,learning_rate=learning_rate,num_iterations=num_iterations/2)
        res['cd'].append(obj.returnHistory())
        # SSD
        _ = ssd(x0,obj,ell=ell,learning_rate=learning_rate_ssd,num_iterations=num_iterations*d/ell)
        res['ssd'].append(obj.returnHistory())
        # SPSA
        _ = spsa(x0,obj,num_iterations=num_iterations*d)
        res['spsa'].append(obj.returnHistory())
        # Random Gredien-free Minimization
        _ = ssd(x0,obj,ell=1,learning_rate=learning_rate_ssd,num_iterations=num_iterations*d)
        res['rgfm'].append(obj.returnHistory())
        # SSD with linesearch (LF)
        _ = ssd_ls(x0,obj,ell=ell,learning_rate=learning_rate_ssd, obj_lowFi= obj_lowFi,
                    num_iterations=num_iterations*d/ell, linesearch_iter=linesearch_iter )
        res['ssd_lf'].append(obj.returnHistory())
        # SSD with linesearch (HF)
        _ = ssd_ls(x0,obj,ell=ell,learning_rate=learning_rate_ssd, obj_lowFi= obj_alt,
                    num_iterations=num_iterations*d/ell, linesearch_iter=linesearch_iter )
        res['ssd_oracle'].append(obj.returnHistory())
        # SSD with backtracking linesearch (BF)
        _ = ssd_bt_temp(x0,obj,ell=ell,obj_lowFi= obj_lowFi, c=c,num_iterations=num_iterations*d/ell,
                linesearch_iter=linesearch_iter, L0=L0 )
        res['ssd_bf'].append(obj.returnHistory())
        # SSD with backtracking linesearch (HF)
        _ = ssd_hbt(x0,obj,ell=ell,c=c,num_iterations=num_iterations*d/ell, 
                    linesearch_iter=linesearch_iter, L0=L0 )
        res['ssd_hf'].append(obj.returnHistory())
        # SSD with SAG
        _ = ssd_sag(x0,obj,ell=ell,learning_rate=learning_rate_ssd,
                    num_iterations=num_iterations*d/ell)
        res['ssd_sag'].append(obj.returnHistory())

    # Collect data and compute mean/std
    for k, v in res.items():
        res[k] = truncate_to_matrix(v)
    bf_ratio = linesearch_iter * subsample_ratio / ((ell + 1) + d)
    save_path = f'results/fine-tuning/fine-tuning-d{d}-li{linesearch_iter}-sr{subsample_ratio}-L0{L0}-ell{ell}-c{c}.npz'
    print(f'Saved results to {save_path}')
    np.savez(save_path, res=res, bf_ratio=bf_ratio)
    print('Done!')
    
if __name__ == '__main__':
    main()