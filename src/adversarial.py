
import sys
import os

# Get the absolute path of the directory containing the notebook
# This assumes your notebook's current working directory IS the 'notebook' folder
notebook_dir = os.getcwd() # Or specify the absolute path if needed

# Get the absolute path of the parent directory ('your_project_root')
parent_dir = os.path.dirname(notebook_dir)
# Or use: parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))

# Add the parent directory to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Option A: Import specific functions
from util.OPT_utilities import objectiveFcn, grad_desc, coor_desc, ssd, ssd_bt_temp, ssd_hbt, ssd_sag, spsa
from CNNTraining import SimpleCNN, load_or_train_model, train_student_with_distillation
import numpy as np
from mnist import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torch.utils.data import Subset

# FIX RANDOM SEED
np.random.seed(0)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def truncate_to_matrix(array_list):
    # Find the minimum length among the arrays
    min_length = min(arr.shape[0] for arr in array_list)
    
    # Truncate each array to the minimum length and stack them into a matrix
    truncated_matrix = np.vstack([arr[:min_length] for arr in array_list])
    
    return truncated_matrix

def generate_initial_x(model, tau, x_dagger, y_dagger, num_steps=50):
    # Load the dataset
    x_dagger_new = x_dagger.unsqueeze(0)  # Add batch dimension

    # Initialize adversarial perturbation
    x_adv = torch.zeros_like(x_dagger_new, requires_grad=True)

    # Adversarial attack loop
    optimizer = torch.optim.Adam([x_adv], lr=0.1)

    for _ in tqdm(range(num_steps)):
        optimizer.zero_grad()

        # Compute logits and loss
        logits = model(x_adv + x_dagger_new)
        y_hat = torch.softmax(logits.squeeze(), dim=0)
        loss = y_hat[y_dagger] + tau * torch.norm(x_adv) ** 2
        
        # Backward pass
        loss.backward()

        # Update perturbation
        optimizer.step()

    # Final adversarial example
    x_adversarial = x_adv.detach() + x_dagger_new

    # Get model predictions on the adversarial example
    with torch.no_grad():
        logits_adv = model(x_adversarial)
        y_hat_adv = torch.softmax(logits_adv.squeeze(), dim=0)
        predicted_label = torch.argmax(y_hat_adv).item()

    # Print model predictions
    print(f"Model Prediction on Adversarial Image: {predicted_label}")
    print(f"Confidence Scores: {y_hat_adv.tolist()}")
    print(f"True Label: {y_dagger}")

    return x_adv.detach().numpy()

def parse_parameters():
    parser = argparse.ArgumentParser(description='SSD Optimization Parameters')
    parser.add_argument('--tau', type=float, default=1e-4, help='Function Parameter')
    parser.add_argument('--d', type=int, default=28*28, help='Problem Dimension')
    parser.add_argument('--ell', type=int, default=10, help='Subspace Dimension')
    parser.add_argument('--epochs', type=int, default=10, help='Number of Epochs')
    parser.add_argument('--line_iter', type=int, default=10, 
                        help='Maximal Number of Line Search Iterations')
    parser.add_argument('--L0', type=float, default=1.0, 
                        help='Initial Learning Rate for Line Search')
    parser.add_argument('--c', type=float, default=0.9, help='Armijo Shrinking Factor')
    parser.add_argument('--num_trials', type=int, default=3, help='Number of Trials')
    parser.add_argument('--dagger_idx', type=int, default=8, help='Index of the Dagger Dataset')
    return parser.parse_args()

def main():
    # Parse the parameters
    args = parse_parameters()
    tau = args.tau
    d = args.d
    ell = args.ell
    num_iterations = args.epochs
    linesearch_iter = args.line_iter
    L0 = args.L0
    c = args.c
    num_trials = args.num_trials
    dagger_idx = args.dagger_idx
    tau = args.tau

    # Main workflow
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_datasets = datasets.MNIST('./mnist', train=False, transform=transform)
    x_dagger, y_dagger = test_datasets[dagger_idx]

    # Load the pre-trained teacher model
    model = CNN()
    model.load_state_dict(torch.load('../model/mnist_cnn_adv.pt'))

    # Load the model from the file
    # svm_clf = load('MNIST/svm_clf_mnist.joblib')
    lf_model = SimpleCNN()
    lf_model.load_state_dict(torch.load('../model/simple_cnn.pt'))

    def unnormalize(x):
        return (x * 0.3081 + 0.1307) * 255.0

    # Define the adversarial attack HF function
    def f_HF(x):
        """The inputs and outputs are numpy arrays."""
        
        # Reshape x to match the input dimensions
        x = x.reshape(1, 1, 28, 28)
        
        # Clamp x to ensure valid pixel values in the range [0, 1]
        x = np.clip(x, 0.0, 1.0)

        # Convert x to a PyTorch tensor
        x = torch.tensor(x).type(torch.FloatTensor)

        # Perform the attack and compute outputs
        with torch.no_grad():
            y_hat = torch.softmax(model(x + x_dagger.unsqueeze(0)).squeeze(), dim=0)

        # Return the desired result
        return y_hat[y_dagger].item() + tau * np.linalg.norm(x.cpu().numpy()) ** 2

    def f_LF(x):
        """The inputs and outputs are numpy arrays."""
        
        # Reshape x to match the input dimensions
        x = x.reshape(1, 1, 28, 28)
        
        # Clamp x to ensure valid pixel values in the range [0, 1]
        x = np.clip(x, 0.0, 1.0)

        # Convert x to a PyTorch tensor
        x = torch.tensor(x).type(torch.FloatTensor)

        # Perform the attack and compute outputs
        with torch.no_grad():
            y_hat = lf_model(x + x_dagger.unsqueeze(0)).squeeze()

        # Return the desired result
        return y_hat[y_dagger].item() + tau * np.linalg.norm(x.cpu().numpy()) ** 2

    obj = objectiveFcn(f_HF,label='kernel')
    obj_lowFi= objectiveFcn(f_LF,label='kernel-LF')
    learning_rate = 1e-2
    learning_rate_ssd = learning_rate * ell / d
    # initial point
    x0 = np.zeros(d)

    # Run methods
    methods = ['gd', 'cd', 'ssd', 'spsa', 'rgfm', 'ssd_hf',
            'ssd_bf', 'ssd_sag']

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
    bf_ratio = 0.0
    save_path = f'../results/adversarial/adversarial-d{d}-L0{L0}-tau{tau}-ell{ell}-c{c}-idx{dagger_idx}.npz'
    print(f'Saved results to {save_path}')
    np.savez(save_path, res=res, bf_ratio=bf_ratio)
    print('Done!')
    
if __name__ == '__main__':
    if not os.path.exists('../results/adversarial'):
        os.makedirs('../results/adversarial')
    main()