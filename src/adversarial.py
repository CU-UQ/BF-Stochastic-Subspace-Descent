
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from wrapper import *
from mnist import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from joblib import load
import argparse

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

def unnormalize(x):
    return (x * 0.3081 + 0.1307) * 255.0

def parse_parameters():
    parser = argparse.ArgumentParser(description='SSD Optimization Parameters')
    parser.add_argument('--tau', type=float, default=10.0, help='Function Parameter')
    parser.add_argument('--d', type=int, default=28*28, help='Problem Dimension')
    parser.add_argument('--ell', type=int, default=10, help='Subspace Dimension')
    parser.add_argument('--epochs', type=int, default=10, help='Number of Epochs')
    parser.add_argument('--line_iter', type=int, default=10, 
                        help='Maximal Number of Line Search Iterations')
    parser.add_argument('--L0', type=float, default=1.0, 
                        help='Initial Learning Rate for Line Search')
    parser.add_argument('--c', type=float, default=0.9, help='Armijo Shrinking Factor')
    parser.add_argument('--num_trials', type=int, default=3, help='Number of Trials')
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


    # Load and transform the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std deviation for MNIST
    ])

    # Load CNN model
    model_path = './mnist/mnist_cnn_adv.pt'
    train_dataset = datasets.MNIST('./mnist', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model = load_or_train_model(model_path, train_loader, device)

    tau = 10.0

    # Learning rate 
    learning_rate = 1e-2
    learning_rate_ssd = learning_rate * ell / d
    # initial point
    x0 = 1e-4 * np.random.randn(d)

    # Load the model from the file
    lr_clf = load('./MNIST/lr_clf_mnist.joblib')

    # Select attack sample
    dagger_idx = 3
    test_datasets = datasets.MNIST('./mnist', train=False, transform=transform)
    x_dagger, y_dagger = test_datasets[dagger_idx]
    other_idx = np.setdiff1d(np.arange(10), y_dagger)

    # define the adversarial attack HF function
    def f_HF(x):
        """The inputs and outputs are np arrays"""
        x = x.reshape(1, 1, 28, 28)
        x = torch.tensor(x).type(torch.FloatTensor)
        with torch.no_grad():
            y_hat = torch.softmax(model(x + x_dagger.unsqueeze(0)).squeeze(), dim=0)
            log_y_diff = torch.log(y_hat[other_idx].max()) - torch.log(y_hat[y_dagger])
        return -tau * torch.relu(log_y_diff).item() + np.linalg.norm(x) ** 2
    
    def f_LF(x):
        """The inputs and outputs are np arrays"""
        x = x.reshape(1, 784)
        x_0 = unnormalize(x_dagger.reshape(1, 784).detach().numpy())
        y_hat = lr_clf.predict_proba(255.0 * x + x_0).squeeze()
        log_y_diff = np.log(y_hat[other_idx].max()) - np.log(y_hat[y_dagger])
        return -tau * np.max([log_y_diff, 0]) + np.linalg.norm(x) ** 2

    # Assign function classes
    # High-fidelity objective function
    obj = objectiveFcn(f_HF,label='adv')
    # Low-fidelity objective function
    obj_lowFi= objectiveFcn(f_LF,label='adv-LF')
    # Alternative High-fidelity objective function (for oracle SSD line search)
    obj_alt = objectiveFcn(f_HF,label='adv-alt')

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
        _ = ssd_ls_temp(x0,obj,ell=ell,learning_rate=learning_rate_ssd, obj_lowFi= obj_lowFi,
                    num_iterations=num_iterations*d/ell, linesearch_iter=linesearch_iter )
        res['ssd_lf'].append(obj.returnHistory())
        # SSD with linesearch (HF)
        _ = ssd_ls_temp(x0,obj,ell=ell,learning_rate=learning_rate_ssd, obj_lowFi= obj_alt,
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
    bf_ratio = 0.0
    save_path = f'results/adversarial/adversarial-d{d}-d{d}-L0{L0}-tau{tau}-ell{ell}-c{c}.npz'
    print(f'Saved results to {save_path}')
    np.savez(save_path, res=res, bf_ratio=bf_ratio)
    print('Done!')
    
if __name__ == '__main__':
    main()