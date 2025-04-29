
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
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.datasets import fetch_california_housing
from numpy.linalg import norm
# FIX RANDOM SEED
np.random.seed(0)
from tqdm import tqdm
import argparse

def truncate_to_matrix(array_list):
    # Find the minimum length among the arrays
    min_length = min(arr.shape[0] for arr in array_list)
    
    # Truncate each array to the minimum length and stack them into a matrix
    truncated_matrix = np.vstack([arr[:min_length] for arr in array_list])
    
    return truncated_matrix

def f_lr(x, lbd, r):
    """Worst function in the world by Nesterov 2013"""
    if r > len(x):
        raise ValueError('r must be less than or equal to the length of x')
    sums = (x[0]**2 + sum((x[i] - x[i+1])**2 for i in range(0, r-1)) 
            + x[r-1]**2)/2 - x[0]
    return lbd * sums/4 + lbd * r/(8*(r+1))

def parse_parameters():
    parser = argparse.ArgumentParser(description='SSD Optimization Parameters')
    parser.add_argument('--tau', type=float, default=0.001, help='Normalization Level')
    parser.add_argument('--d', type=int, default=1000, help='Problem Dimension')
    parser.add_argument('--lr', type=int, default=10, help='Nystrom Reduced Dimension')
    parser.add_argument('--ell', type=int, default=50, help='Subspace Dimension')
    parser.add_argument('--epochs', type=int, default=100, help='Number of Epochs')
    parser.add_argument('--line_iter', type=int, default=10, 
                        help='Maximal Number of Line Search Iterations')
    parser.add_argument('--L0', type=float, default=1.0, 
                        help='Initial Learning Rate for Line Search')
    parser.add_argument('--c', type=float, default=0.95, help='Armijo Shrinking Factor')
    parser.add_argument('--num_trials', type=int, default=3, help='Number of Trials')
    return parser.parse_args()

def main():
    # Parse the parameters
    args = parse_parameters()
    tau = args.tau
    d = args.d
    lr = args.lr
    ell = args.ell
    num_iterations = args.epochs
    linesearch_iter = args.line_iter
    L0 = args.L0
    c = args.c
    num_trials = args.num_trials

    # Collect data matrix
    X_cal, y_cal = fetch_california_housing(return_X_y=True)
    X = X_cal[:d]
    y = y_cal[:d]

    # Initialize the worst function
    K = pairwise_kernels(X, metric='rbf') # kernel matrix
    L_prime = Nystroem(kernel='rbf', n_components=lr).fit_transform(X) # low-rank approximation
    K_prime = L_prime @ L_prime.T # low-rank kernel matrix
    # define functions
    def A_HF(a):
        a = a.ravel()
        return a @ K @ a - 2 * a @ y + tau * (a @ a)
    def f_LF(a):
        a = a.ravel()
        return a @ K_prime @ a - 2 * a @ y + tau * (a @ a)

    # Optimal value
    a_opt = np.linalg.solve(K + tau * np.eye(d), y)
    A_opt = A_HF(a_opt)
    f_HF = lambda a: A_HF(a) - A_opt
    x0  = np.zeros(d)

    # Learning rate
    ev, _ = np.linalg.eigh(K)
    lmda = 2 * (ev[-1] + tau)
    learning_rate = 1 / lmda
    learning_rate_ssd = learning_rate * ell / d

    # Assign function classes
    # High-fidelity objective function
    obj = objectiveFcn(f_HF,label='kernel')
    # Low-fidelity objective function
    obj_lowFi= objectiveFcn(f_LF,label='kernel-LF')
    # Alternative High-fidelity objective function (for oracle SSD line search)
    obj_alt = objectiveFcn(f_HF,label='kernel-alt')

    # Run methods
    methods = ['gd', 'cd', 'ssd', 'spsa', 'rgfm', 'ssd_hf',
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
    T = num_iterations * d
    for k, v in res.items():
        res[k] = truncate_to_matrix(v)

    bf_ratio = linesearch_iter * lr / ((ell + 1) * d)

    save_path = f'../results/kernel/kernel-d{d}-lr{lr}-L0{L0}-tau{tau}-ell{ell}-c{c}.npz'
    print(f'Saved results to {save_path}')
    np.savez(save_path, res=res, bf_ratio=bf_ratio)
    print('Done!')
    
if __name__ == '__main__':
    if not os.path.exists('../results/kernel'):
        os.makedirs('../results/kernel')
    main()