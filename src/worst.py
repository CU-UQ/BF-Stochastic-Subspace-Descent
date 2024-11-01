
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from wrapper import *
import numpy as np
import matplotlib.pyplot as plt
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
    parser.add_argument('--lmba', type=float, default=20.0, help='Lambda value')
    parser.add_argument('--d', type=int, default=1000, help='Problem Dimension')
    parser.add_argument('--r1', type=int, default=100, help='HF Dimension')
    parser.add_argument('--r2', type=int, default=2, help='LF Dimension')
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
    lmda = args.lmba
    d = args.d
    r1 = args.r1
    r2 = args.r2
    ell = args.ell
    num_iterations = args.epochs
    linesearch_iter = args.line_iter
    L0 = args.L0
    c = args.c
    num_trials = args.num_trials
    learning_rate = 1 / lmda
    learning_rate_ssd = learning_rate * ell / d

    # Initialize the worst function
    f    = lambda x : f_lr(x, lmda, r1)
    f_LF = lambda x : f_lr(x, lmda, r2)
    x0  = np.zeros(d)

    # Assign function classes
    # High-fidelity objective function
    obj = objectiveFcn(f,label='Low-rank Function')
    # Low-fidelity objective function
    obj_lowFi= objectiveFcn(f_LF)
    # Alternative High-fidelity objective function (for oracle SSD line search)
    obj_alt = objectiveFcn(f)

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
    bf_ratio = linesearch_iter * r2 / ((ell + 1) * r1)

    save_path = f'results/worst/worst-d{d}-rH{r1}-rL{r2}-lmda{lmda}-ell{ell}-c{c}.npz'
    print(f'Saved results to {save_path}')
    np.savez(save_path, res=res, bf_ratio=bf_ratio)
    print('Done!')
    
if __name__ == '__main__':
    main()