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
import os
import torch
from torch import nn
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import argparse

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained transformer model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Freeze the transformer's parameters
for param in model.parameters():
    param.requires_grad = False

# Define the path to your aclImdb dataset
dataset_path = "./aclImdb"

# Load the aclImdb dataset
def load_aclImdb(data_dir, split="train"):
    """
    Load aclImdb dataset from the given directory.
    Args:
        data_dir (str): Path to the aclImdb dataset.
        split (str): Either 'train' or 'test'.
    Returns:
        List of tuples (text, label).
    """
    dataset = []
    for label, sentiment in enumerate(["neg", "pos"]):
        sentiment_path = os.path.join(data_dir, split, sentiment)
        for filename in os.listdir(sentiment_path):
            file_path = os.path.join(sentiment_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()
                dataset.append((text, label))
    return dataset

# Define a function to generate prompts
def generate_prompt(N):
    train_data = load_aclImdb(dataset_path, split="train")
    # Sub-sample 50 items
    indices = random.sample(range(len(train_data)), N)
    texts, labels = [], []
    for i in indices:
        texts.append(train_data[i][0])
        labels.append(train_data[i][1])
    return texts, torch.tensor(labels)

def truncate_to_matrix(array_list):
    # Find the minimum length among the arrays
    min_length = min(arr.shape[0] for arr in array_list)
    
    # Truncate each array to the minimum length and stack them into a matrix
    truncated_matrix = np.vstack([arr[:min_length] for arr in array_list])
    
    return truncated_matrix

# Forward pass with soft prompt
def forward_with_prompt(inputs, low_dim_soft_prompt):
    # Project the low-dimensional soft prompt to BERT's hidden size
    projected_prompt = low_dim_soft_prompt

    # Get input embeddings
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    embeddings = model.get_input_embeddings()(input_ids)  # Shape: (batch_size, seq_len, hidden_size)

    # Concatenate soft prompt embeddings
    embeddings = torch.cat([projected_prompt.expand(embeddings.size(0), -1, -1), embeddings], dim=1)
    # Fix attention mask size
    num_prompt_tokens = projected_prompt.size(0)  # 100
    prompt_attention = torch.ones((attention_mask.size(0), num_prompt_tokens), device=attention_mask.device)
    attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)

    # Forward pass through BERT
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    return outputs.logits

def parse_parameters():
    parser = argparse.ArgumentParser(description='SSD Optimization Parameters')
    parser.add_argument('--d', type=int, default=768, help='Problem Dimension')
    parser.add_argument('--nH', type=int, default=20, help='HF Sample Size')
    parser.add_argument('--nL', type=int, default=4, help='LF Sample Size')
    parser.add_argument('--ell', type=int, default=50, help='Subspace Dimension')
    parser.add_argument('--epochs', type=int, default=5, help='Number of Epochs')
    parser.add_argument('--line_iter', type=int, default=10, 
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
    nH = args.nH
    nL = args.nL
    ell = args.ell
    num_iterations = args.epochs
    linesearch_iter = args.line_iter
    L0 = args.L0
    c = args.c
    num_trials = args.num_trials

    texts, labels = generate_prompt(nH)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=10)

    # Sub-sample 2 items
    indices = random.sample(range(len(texts)), nL)
    sub_texts = [texts[i] for i in indices]
    sub_labels = torch.tensor([labels[i] for i in indices])
    sub_inputs = tokenizer(sub_texts, return_tensors="pt", padding=True, truncation=True, max_length=10)

    # High-Fidelity (HF) loss function
    def f_HF(x):
        soft_prompt = torch.from_numpy(x).float().reshape(1, d)
        logits = forward_with_prompt(inputs, soft_prompt)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss.item()

    # Low-Fidelity (LF) loss function
    def f_LF(x):
        soft_prompt = torch.from_numpy(x).float().reshape(1, d)
        logits = forward_with_prompt(sub_inputs, soft_prompt)
        loss = torch.nn.functional.cross_entropy(logits, sub_labels)
        return loss.item()
    
    obj = objectiveFcn(f_HF,label='HF')
    obj_lowFi= objectiveFcn(f_LF,label='LF')

    x0 = np.random.randn(1, d).astype(np.float32)
    learning_rate = 1e-2
    learning_rate_ssd = learning_rate * ell / d

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
    save_path = f'../results/prompting/prompting-d{d}-L0{L0}-nH{nH}-nL{nL}-ell{ell}-c{c}.npz'
    print(f'Saved results to {save_path}')
    np.savez(save_path, res=res, bf_ratio=bf_ratio)
    print('Done!')
    
if __name__ == '__main__':
    if not os.path.exists('../results/prompting'):
        os.makedirs('../results/prompting')
    main()
