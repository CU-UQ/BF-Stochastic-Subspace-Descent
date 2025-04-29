# BF-Stochastic-Subspace-Descent: Codebase

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


This codebase implements the proposed [Stochastic Subspace Descent Accelerated via Bi-fidelity Line Search] () methods and provides scripts to reproduce the optimization results presented in the paper.

## Repository Structure

The repository is organized as follows:

```
.
├── reproduce.ipynb         # Jupyter Notebook to reproduce main results/figures from the paper.
├── src/                    # Scripts for individual test cases/experiments.
│   ├── worst.py            # Code for Test Case 4.1 (e.g., Worst-case function optimization).
│   ├── kernel.py           # Code for Test Case 4.2.1 (e.g., Kernel ridge regression).
│   ├── adversarial.py      # Code for Test Case 4.2.2 (e.g., Adversarial attack example).
│   └── soft_prompting.py   # Code for Test Case 4.2.3 (e.g., BERT soft prompting example).
├── util/                   # Core implementation of optimization methods.
│   ├── DFO_utilities.py    # Utilities for implementing the  SSD methods.
│   └── OPT_utilities.py    # Utilities for competing/baseline optimization methods used in the paper.
├── configs/                # Configuration files.
│   └── configs.py          # Configuration settings, primarily for plotting results.
├── model/                  # Placeholder directory for saving models (if applicable).
├── results/                # Placeholder directory for saving raw optimization results/logs.
└── README.md               # This file.
```

## Usage and Reproducibility


### Running Individual Test Cases

The Python scripts within the `src/` directory correspond to the specific test cases discussed in the paper. You can typically run these scripts directly to generate the optimization results for each scenario:

```bash
python src/worst.py       # Run Test Case 1
python src/kernel.py      # Run Test Case 2
python src/adversarial.py # Run Test Case 3
python src/soft_prompting.py # Run Test Case 4
```

These scripts will likely perform the optimization runs using methods defined in `util/` and may save detailed logs or raw numerical results into the `results/` directory. Check the individual scripts for specific command-line arguments or configurations they might accept.

### Reproducing Paper Results

The primary way to reproduce the main figures and results from the paper is by running the Jupyter Notebook:

```bash
jupyter notebook reproduce.ipynb
```

This notebook utilizes functions from the `util/` directory to run experiments and `configs/configs.py` for generating plots, likely saving figures or summarized results.

## Code Overview

* **`util/DFO_utilities.py`**: Contains the implementation of the Stochastic Subspace Descent (SSD) algorithms.
* **`util/OPT_utilities.py`**: Implements core idea (BF-SSD) and other competing methods used for comparison in the paper.
* **`configs/configs.py`**: Centralizes configuration parameters, especially those related to plotting aesthetics (colors, line styles, figure sizes) for consistency.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{YourPaperCitationKey,
  title   = {Insert Paper Title Here},
  author  = {Author One and Author Two and ...},
  journal = {Journal or Conference Name},
  year    = {Year},
  volume  = {Volume},
  pages   = {Pages},
  % Add other relevant fields like DOI, URL etc.
}
```

## License

This project is licensed under the MIT License. 

## Contact

Please contact the author through e-mail *Nuojin.Cheng@colorado.edu* or *Nuojin.Cheng@gmail.com*.
