"""
author: NUOJIN
"""
import numpy as np
import emukit.multi_fidelity
import emukit.test_functions
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from GPy.kern.src.stationary import Matern32

def bf_linear_gp_regression(X_low: np.ndarray, 
                         Y_low: np.ndarray, 
                         X_high: np.ndarray, 
                         Y_high: np.ndarray, 
                         n_fidelity = 2) -> GPyMultiOutputWrapper:
    """
    Multi-fidelity linear Gaussian process regression
    following Kennedy and O'Hagan (2000)
    X_low: low-fidelity input data
    Y_low: low-fidelity output data
    X_high: high-fidelity input data
    Y_high: high-fidelity output data
    X_test: test input data
    Y_test: test output data
    n_fidelity: number of fidelity levels, default as 2
    """
    # Create a linear multi-fidelity model that combines the two
    kern_dim = X_low.shape[1]
    kern_low = [Matern32(kern_dim), Matern32(kern_dim)]
    kern_linear_mf = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kern_low)
    X_train, Y_train = convert_xy_lists_to_arrays([X_low, X_high], [Y_low, Y_high])
    gpy_linear_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, kernel=kern_linear_mf, n_fidelities=n_fidelity)
    gpy_linear_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.fix(0)
    # Wrap the model using the given conversion function
    linear_mf_model = GPyMultiOutputWrapper(gpy_linear_mf_model, n_fidelity, n_optimization_restarts=5)
    linear_mf_model.optimize()
    return linear_mf_model

def bf_linear_gp_prediction(linear_mf_model: GPyMultiOutputWrapper, 
                            X_test: np.ndarray,
                            return_var: bool = False
                            ) -> np.ndarray:
    """
    Predict using the linear multi-fidelity model
    linear_mf_model: linear multi-fidelity model
    X_test: test input data
    """
    X = convert_x_list_to_array([X_test, X_test])
    X_LF = X[:len(X_test)]
    X_HF = X[len(X_test):]
    Y_LF_mean, Y_LF_var = linear_mf_model.predict(X_LF)
    Y_HF_mean, Y_HF_var = linear_mf_model.predict(X_HF)
    if return_var:
        return Y_LF_mean, Y_LF_var, Y_HF_mean, Y_HF_var
    return Y_LF_mean, Y_HF_mean