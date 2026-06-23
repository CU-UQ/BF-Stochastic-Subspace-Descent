from DFO_utilities import as_column_vec
import DFO_utilities
import numpy as np
from typing import Optional

class objectiveFcn:
    """ Note: in numpy, vectors can be of size (n,) or of size (n,1), 
    and the two conventions do not mix well. The convention this code takes
    is that they are of size (n,1) """
    def __init__(self, f, dimension=None, column_vectors = True, errFcn = None, label=None):
        """ 
        'f' is the objective function
        This class will make a 'counter' class to record how
        often it is called """
        self.f = f
        self.fcnHistory = []
        self.posHistory = []
        self.errHistory = []
        self.dim = dimension
        self.errFcn = errFcn
        if column_vectors:
            self.vec = DFO_utilities.as_column_vec
        else:
            self.vec = np.asarray
        self.label=label
    def eval(self,x):
        x = self.vec(x)
        fx = self.f(x)
        if isinstance(fx,np.ndarray):
            if fx.size > 1 :
                raise ValueError("Objective Fcn must return a scalar!")
            fx = fx.item() # make sure it is a scalar
        self.posHistory.append(x.flatten())
        self.fcnHistory.append(fx)
        if self.errFcn is not None:
            self.errHistory.append( self.errFcn(x) )
        if self.dim == None:
            self.dim = x.shape[0]
        elif self.dim != x.size:
            raise ValueError(f'x is not the right size, is {x.size}, expecting {self.dim}')
        return fx
    def directionalDerivative(self,x,p, return_fx = False, centeredDifferences=False):
        deriv, fx = DFO_utilities.directionalDerivative(x,p,self.eval,centeredDifferences=centeredDifferences)
        deriv = self.vec(deriv)
        if return_fx:
            return deriv, fx
        else:
            return deriv
    def gradient(self, x, return_fx = False):
        """ Finds gradient via finite-differences """
        deriv, fx = DFO_utilities.gradient(x, self.eval)
        deriv = self.vec(deriv)
        if return_fx:
            return deriv, fx
        else:
            return deriv
    def returnHistory(self, return_pos = False, return_err = False):
        if not return_pos and ((self.errFcn is None)  or  (not return_err)) :
            return np.asarray(self.fcnHistory)
        elif return_pos and ((self.errFcn is None)  or  (not return_err)) :
            return np.asarray(self.fcnHistory), np.asarray(self.posHistory)
        else:
            return np.asarray(self.fcnHistory), np.asarray(self.errHistory)
        
    def returnMinimumHistory(self, return_err = False):
        """ Function evaluation history, but always taking the best-so-far """
        if (self.errFcn is None)  or  (not return_err) :
            return np.minimum.accumulate(np.asarray(self.fcnHistory))
        else:
            return np.minimum.accumulate(np.asarray(self.fcnHistory)), np.minimum.accumulate(np.asarray(self.errHistory))
        
    def reset(self):
        self.fcnHistory = []
        self.errHistory = []
        self.posHistory = []
        return

# class calibrateFcn:
#     """Function class representing the calibration function after 
#     the provided HF evaluation"""
#     def __init__(self,
#                  objHF: objectiveFcn,
#                  objLF: objectiveFcn):
#         self.objHF = objHF
#         self.objLF = objLF
#         self.update()

#     def update(self):
#         hf_fcn, hf_pos = self.objHF.returnHistory(return_pos=True)
#         lf_fcn, lf_pos = self.objLF.returnHistory(return_pos=True)
#         model = bf_linear_gp_regression(lf_pos, lf_fcn.reshape(-1, 1), 
#                                         hf_pos, hf_fcn.reshape(-1, 1))
#         self.model = model
#         return
    
#     def eval(self, x: np.ndarray):
#         if x.ndim == 1:
#             x = x.reshape(1, -1)
#         _, hf_mean = bf_linear_gp_prediction(self.model, x)
#         return hf_mean.flatten()
    
# class bfFcn1D:
#     """Function class representing the 1D bi-fidelity function after 
#     the provided HF evaluation"""
#     def __init__(self,
#                  hf_pos: np.ndarray, # 1D position
#                  hf_fcn: np.ndarray,
#                  lf_pos: np.ndarray, # 1D position
#                  lf_fcn: np.ndarray):
#         model = bf_linear_gp_regression(lf_pos.reshape(-1, 1), lf_fcn.reshape(-1, 1), 
#                                         hf_pos.reshape(-1, 1), hf_fcn.reshape(-1, 1))
#         self.model = model
    
#     def forward(self, x: np.ndarray):
#         if x.ndim == 1:
#             x = x.reshape(1, -1)
#         _, hf_mean = bf_linear_gp_prediction(self.model, x)
#         return hf_mean.flatten()

def grad_desc(x0: np.ndarray,
              obj: objectiveFcn,
              learning_rate: Optional[float] = 1e-2,
              num_iterations: Optional[int] = 1e2, 
              ) -> np.ndarray:
    """ Basic gradient descent, using finite differences to estimate gradient """
    print('======== gradient descent ======')
    x = as_column_vec(x0,copy=True)
    obj.reset()
    for i in range(int(num_iterations)):
        gradx,fx = obj.gradient(x, return_fx=True)
        if np.isnan(fx):
            print('Found NaN')
            return x
        x -= learning_rate*gradx

    return x

def grad_desc_ls(x0: np.ndarray,
              obj,
              learning_rate: Optional[float] = 1.0,  # Initial guess for learning rate
              num_iterations: Optional[int] = 1e2, 
              beta: float = 0.5,  # Backtracking step reduction factor
              c: float = 1e-4,   # Armijo condition parameter
              max_backtrack_iters: int = 20,  # Maximum backtracking iterations
              ) -> np.ndarray:
    """
    Gradient descent with backtracking line search using the Armijo condition.
    
    Parameters:
        x0 : np.ndarray
            Initial guess for the optimization variable.
        obj : Objective function object with methods 'gradient' and 'evaluate'.
        learning_rate : float
            Initial guess for the learning rate.
        num_iterations : int
            Number of iterations for gradient descent.
        beta : float
            Factor by which to reduce the step size during backtracking.
        c : float
            Armijo condition parameter (typically small, e.g., 1e-4).
        max_backtrack_iters : int
            Maximum number of backtracking iterations.
        printEvery : int
            Frequency of printing progress.
    
    Returns:
        np.ndarray
            Optimized variable.
    """
    print('======== gradient descent with backtracking ======')
    x = np.atleast_2d(x0).T if x0.ndim == 1 else x0  # Ensure x is a column vector
    
    obj.reset()  # Reset objective function state

    for i in range(int(num_iterations)):
        gradx, fx = obj.gradient(x, return_fx=True)
        
        if np.isnan(fx):
            print('Found NaN')
            return x
        
        # Backtracking line search with max iterations
        step_size = learning_rate
        for _ in range(max_backtrack_iters):
            x_new = x - step_size * gradx
            fx_new = obj.eval(x_new)  # Assuming 'evaluate' computes the function value at x_new
            if fx_new <= fx - c * step_size * np.dot(gradx.T, gradx):  # Armijo condition
                break
            step_size *= beta  # Reduce step size
        else:
            print(f"Warning: Maximum backtracking iterations reached at iteration {i}.")
        
        x = x - step_size * gradx  # Update x
    
    return x


def coor_desc(x0: np.ndarray,
        obj: objectiveFcn,
        learning_rate: Optional[float] = 1e-2,
        num_iterations: Optional[int] = 1e2, 
        ) -> np.ndarray:
    """ Coordinate descent, using finite differences to estimate gradient"""
    x = as_column_vec(x0,copy=True)
    n = x.shape[0]
    obj.reset()
    print('======== coordinate descent ========')
    for i in range(int(num_iterations)):
        eye = np.eye(n)
        for j in range(n):
            # Standard Coordinate Descent
            U = eye[j].reshape(-1,1)
            gradx,fx = obj.directionalDerivative(x, U, return_fx=True)
            p = U @ gradx  # projected gradient
            x -= learning_rate*p
        
    return x

def coor_desc_ls(x0: np.ndarray,
              obj,
              learning_rate: Optional[float] = 1.0,  # Initial guess for learning rate
              num_iterations: Optional[int] = 1e2,
              beta: float = 0.5,  # Backtracking step reduction factor
              c: float = 1e-4,   # Armijo condition parameter
              max_backtrack_iters: int = 20,  # Maximum backtracking iterations
              ) -> np.ndarray:
    """
    Coordinate descent with backtracking line search using the Armijo condition.
    
    Parameters:
        x0 : np.ndarray
            Initial guess for the optimization variable.
        obj : Objective function object with methods 'directionalDerivative' and 'reset'.
        learning_rate : float
            Initial guess for the learning rate.
        num_iterations : int
            Number of iterations for coordinate descent.
        beta : float
            Factor by which to reduce the step size during backtracking.
        c : float
            Armijo condition parameter (typically small, e.g., 1e-4).
        max_backtrack_iters : int
            Maximum number of backtracking iterations.
        printEvery : int
            Frequency of printing progress.
    
    Returns:
        np.ndarray
            Optimized variable.
    """
    x = np.atleast_2d(x0).T if x0.ndim == 1 else x0  # Ensure x is a column vector
    n = x.shape[0]
    
    obj.reset()  # Reset objective function state
    print('======== coordinate descent with backtracking ========')
    
    for i in range(int(num_iterations)):
        eye = np.eye(n)
        for j in range(n):
            # Standard Coordinate Descent
            U = eye[j].reshape(-1, 1)  # Direction vector
            gradx, fx = obj.directionalDerivative(x, U, return_fx=True)  # Gradient in direction U
            p = U @ gradx  # Projected gradient
            
            # Backtracking line search for the step size
            step_size = learning_rate
            for _ in range(max_backtrack_iters):
                x_new = x - step_size * p
                fx_new = obj.eval(x_new)  # Assuming 'evaluate' computes the function value
                if fx_new <= fx - c * step_size * np.dot(p.T, p):  # Armijo condition
                    break
                step_size *= beta  # Reduce step size
            else:
                print(f"Warning: Maximum backtracking iterations reached for coordinate {j} in iteration {i}.")
            
            # Update the current coordinate
            x -= step_size * p
    
    return x

def ssd(x0: np.ndarray,
        obj: objectiveFcn,
        ell: Optional[int] = 1,
        learning_rate: Optional[float] = 1e-2,
        num_iterations: Optional[int] = 1e2, 
        printEvery: Optional[int] = None,) -> np.ndarray:
    """ Stochastic Substace Descent of Kozak, Tenorio, Becker, Doostan"""
    x = as_column_vec(x0,copy=True)
    n = x.shape[0]
    if printEvery is None: printEvery = int( num_iterations / 10 )
    obj.reset()
    print('======== SSD ===================')
    for i in range(int(num_iterations)):
        # Standard SSO
        U = DFO_utilities.haar_QR(n, ell, ignoreDiagScaling=True, transpose=False)
        gradx,fx = obj.directionalDerivative(x, U, return_fx=True)
        p = U @ gradx  # projected gradient
        x -= learning_rate*p
        
    return x

def ssd_polyak(x0: np.ndarray,
        obj: objectiveFcn,
        ell: Optional[int] = 1,
        num_iterations: Optional[int] = 1e2, 
        f_min: Optional[float] = None,
        printEvery: Optional[int] = None,) -> np.ndarray:
    """ Stochastic Substace Descent of Kozak, Tenorio, Becker, Doostan"""
    x = as_column_vec(x0,copy=True)
    n = x.shape[0]
    if printEvery is None: printEvery = int( num_iterations / 10 )
    obj.reset()
    print('======== SSD-Polyak =================')
    for i in range(int(num_iterations)):
        # Standard SSO
        U = DFO_utilities.haar_QR(n, ell, ignoreDiagScaling=True, transpose=False)
        gradx,fx = obj.directionalDerivative(x, U, return_fx=True)
        p = U @ gradx  # projected gradient
        if f_min is None:
            f_min = np.min(obj.returnHistory()) - fx / i
        learning_rate = min((fx - f_min) / np.linalg.norm(p)**2, 1e-2)
        x -= learning_rate*p
        
    return x

def spsa(x0: np.ndarray,
        obj: objectiveFcn,
        alpha: Optional[float] = 0.602,
        gamma: Optional[float] = 0.101,
        c: Optional[float] = 1e-2,
        num_iterations: Optional[int] = 1e2, 
        printEvery: Optional[int] = None,) -> np.ndarray:
    """ Simutanous Perturbation Stochastic Approximation of Spall"""
    x = as_column_vec(x0,copy=True)
    n = x.shape[0]
    A, a = 0.1 * num_iterations, 0.16
    if printEvery is None: printEvery = int( num_iterations / 10 )
    obj.reset()
    print('======== SPSA ===================')
    for k in range(int(num_iterations)):
        a_k = a/(A+k+1)**alpha
        c_k = c/(k+1)**gamma
        # Standard SSO
        u = (2 * np.random.binomial(1, 0.5, n) - 1).reshape(-1,1)
        p = np.asarray((obj.eval(x+c_k*u) - obj.eval(x-c_k*u))/(2*c_k) * u)
        x -= a_k * p
        
    return x

def ssd_ls(x0: np.ndarray, 
           obj: objectiveFcn,
           obj_lowFi: objectiveFcn = None,
           learning_rate: Optional[float] = 1e-2,
           num_iterations: Optional[int] = 1e2, 
           ell: Optional[int] = 1,
           linesearch_iter: Optional[int] = 20) -> np.ndarray:
    """
    Stochastic Subspace Descent with a low-fidelity model for line search.
    
    Parameters:
    - x0: np.ndarray - Initial point for optimization.
    - obj: objectiveFcn - High-fidelity objective function.
    - obj_lowFi: Optional[objectiveFcn] - Low-fidelity objective function; must be provided.
    - learning_rate: float - Base learning rate for line search.
    - num_iterations: int - Maximum number of iterations.
    - print_every: Optional[int] - Interval for printing progress.
    - ell: int - Number of random subspace directions.
    - linesearch_iter: int - Number of iterations for line search.

    Returns:
    - np.ndarray: Optimized point after performing the algorithm.
    """
    x = as_column_vec(x0,copy=True)
    n = x.shape[0]
    obj.reset()
    if obj_lowFi is not None:
        obj_lowFi.reset()
    else:
        raise ValueError("Must provide low-fidelity model")

    print('======== SSD w/ low-fi linesearch =')
    for i in range(int(num_iterations)):
        U = DFO_utilities.haar_QR(n,ell,ignoreDiagScaling=True,transpose=False)
        gradx = obj.directionalDerivative(x, U)
        p = U @ gradx  # projected gradient
        step_sizes = learning_rate * np.logspace(-2,2,num=linesearch_iter) # one way to do it
        fVals = [obj_lowFi.eval(x.ravel() - step_size * p.ravel() ) for step_size in step_sizes]
        # fVals = [obj_lowFi.eval(x.ravel() - step_size * p.ravel() ) for step_size in step_sizes]
        learning_rate =  step_sizes[np.argmin(fVals)]

        x -= learning_rate * p

        
    return x

def ssd_ls_temp(x0: np.ndarray, 
           obj: objectiveFcn,
           obj_lowFi: objectiveFcn = None,
           learning_rate: Optional[float] = 1e-2,
           num_iterations: Optional[int] = 1e2, 
           printEvery: Optional[int] = None,
           ell: Optional[int] = 1,
           linesearch_iter: Optional[int] = 20) -> np.ndarray:
    """ Stochastic Substace Descent with low-fidelity model for linesearch """
    x = as_column_vec(x0,copy=True)
    n = x.shape[0]
    if printEvery is None: printEvery = int( num_iterations / 10 )
    obj.reset()
    if obj_lowFi is not None:
        obj_lowFi.reset()
    else:
        raise ValueError("Must provide low-fidelity model")

    print('======== SSD w/ low-fi linesearch =')
    # Evaluate high- and low-fidelity models at random points to calibrate
    x_init = np.random.rand(5, n)
    for x_init_i in x_init:
        obj.eval(x_init_i)
        obj_lowFi.eval(x_init_i)
    for i in range(int(num_iterations)):
        U = DFO_utilities.haar_QR(n,ell,ignoreDiagScaling=True,transpose=False)
        gradx = obj.directionalDerivative(x, U)
        p = U @ gradx  # projected gradient
        step_sizes = learning_rate * np.logspace(-2,2,num=linesearch_iter) # one way to do it
        # step_sizes = learning_rate * np.logspace(-2,2,num=linesearchIter) # another way to do it, but often then gets stuck at tiny steps
        # if calibrated:
        #     cand_x = []
        #     for step_size in step_sizes:
        #         x_new = x.ravel() - step_size * p.ravel()
        #         cand_x.append(x_new)
        #         _ = obj_lowFi.eval(x_new)
        #     caliModel.update()
        #     fVals = caliModel.eval(np.asarray(cand_x))
        # else:
        fVals = [obj_lowFi.eval(x.ravel() - step_size * p.ravel() ) for step_size in step_sizes]
        # fVals = [obj_lowFi.eval(x.ravel() - step_size * p.ravel() ) for step_size in step_sizes]
        learning_rate =  step_sizes[np.argmin(fVals)]

        x -= learning_rate * p

        
    return x

def ssd_bt_temp(x0: np.ndarray, 
           obj: objectiveFcn,
           obj_lowFi: objectiveFcn = None,
           num_iterations: Optional[int] = 1e2, 
           printEvery: Optional[int] = None,
           ell: Optional[int] = 1,
           linesearch_iter: Optional[int] = 20,
           beta: Optional[float] = None,
           c: Optional[float] = 0.9,
           L0: Optional[float] = 1.0) -> np.ndarray:
    """ Stochastic Substace Descent with low-fidelity model for linesearch """
    x = as_column_vec(x0,copy=True)
    n = x.shape[0]
    ell -= 1
    if printEvery is None: printEvery = int( num_iterations / 10 )
    if beta is None: beta = 0.5 * ell / n
    obj.reset()
    if obj_lowFi is not None:
        obj_lowFi.reset()
    else:
        raise ValueError("Must provide low-fidelity model")
    print('======== SSD w/ backtracking bi-fi linesearch =')
    for i in range(int(num_iterations)):
        U = DFO_utilities.haar_QR(n,ell,ignoreDiagScaling=True,transpose=False)
        gradx, fx = obj.directionalDerivative(x, U, return_fx=True)
        p = U @ gradx  # projected gradient
        v = p/np.linalg.norm(p)
        # Build surrogate
        hf_L0 = obj.eval(x.ravel() + L0 * p.ravel())
        lf_0, lf_L0 = obj_lowFi.eval(x.ravel()), obj_lowFi.eval(x.ravel() + L0 * v.ravel())
        rho = hf_L0 / lf_L0
        bi_func = lambda ss: rho * obj_lowFi.eval(x.ravel()+ss*v.ravel()) +\
              (hf_L0 - rho * lf_L0 - fx + rho * lf_0)/L0 * ss + fx - rho * lf_0

        step_sizes = [c**n*L0 for n in range(linesearch_iter)]# one way to do it
        fVals = [bi_func(ss) for ss in step_sizes]
        learning_rate = step_sizes[-1]
        for i, step_size in enumerate(step_sizes):
            if fVals[i] < fx - beta * step_size * np.linalg.norm(p)**2 * ell/n:
                learning_rate = step_size
                break

        x -= learning_rate * p

        
    return x

def ssd_bt_temp_HermiteInterp(x0: np.ndarray, 
           obj: objectiveFcn,
           obj_lowFi: objectiveFcn = None,
           num_iterations: Optional[int] = 1e2, 
           printEvery: Optional[int] = None,
           ell: Optional[int] = 1,
           linesearch_iter: Optional[int] = 20,
           beta: Optional[float] = None,
           c: Optional[float] = 0.9,
           L0: Optional[float] = 1.0) -> np.ndarray:
    """ Stochastic Substace Descent with bi-fidelity model for linesearch, ver 2026 with Hermite interpolation """
    x = as_column_vec(x0,copy=True)
    n = x.shape[0]
    ell -= 1
    if printEvery is None: printEvery = int( num_iterations / 10 )
    if beta is None: beta = 0.5 * ell / n
    obj.reset()
    if obj_lowFi is not None:
        obj_lowFi.reset()
    else:
        raise ValueError("Must provide low-fidelity model")
    print('======== SSD w/ backtracking bi-fi linesearch, 2026 variant with Hermite interpolation =')
    for i in range(int(num_iterations)):
        U = DFO_utilities.haar_QR(n,ell,ignoreDiagScaling=True,transpose=False)
        gradx, fx = obj.directionalDerivative(x, U, return_fx=True)
        p = U @ gradx  # projected gradient
        v = p/np.linalg.norm(p)
        # Build surrogate
        # hf_L0 = obj.eval(x.ravel() + L0 * p.ravel()) # Not needed in this variant
        # lf_0, lf_L0 = obj_lowFi.eval(x.ravel()), obj_lowFi.eval(x.ravel() + L0 * v.ravel())
        # -- New Hermite interpolation --
        hf_deriv = np.linalg.norm(p)**2 # derivative of the 1D function at ss=0
        p_lf, lf_0 = obj_lowFi.directionalDerivative(x.ravel(), U, return_fx=True)
        lf_deriv = np.dot( p_lf, p ) # derivative of the 1D function at ss=0
        rho = hf_deriv / lf_deriv
        bi_func = lambda ss: rho * obj_lowFi.eval(x.ravel()+ss*v.ravel()) + fx - lf_0

        step_sizes = [c**n*L0 for n in range(linesearch_iter)]# one way to do it
        fVals = [bi_func(ss) for ss in step_sizes]
        learning_rate = step_sizes[-1]
        for i, step_size in enumerate(step_sizes):
            if fVals[i] < fx - beta * step_size * np.linalg.norm(p)**2 * ell/n:
                learning_rate = step_size
                break

        x -= learning_rate * p

        
    return x

def ssd_bt(
    x0: np.ndarray, 
    obj: objectiveFcn, 
    obj_lowFi: Optional[objectiveFcn] = None, 
    num_iterations: int = 100, 
    ell: int = 1, 
    linesearch_iter: int = 20, 
    beta: Optional[float] = None, 
    c: float = 0.9, 
    L0: float = 1.0
) -> np.ndarray:
    """
    Performs Stochastic Subspace Descent with a bi-fidelity model for line search.
    
    Parameters:
    - x0: np.ndarray - Initial point for optimization.
    - obj: objectiveFcn - High-fidelity objective function.
    - obj_lowFi: Optional[objectiveFcn] - Low-fidelity objective function; must be provided.
    - num_iterations: int - Maximum number of iterations.
    - print_every: Optional[int] - Interval for printing progress.
    - ell: int - Number of random subspace directions.
    - linesearch_iter: int - Maximum iterations for line search.
    - beta: Optional[float] - Step size adjustment parameter.
    - c: float - Line search scaling factor.
    - L0: float - Initial step length for line search.

    Returns:
    - np.ndarray: Optimized point after performing the algorithm.
    """
    
    x = as_column_vec(x0, copy=True)
    n = x.shape[0]
    ell -= 1  # Adjust ell for internal calculations

    if beta is None:
        beta = 0.5 * ell / n

    obj.reset()
    if obj_lowFi is None:
        raise ValueError("Low-fidelity model must be provided.")
    obj_lowFi.reset()
    
    print('======== SSD with Backtracking Bi-fidelity Line Search ========')
    
    for iteration in range(int(num_iterations)):
        U = DFO_utilities.haar_QR(n, ell, ignoreDiagScaling=True, transpose=False)
        gradx, fx = obj.directionalDerivative(x, U, return_fx=True)
        
        # Compute and normalize the projected gradient
        p = U @ gradx
        p_norm = np.linalg.norm(p)
        p /= p_norm

        # Evaluate high- and low-fidelity models for surrogate building
        hf_L0 = obj.eval(x.ravel() + L0 * p.ravel())
        lf_0, lf_L0 = obj_lowFi.eval(x.ravel()), obj_lowFi.eval(x.ravel() + L0 * p.ravel())
        
        rho = hf_L0 / lf_L0
        bi_func = lambda ss: (
            rho * obj_lowFi.eval(x.ravel() + ss * p.ravel()) + 
            (hf_L0 - rho * lf_L0 - fx + rho * lf_0) / L0 * ss + 
            fx - rho * lf_0
        )

        # Line search for optimal step size
        value = bi_func(L0)
        cter = 0
        fctr = 1
        while cter <= linesearch_iter and \
        value > fx - beta * c**ell * p_norm**2 * ell / n:
            cter += 1
            fctr *= c
            value = bi_func(fctr*L0)
        learning_rate = fctr * L0
        x -= learning_rate * p

    return x

def ssd_hbt(
    x0: np.ndarray, 
    obj: objectiveFcn, 
    num_iterations: int = 100, 
    ell: int = 1, 
    linesearch_iter: int = 20, 
    beta: Optional[float] = None, 
    c: float = 0.9, 
    L0: float = 1.0
) -> np.ndarray:
    """
    Stochastic Subspace Descent with single-fidelity model for backtracking line search.
    
    Parameters:
    - x0: np.ndarray - Initial point for optimization.
    - obj: objectiveFcn - High-fidelity objective function.
    - num_iterations: int - Maximum number of iterations.
    - ell: int - Number of random subspace directions.
    - linesearch_iter: int - Maximum iterations for line search.
    - beta: Optional[float] - Step size adjustment parameter.
    - c: float - Line search scaling factor.
    - L0: float - Initial step length for line search.

    Returns:
    - np.ndarray: Optimized point after performing the algorithm.
    """
    
    x = as_column_vec(x0, copy=True)
    n = x.shape[0]
    ell -= 1  # Adjust ell for internal calculations

    if beta is None:
        beta = 0.5 * ell / n

    obj.reset()
    
    print("======== SSD with HF Backtracking Line Search ========")
    
    for iteration in range(int(num_iterations)):
        U = DFO_utilities.haar_QR(n, ell, ignoreDiagScaling=True, transpose=False)
        gradx, fx = obj.directionalDerivative(x, U, return_fx=True)
        
        # Compute and normalize the projected gradient
        p = U @ gradx
        p_norm = np.linalg.norm(p)
        p /= p_norm

        # Line search for optimal step size
        value = obj.eval(x.ravel() - L0*p.ravel())
        cter = 0
        fctr = 1
        while cter <= linesearch_iter and \
        value > fx - beta * c**ell * p_norm**2 * ell / n:
            cter += 1
            fctr *= c
            value = obj.eval(x.ravel() - fctr*L0*p.ravel())
        learning_rate = fctr * L0
        x -= learning_rate * p

    return x

def ssd_sag(x0: np.ndarray,
        obj: objectiveFcn,
        ell: Optional[int] = 1,
        learning_rate: Optional[float] = 1e-2,
        num_iterations: Optional[int] = 1e2,) -> np.ndarray:
    """ Stochastic Substace Descent of Kozak, Tenorio, Becker, Doostan"""
    x = as_column_vec(x0,copy=True)
    n = x.shape[0]
    obj.reset()
    print('======== SSD ===================')
    g = np.zeros_like(x)
    for i in range(int(num_iterations)):
        # SSO with SAG
        U = DFO_utilities.haar_QR(n, ell, ignoreDiagScaling=True, transpose=False)
        gradx = obj.directionalDerivative(x, U)
        p = U @ gradx + g - U @ (U.T @ g) # projected gradient with previous gradient
        x -= learning_rate*p
        g = p
        
    return x

def ssd_bt_sag(x0: np.ndarray, 
           obj: objectiveFcn,
           obj_lowFi: objectiveFcn = None,
           num_iterations: Optional[int] = 1e2, 
           printEvery: Optional[int] = None,
           ell: Optional[int] = 1,
           linesearchIter: Optional[int] = 20,
           beta: Optional[float] = None,
           c: Optional[float] = 0.9,
           L0: Optional[float] = 1.0) -> np.ndarray:
    """ Stochastic Substace Descent with low-fidelity model for linesearch """
    x = as_column_vec(x0,copy=True)
    n = x.shape[0]
    ell -= 1
    if beta is None: beta = 0.5 * ell / n
    obj.reset()
    if obj_lowFi is not None:
        obj_lowFi.reset()
    else:
        raise ValueError("Must provide low-fidelity model")
    print('======== SSD w/ backtracking bi-fi linesearch =')
    g = np.zeros_like(x)
    for i in range(int(num_iterations)):
        U = DFO_utilities.haar_QR(n,ell,ignoreDiagScaling=True,transpose=False)
        gradx, fx = obj.directionalDerivative(x, U, return_fx=True)
        p = U @ gradx + g - U @ (U.T @ g) # projected gradient with previous gradient
        v = p/np.linalg.norm(p)
        # Build surrogate
        hf_L0 = obj.eval(x.ravel() + L0 * p.ravel())
        lf_0, lf_L0 = obj_lowFi.eval(x.ravel()), obj_lowFi.eval(x.ravel() + L0 * p.ravel())
        rho = hf_L0 / lf_L0
        bi_func = lambda ss: rho * obj_lowFi.eval(x.ravel()+ss*p.ravel()) +\
              (hf_L0 - rho * lf_L0 - fx + rho * lf_0)/L0 * ss + fx - rho * lf_0

        step_sizes = [c**n*L0 for n in range(linesearchIter)]# one way to do it
        fVals = [bi_func(ss) for ss in step_sizes]
        learning_rate = step_sizes[-1]
        for i, step_size in enumerate(step_sizes):
            if fVals[i] < fx - beta * step_size * np.linalg.norm(p)**2 * ell/n:
                learning_rate = step_size
                break

        x -= learning_rate * p
        g = p

    return x
