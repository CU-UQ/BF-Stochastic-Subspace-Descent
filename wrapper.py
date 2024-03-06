from DFO_utilities import as_column_vec
import DFO_utilities
import numpy as np
from typing import Callable, Optional
from bf_regression import *

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

class calibrateFcn:
    """Function class representing the calibration function after 
    the provided HF evaluation"""
    def __init__(self,
                 objHF: objectiveFcn,
                 objLF: objectiveFcn):
        self.objHF = objHF
        self.objLF = objLF
        self.update()

    def update(self):
        hf_fcn, hf_pos = self.objHF.returnHistory(return_pos=True)
        lf_fcn, lf_pos = self.objLF.returnHistory(return_pos=True)
        model = bf_linear_gp_regression(lf_pos, lf_fcn.reshape(-1, 1), 
                                        hf_pos, hf_fcn.reshape(-1, 1))
        self.model = model
        return
    
    def eval(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        _, hf_mean = bf_linear_gp_prediction(self.model, x)
        return hf_mean.flatten()
    
class bfFcn1D:
    """Function class representing the 1D bi-fidelity function after 
    the provided HF evaluation"""
    def __init__(self,
                 hf_pos: np.ndarray, # 1D position
                 hf_fcn: np.ndarray,
                 lf_pos: np.ndarray, # 1D position
                 lf_fcn: np.ndarray):
        model = bf_linear_gp_regression(lf_pos.reshape(-1, 1), lf_fcn.reshape(-1, 1), 
                                        hf_pos.reshape(-1, 1), hf_fcn.reshape(-1, 1))
        self.model = model
    
    def forward(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        _, hf_mean = bf_linear_gp_prediction(self.model, x)
        return hf_mean.flatten()

def grad_desc(x0: np.ndarray,
              obj: objectiveFcn,
              learning_rate: Optional[float] = 1e-2,
              num_iterations: Optional[int] = 1e2, 
              printEvery: Optional[int] = None) -> np.ndarray:
    """ Basic gradient descent, using finite differences to estimate gradient """
    print('======== gradient descent ======')
    x = as_column_vec(x0,copy=True)
    if printEvery is None: printEvery = int( num_iterations / 10 )
    obj.reset()
    for i in range(int(num_iterations)):
        gradx,fx = obj.gradient(x, return_fx=True)
        if np.isnan(fx):
            print('Found NaN')
            return x
        x -= learning_rate*gradx

    return x


def coor_desc(x0: np.ndarray,
        obj: objectiveFcn,
        learning_rate: Optional[float] = 1e-2,
        num_iterations: Optional[int] = 1e2, 
        printEvery: Optional[int] = None,) -> np.ndarray:
    """ Coordinate descent, using finite differences to estimate gradient"""
    x = as_column_vec(x0,copy=True)
    n = x.shape[0]
    if printEvery is None: printEvery = int( num_iterations / 10 )
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
           printEvery: Optional[int] = None,
           ell: Optional[int] = 1,
           linesearchIter: Optional[int] = 20) -> np.ndarray:
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
        step_sizes = learning_rate * np.logspace(-2,2,num=linesearchIter) # one way to do it
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

def ssd_bt(x0: np.ndarray, 
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

        step_sizes = [c**n*L0 for n in range(linesearchIter)]# one way to do it
        fVals = [bi_func(ss) for ss in step_sizes]
        learning_rate = step_sizes[-1]
        for i, step_size in enumerate(step_sizes):
            if fVals[i] < fx - beta * step_size * np.linalg.norm(p)**2 * ell/n:
                learning_rate = step_size
                break

        x -= learning_rate * p

        
    return x

def ssd_hbt(x0: np.ndarray, 
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
    if printEvery is None: printEvery = int( num_iterations / 10 )
    if beta is None: beta = 0.5 * ell / n
    obj.reset()
    if obj_lowFi is not None:
        obj_lowFi.reset()
    else:
        raise ValueError("Must provide low-fidelity model")
    print('======== SSD w/ backtracking linesearch =')
    for i in range(int(num_iterations)):
        U = DFO_utilities.haar_QR(n,ell,ignoreDiagScaling=True,transpose=False)
        gradx, fx = obj.directionalDerivative(x, U, return_fx=True)
        p = U @ gradx  # projected gradient
        # start backtracking
        new_fx = obj.eval(x.ravel() + L0 * p.ravel())
        cnt = 0
        ss = L0
        while new_fx >= fx - beta * ss * np.linalg.norm(p)**2 * ell/n:
            ss *= c
            new_fx = obj.eval(x.ravel() + ss * p.ravel())
            cnt += 1
            if cnt > linesearchIter:
                break
        learning_rate = ss 

        x -= learning_rate * p
        # if not np.mod(i,int(printEvery)):
        #     print(f'Iter {i:3d}, f(x) is {fx:g}')
        
    return x