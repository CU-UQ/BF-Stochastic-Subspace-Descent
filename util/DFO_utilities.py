# Nov 2022, Stephen Becker
import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm, qr

eps = np.finfo(float).eps

def project(x,lower,upper):
    ''' 1D projection of x onto [lower,upper] '''
    return np.maximum( np.minimum( x,upper), lower )

def as_column_vec(x,copy=False):
  """ Input x is of size (n,) or (n,1) or (1,n)
  and output is always of size (n,1). This lets us be consistent """
  # by default, np.array makes a copy; np.asarray does not
  return np.reshape( np.array(x,copy=copy).ravel(), (-1,1) )

# Copied from Stephen's 2018 Matlab code
def directionalDerivative_centered(x,p,fcn, h=None):
    """
    df(x,p) = lim_{h-->0}  [ f(x+h*p) - f(x) ]/h
    But use centered differences for 2nd order accuracy
    Following Andrei 2009 (cited on Tim Veira's blog) for choice of h
    Tim Veira's blog: http://timvieira.github.io/blog/post/2014/02/10/gradient-vector-product/
    Neculai Andrei's paper is:
    "Accelerated conjugate gradient algorithm with finite difference
     Hessian/vector product approximation for unconstrained optimization"
    (Actually, Andrei uses something a bit different. Let's revert
     to Nocedal and Wright)
     """
    if h is None:
        #h   = np.sqrt(eps)*( 1 + norm(x,np.inf) )/norm(p,np.inf)
        # May 22 2018, changing to:
        h   = eps**(1/3)*( 1 + norm(x,np.inf) )/norm(p,np.inf)
    if x.ndim > 1 :
        # x is size (d,1) so change p from possible (d,) to (d,1)
        # p.resize( (-1,1)) # .resize changes in place, .reshape returns a copy. But this causes problems if p is a slice
        p = p.reshape( (-1,1) )
    f1  = fcn(x+h*p)
    f2  = fcn(x-h*p)
    gp  = ( f1 - f2 )/(2*h)
    return np.array(gp)

def directionalDerivative_forward( x, p, fcn, f0 = None, h=None ):
    """
    Inputs:  x, p, fcn, f0 = None, h=None
        p should be a single column
    Outputs: gp, f0
    df(x,p) = lim_{h-->0}  [ f(x+h*p) - f(x) ]/h
    Uses 1st order, and can re-use computation of f0 = f(x) if precomputed.
    Following Andrei 2009 (cited on Tim Veira's blog) for choice of h
    """
    if h is None:
        h   = np.sqrt(eps)*( 1 + norm(x,np.inf) )/norm(p,np.inf)
    if x.ndim > 1 :
        # x is size (d,1) so change p from possible (d,) to (d,1)
        # p.resize( (-1,1)) # .resize changes in place, .reshape returns a copy.  But this causes problems if p is a slice
        p = p.reshape( (-1,1) )
    f1  = fcn(x+h*p) # if x is size (d,1) and p is (d,) then p is broadcast to (1,d) and x+p is size (d,d) which causes huge problems!!
    if f0 is None:
        f0 = fcn(x)

    gp  = ( f1 - f0 )/h
    return np.array(gp), f0


def directionalDerivative(x,p,fcn,centeredDifferences=False, fx=None, h=None ):
    """
  Basic code to compute directional derivatives, good value of eps

  x is current point, p is direction (or matrix, each column a
  direction),

  centeredDifferences is either true or false; if true, uses 
      centered difference method, otherwise uses forward differences
      Centered differences is more accurate but requires twice the fcn
      calls.

  fx [optional] is f(x). If this is already computed, you can supply
      it here to save time.
  h [optional] is a stepsize; if not provided, a reasonable estimate is used

Stephen Becker, stephen.becker@colorado.edu, 2017/2018
    Python version November 2022
    """

    # simple wrapper, allows "fx" to be optional
    # also loops over columns in p
    x = np.asarray(x)
    nCols = np.shape(p)[1]
    gp = np.zeros( (nCols,) ) # NOT (nCols,1)
    for col in range(nCols):
        if centeredDifferences:
            gp[col] = directionalDerivative_centered(x,p[:,col],fcn,h=h)
            fx = None
        else:
            gpp, fx = directionalDerivative_forward(x,p[:,col],fcn, fx,h=h )
            gp[col] = gpp
    return gp, fx

# adding Nov 2023
def gradient(x,fcn, centeredDifferences=False, fx=None, h=None):
    """
    Finds the gradient via finite-differences
    fx [optional] is f(x). If this is already computed, you can supply
      it here to save time.
    h [optional] is a stepsize; if not provided, a reasonable estimate is used
    """
    d = np.shape(x)[0]
    p = np.eye(d)
    return directionalDerivative(x,p,fcn,centeredDifferences,fx,h)
    


def haar_QR( n, m=None, rng = None, scale=False, transpose = True, ignoreDiagScaling=False ):
    """
    Circular Orthogonal Ensemble (COE)
    
    For more info, see http://www.ams.org/notices/200705/fea-mezzadri-web.pdf
    "How to Generate Random Matrices fromthe Classical Compact Groups"
    by Francesco Mezzadri (also at http://arxiv.org/abs/math-ph/0609050 )

    Other references:
    "How to generate a random unitary matrix" by Maris Ozols
    """
    if m is None:
        m = n
    if rng is None:
        rng = default_rng()
    z = rng.standard_normal( size=(n,m) )
    Q = qr( z, mode="reduced")[0]

    if not ignoreDiagScaling:
        raise ValueError("Not yet implemented, need to convert Matlab code to Python")
        # (Actually Mezzadri paper had python code!)

        # Do the Mezzadri adjustment:
        # d = diag(R);
        # ph = d./abs(d);
        # % Q = multiply(q,ph,q) % in python. this is q <-- multiply(q,ph)
        # %   where we multiply each column of q by an element in ph

        # Q = Q.*repmat( ph', n, 1 );

        # Q   = Q(:,1:m);
    
    if scale and (n != m):
        Q *=  np.sqrt(n/m)
    if transpose:
        Q = Q.T
    return Q



def ArmijoBackgrackingLinesearch(x0,fcn,g,stepsize,fxOld=None,p=None,armijoConstant=1e-5,stepsizeFactor=0.5,
        maxLinesearchIters = 20, stepmin = 1e-10):
    """ Armijo condition backtracking linesearch (very simple)
    Inputs: x0 (current point), fcn (functon), g (gradient), stepsize, 
    Optional: fxOld = f(x), p (direction, e.g., -g)
    Optional: armijoConstant, stepsizeFactor (between 0 and 1), maxlinesearchIters, stepmin
    Returns: xNew, fxOld, stepsize, linesearchIters, flag in {"max iters","too small","success"}
    """
    linesearchIters = 0
    x = x0.copy()  # possibly unnecessary, but better safe than sorry. Do *not* want side effects!
    if fxOld is None:
        fxOld = fcn(x)
    if p is None:
        p = -g
    flag = 'max iters'
    while linesearchIters < maxLinesearchIters:
        xTrial = x - stepsize*g

        # Test Wolfe or Armjio conditions or similar
        fx = fcn(xTrial)
        if fx < fxOld + armijoConstant*stepsize*np.dot(g,p) :
            flag = 'success'
            break
        elif stepsize < stepmin :
            flag = 'too small'
            break
        else:
            linesearchIters = linesearchIters + 1
            stepsize *= stepsizeFactor

    return xTrial, fx, stepsize, linesearchIters, flag
