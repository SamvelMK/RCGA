import numpy as np
from simulator.simulator import FcmSimulator
def sig(x, l):
    """ Sigmoidal transfer function.
        
        Parameters
        ----------
        x : float,
        l : A parameter that determines the steepness of the sigmoid and hyperbolic tangent function at values around 0. 
        
        Return
        ----------
        y : float,
            domain R,
            range [0,1].
        """
    e = np.exp(1)
    res = 1/(1+(e**(-l*x)))
    return res

def h(x, a):
    return 1/(a*x+1)

def b(x, S):
    return np.sqrt(sum(x)/S)
    
def conceptUpdate(state_vector, weightMat, l):
    
    res = sig(weightMat.dot(state_vector).sum()+state_vector, l)
    return res

def predict(initial_state, weightMat, nIter,  l=0.001):
    """
    initial_state: df of the initial states
    nIter: number of iterations (n=obs(t))
    l: a parameter for the sigmoid function for updating the FCM
    """
    initial = initial_state
    res = {}
    for i in range(nIter):
        df = initial.apply(lambda x: conceptUpdate(x, weightMat, l), axis=1)
        df.rename(lambda x: x+f't{i+1}', axis='columns', inplace=True)
        res[f't{i+1}'] = df
        initial = df.copy(deep=True)
        initial.columns = initial_state.columns
    predicted = pd.concat([i for i in res.values()], axis = 1)
    
    return predicted

def costFunc(initial_state, tn, weightMatrix, t, sampleSize, p = 2, a = 100):
    """ 
    initial_state: pd.Series. The observations at t=0.
    tn: pd.df. the columns of the df represent the observations at t=n.
    """
    observed = tn    
    predicted = predict(initial_state, weightMatrix ,t)
    
    alpha = 1/(t-1)
    cost = h(b(alpha*(abs(observed - predicted)**p).sum(), sampleSize), a)
    return cost

