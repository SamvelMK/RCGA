import numpy as np
import pandas as pd
from ypstruct import structure

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
    
def conceptUpdate(state_vector, weightMat, l=0.001):
    """
    FCM update function.
    --------------------
    state_vector: initial state vector
    weightMat: the connection matrix.
    l: smoothing paramenter for the sigmoid function.
    """
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
        df = initial.apply(lambda x: conceptUpdate(x, weightMat), axis=1)
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

def genChromosome(Ngen):
    """
    generate a chromosome with N number of genes:
    --------------------------------------------
    Ngen: int, number of genes in a chromosome
    --------------------------------------------
    return: structure
    """
    chromosome = structure()
    for i in range(Ngen):
        for y in range(Ngen):
            chromosome[f'{i},{y}'] = np.random.uniform(-1, 1)
    return chromosome

def initialPopulation(Nchrom, Ngen):
    """
    create an initial set of random solutions.
    ------------------------------
    Nchrom: number of chromosomes in a solution set
    Ngen: number of genes in each chromosome
    """
    initial_pop = [genChromosome(Ngen) for i in range(Nchrom)]
    return initial_pop

def decode(dimensions, chromosome):
    """
    convert a chromosome structure to a connection matrix.
    ------------------------------------------------------
    dimensions: tuple, dimensions of the connection matrix
    chromosome: structure, the candidate chromosome.
    """
    emptyW = np.zeros((dimensions[0],dimensions[1]))
    for i in chromosome.keys():
        if i != 'fitness':
            res = tuple(map(int, i.split(',')))
            emptyW[res] = chromosome[i]
    return emptyW

def maxFit(candidateList):
    fit = []
    for i in candidateList:
        fit.append(i.fitness)
    return max(fit)

def selectBestFit(fitnessScore, population):
    res = [i for i in population if i.fitness == fitnessScore]
    return res

def tournament(population, K, nPop, initial_state, tn):
    """
    run a tournament between the elements in the initial solution set.
    pop: the initial solution set.
    K: number of participants in the tournament.
    nPop: number of chromosomes in the population to be generated.
    initial_state: pd.Series. The observations at t=0.
    tn: pd.df. the columns of the df represent the observations at t=n. 
    """
    counter = 0
    pop = []
    while len(pop) < nPop:
        
        # clear_output(wait=True)
        print(f'Running the turnament among {K} chromosomes.')
        print(f'Number of chromosomes selected: {counter}')
        
        participants = np.random.choice(population,K)
        for participant in participants:
            candidate = decode((len(initial_state.columns), len(initial_state.columns)), participant)
            participant.fitness = costFunc(initial_state, tn, candidate, 2, 100)
        # select the candidate with the best fit and append it to the population.
        bestFit = selectBestFit(maxFit(participants), initial_pop)
        pop.append(bestFit)
        counter+=1
        
    return pop, fitScores

def singlePointCrossover(parentOne, parentTwo, nVar):
    singlePoint = nVar//2
    childOne = structure(list(parentOne.items())[:singlePoint] + list(parentTwo.items())[singlePoint:nVar])
    childTwo = structure(list(parentOne.items())[singlePoint:nVar] + list(parentTwo.items())[:singlePoint])
    return childOne, childTwo