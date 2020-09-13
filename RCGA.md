# Real-Coded Genetic Algorithm for FCMs

### Learning objective:

Determine a connection matrix <img src="https://latex.codecogs.com/gif.latex?\hat{E}" title="\hat{E}" /> given an input data <img src="https://latex.codecogs.com/gif.latex?C_{t}" title="C_{t}" /> at certain iteration interval t ∈ {0,...,tK} such that the FCM minimizes the error between the observed <img src="https://latex.codecogs.com/gif.latex?C_{t}" title="C_{t}" /> and the predicted values <img src="https://latex.codecogs.com/gif.latex?\hat{C_{t+1}}" title="C_{t}" />.
<br>
### Chromosome Structure
Each chromosome is a vector of floating point numbers ranging from [-1, 1]. Each element of the vector is called gene. The length of the chromosome corresponds to the number of variables in a given problem (N(N-1)). 
<br><br>
<img src="https://latex.codecogs.com/gif.latex?\hat{E}&space;=&space;[e_{12},e_{13},...,e_{1N},e_{21},e_{23},...,e_{2N},...,e_{NN-1}]^{T}" title="\hat{E} = [e_{12},e_{13},...,e_{1N},e_{21},e_{23},...,e_{2N},...,e_{NN-1}]^{T}" />
<br><br> 
where <img src="https://latex.codecogs.com/gif.latex?e_{ij}" title="e_{ij}" /> specifies the value of a weight for an edge from ith to jth concept node.
<br><br>
The number ofchromosomes in a population is constant for each generation, and it is specified by the population_size parameter.