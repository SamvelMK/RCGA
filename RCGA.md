# Real-Coded Genetic Algorithm for FCMs

### Learning objective:

Determine a connection matrix <img src="https://latex.codecogs.com/gif.latex?\hat{E}" title="\hat{E}" /> given an input data <img src="https://latex.codecogs.com/gif.latex?C_{t}" title="C_{t}" /> at certain iteration interval t ∈ {0,...,tK} such that the FCM minimizes the <em>average</em> error between the observed <img src="https://latex.codecogs.com/gif.latex?C_{t}" title="C_{t}" /> and the predicted values <img src="https://latex.codecogs.com/gif.latex?\hat{C}_{t+1}" title="C_{t}" />.
<br>
### Chromosome Structure
Each chromosome is a vector of floating point numbers ranging from [-1, 1]. Each element of the vector is called gene. The length of the chromosome corresponds to the number of variables in a given problem (N(N-1)). 
<br><br>
<img src="https://latex.codecogs.com/gif.latex?\hat{E}&space;=&space;[e_{12},e_{13},...,e_{1N},e_{21},e_{23},...,e_{2N},...,e_{NN-1}]^{T}" title="\hat{E} = [e_{12},e_{13},...,e_{1N},e_{21},e_{23},...,e_{2N},...,e_{NN-1}]^{T}" />
<br><br> 
where <img src="https://latex.codecogs.com/gif.latex?e_{ij}" title="e_{ij}" /> specifies the value of a weight for an edge from ith to jth concept node.
<br><br>
The number ofchromosomes in a population is constant for each generation, and it is specified by the population_size parameter.

Let us assume that the input data length is <em>K</em>, where all iterations that happen after a limit cycle or fixed
pattern attractor occur, are already ignored. By grouping each two adjacent state vectors, <em>K−1</em> different
pairs can be formed:

<img src="https://latex.codecogs.com/gif.latex?C_{t}\rightarrow&space;C_{t}&space;\forall&space;=&space;0,&space;...,K-1" title="C_{t}\rightarrow C_{t} \forall = 0, ...,K-1" />

If we define <img src="https://latex.codecogs.com/gif.latex?C_{t}" title="C_{t}" /> as an <em>initial vector</em>, and <img src="https://latex.codecogs.com/gif.latex?C_{t+1}" title="C_{t}" /> as system response, <em>K-1</em> pairs in the form of {initial vector, system response} can be generated from the input data. The larger is <em>K</em> the more information about the system behavior we have. The fintness function is calculated for each chromosome by computing the difference between system response generated using a candidate FCM and a corresponding system response, which is known directly from the input data. 

The system response of the candidate FCM is computed by decoding the chromosome into a FCM model and performing <b>one</b> iteration simulation for initial state vector equal to the initial vector. The difference is computed across all <em>K−1</em> initial vector/system response pairs, and for the same initial state vector.

<img src="https://latex.codecogs.com/gif.latex?Error_{L_{p}}=&space;\alpha\sum_{t=1}^{K-1}\sum_{n=1}^{N}|C_{n}(t)-\hat{C_{n}(t)}|^{P})" title="Error_{L_{p}}= \alpha\sum_{t=1}^{K-1}\sum_{n=1}^{N}|C_{n}(t)-\hat{C_{n}(t)}|^{P})" />

where <img src="https://latex.codecogs.com/gif.latex?C(t)=[C_{1}(t),C_{2}(t),...,&space;C_{n}(t)]" title="C(t)=[C_{1}(t),C_{2}(t),..., C_{n}(t)]" /> the known system response for <img src="https://latex.codecogs.com/gif.latex?C_{t-1}" title="C_{t-1}" /> initial vector, <img src="https://latex.codecogs.com/gif.latex?\hat{C_{t}}=[\hat{C_{1}}(t),&space;\hat{C_{2}}(t),...,&space;\hat{C_{n}}(t)]" title="\hat{C_{t}}=[\hat{C_{1}}(t), \hat{C_{2}}(t),..., \hat{C_{n}}(t)]" /> the system response of the candidate FCM for <img src="https://latex.codecogs.com/gif.latex?C_{t-1}" title="C_{t-1}" /> initial vector, <em>p = 1, 2 <img src="https://latex.codecogs.com/gif.latex?\infty" title="\infty" /> </em> the norm type, <img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /> the parameter used to normalize error rate, which is equal to 1/(K-1)*N for  <img src="https://latex.codecogs.com/gif.latex?p\in\begin{Bmatrix}&space;1,2&space;\end{Bmatrix}" title="p\in\begin{Bmatrix} 1,2 \end{Bmatrix}" /> and and 1/K − 1 for p =  <img src="https://latex.codecogs.com/gif.latex?\infty" title="\infty" /> respoectively.


We need to extend this formula to average across all the individuals in the sample:

<img src="https://latex.codecogs.com/gif.latex?\beta\sum_{i=1}^{S}\alpha&space;\sum_{t=1}^{K-1}\sum_{n=1}^{N}|C_{n}(t)-\hat{C_{n}(t)}|_{i}^{p}" title="\beta\sum_{i=1}^{S}\alpha \sum_{t=1}^{K-1}\sum_{n=1}^{N}|C_{n}(t)-\hat{C_{n}(t)}|_{i}^{p}" />

where <img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta" /> is 1/S; S = Sample size.

Another option would be to take take the square root of the average:
<img src="https://latex.codecogs.com/gif.latex?\beta(x)=\sqrt{\frac{1}{S}(x)}" title="\beta(x)=\sqrt{\frac{1}{S}(x)}" />



Fitness function = <em>h</em>(Error_Lp)
where h is:

<img src="https://latex.codecogs.com/gif.latex?h(x)=\frac{1}{ax&plus;1}" title="h(x)=\frac{1}{ax+1}" />

The fitness function is normalized to the (0, 1] where:
* the worse is the individual the closer to zero its fitness value is;
* fitness value for an ideal chromosome, which results is exactly the same state vector sequence as the
input data, is equal to one.

### RCGA parameters

* Recombination method: Signle-point corssover
* Mutation method: randomly chosen from random mutation, non-uniform mutation, and Muhlenbein's mutation
* Selection method: randomly chosen from roulette wheel and tournament
* Probability of mutation: 0.5
* Pop_size: 100 chromosomes 
* max_generation: 300,000
* max fitness: 0.999
* fitness function: L2 norm.

