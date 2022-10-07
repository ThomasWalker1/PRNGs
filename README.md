# PRNGs

As part of an Undergraduate Research Project I investigated pseudo random number generators (PRNG) in Python, R and C++. The aim of the investigation was to understand how the languages go about generating random numbers, how they then go onto generate samples from a uniform distribution, with the intention at the end to provide frameworks to allows to align the generation of uniform variates across the languages. 

For example, if we are working in Python and R we note that even if we set the seed using the same integer when we go to generate uniform observations we get different outputs.

```
import numpy.random as rng
from rpy2 import robjects

rng.seed(1)
print(rng.uniform(size=5))

robjects.r('''
set.seed(1)
print(runif(5))
''')
```

Similarly, if we run the following code in C++ we obtain discrepencies in the samples observed, despite setting the seed to be $1$ in each of the languages.

```
#include <iostream>
#include <random>

int main()
{
	std::mt19937 gen(1);
	for (int n = 0; n < 5; n++) {
		std::cout << std::generate_canonical<double, std::numeric_limits<double>::digits>(gen)<< '\n';;
	}
}
```

Outlined in this repository are functions to replicate observations obtained in R and C++ in Python. Given that we know the integer seed (let suppose it is $1$ for the examples below) provided in either of these languages (R or C++) we can run the complmentary functions, found within this repository, in Python in order to replicate the observations. 
```
from Align_PRNG_Python_R import r_state_from_seed, r_random_uniform

print(r_random_uniform(state=r_state_from_seed(1), size=5))

from Align_PRNG_Python_Cplusplus import cplusplus_state_from_seed, cplusplus_generate_canonical

print(cplusplus_generate_canonical(state=cplusplus_state_from_seed(1), size=5))
```

Navigation:
- [Click here](https://github.com/ThomasWalker1/PRNGs/blob/main/Reports/Pseudo%20Random%20Number%20Generators%20in%20Python%2C%20R%20and%20C%2B%2B%20With%20Applications%20to%20Generating%20Uniform%20Variates.pdf) to learn more about the languages generate random integers and generate uniform observations from these
- [Click here](https://github.com/ThomasWalker1/PRNGs/blob/main/Reports/Generating%20Normal%20and%20Exponential%20Variates%20in%20Python%2C%20R%20and%20C%2B%2B.pdf) to learn more about how the languages generate observations from other distributions
- [Click here](https://github.com/ThomasWalker1/PRNGs/blob/main/Reports/Randomly%20Sampling%20and%20Shuffling%201-D%20lists%20in%20Python%20and%20R%20Report.pdf) to learn more about how the languages randomly shuflle and sample from 1-D lists
- [Click here](https://github.com/ThomasWalker1/PRNGs/blob/main/Code/Align_PRNG_Python_R.py) to find functions for replicating random observations from R in Python
- [Click here](https://github.com/ThomasWalker1/PRNGs/blob/main/Code/Align_PRNG_Python_Cplusplus.py) to find functions for replicating random observations from C++ in Python

There is a slight exception in regard to generating normal observations in python, read more about that [here](https://github.com/ThomasWalker1/PRNGs/blob/main/Reports/Generating%20Normal%20Variates%20in%20Python.pdf)
