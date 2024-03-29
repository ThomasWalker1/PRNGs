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

Outlined in this repository are functions to replicate observations obtained in R and C++ in Python. Given that we know the integer seed (let suppose it is $1$ for the examples below) provided in either of these languages (R or C++) we can run the complementary functions, found within this repository, in Python in order to replicate the observations. 
```
from Align_PRNG_Python_R import r_state_from_seed, r_random_uniform

print(r_random_uniform(state=r_state_from_seed(1), size=5))

from Align_PRNG_Python_Cplusplus import cplusplus_state_from_seed, cplusplus_generate_canonical

print(cplusplus_generate_canonical(state=cplusplus_state_from_seed(1), size=5))
```
