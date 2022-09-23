# PRNGs

As part of an Undergraduate Research Project I investigated pseudo random number generators (PRNG) in Python, R and C++. The aim of the investigation was to understand how the languages go about generating random numbers, how they then go onto generate samples from a uniform distribution, with the intention at the end to provide frameworks to allows to align the generation of uniform variates across the languages. 

For example, if we are working in Python and R we note that even if we set the seed using the same integer when we go to generate uniform observations we get different outputs.

```
import numpy.random as rng
```
