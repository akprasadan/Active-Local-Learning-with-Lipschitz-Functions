# Active-Local-Learning-with-Lipschitz-Functions
This repository implements the algorithm in [[1]](#1).

Table of Contents:

- linearprogram.py computes an empirical risk minimizing estimate of Lipschitz functions with respect to the l1 norm. 
    - Fitting method is complete, but still need interpolating functionality
- generate_data.py creates a sample of x and y datapoints (the unlabeled points in theory).
    - Mostly complete except a greater range of distributions is needed
- partition.py creates the partition into long and short intervals (Complete)
- partitiondata.py stores the partition location and type for each of the datapoints from generate_data.py using the output of partition.py (Complete)
- main.py: Perform the actual querying (incomplete) at a test point

## References
<a id="1">[1]</a> 
Backurs, A., Blum, A., & Gupta, N. (2020, July).
Active local learning. *In Conference on Learning Theory* (pp. 363-390). PMLR.