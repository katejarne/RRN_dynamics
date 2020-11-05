# RRN_dynamics
Analysis of RNN  (paper under review, preprint https://arxiv.org/abs/2005.13074):

Simulations were generated using python 3.6.9 Tensorflow version 2.0 and Keras 2.3.1 Following the procedure described previously in https://arxiv.org/abs/1906.01094

The trained networks are saved in hdf5 format and can be opened using.

generate_figures.py 

to generate all the figures corresponding to one particular realization using the corresponding testing set generator. 

For example to test the "and" task, use generate_data_set_and.py




# ID of Supplementary Table 1 from paper corresponds to the examples in the corresponding directory of this repo.
For example #_xor_10_ran has the figures corresponding to the Network 10 trained for Xor task with random normal initial condition.
