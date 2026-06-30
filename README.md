# Monkeypox-Model-Codes
MATLAB codes for the stochastic Mpox transmission model and neural network simulations presented in the manuscript.
# Investigating the Deterministic and Stochastic Dynamics of Monkeypox: A Hybrid Numerical  Approach Using Spectral Method and Feedforward Neural Networks

Authors: Feroz Khan, Coauthors

This repository contains the MATLAB used in the manuscript:

"Investigating the Deterministic and Stochastic Dynamics of Monkeypox: A Hybrid Numerical  Approach Using Spectral Method and Feedforward Neural Networks"

Contents:
- MATLAB codes for deterministic and stochastic simulations
- Matlab codes for neural network predictions
- Scripts for reproducing the figures in the manuscript

Software requirements:
- MATLAB R2015
 
- Instructions:
1. Run lscm_mpox.m for deterministic simulations with sigma_i=0.
2. Run lscm_mpox.m for stochastic simulations when sigma_i is non zero .
3. Run lscm_beta1_Ih.m to see beta1 effect on Mpox dynamics for both deterministic and stochastic (similarly for other parameters).
4. Run sensitivity.m for sensitivity of parameters.
4. lglnodes.m file must be saved in the same folder as a support file.
