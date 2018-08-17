# nn-dependability-kit

nn-dependability kit is a research prototype to assist engineering neural networks for safety-critical domains. 

## License

GNU Affero General Public License (AGPL) Version 3

## Trying the tool

Examples are presented as jupyter notebooks to allow step-by-step understanding. 

- [Formal verification] TargetVehicleProcessingNetwork_FormalVerification.ipynb
- [Test case generation] GTSRB_Pytorch_Neuron2ProjectionCoverage_TestGen.ipynb  

## Structure

There are four packages under nndependability, namely
- metrics: compute dependability metrics for neural networks
- atg: automatic test case generation to improve the metrics
- formal: formal verification (static analysis, constraint solving) of neural networks
- rv: runtime monitoring of neural networks

## Important python packages as requirements

- PyTorch 4.0 + Numpy + matplotlib + jupyter
- [Test case generation] Google optimization research tools (https://developers.google.com/optimization/introduction/installing/binary)
- [Verification / static analysis] pulp (python-based MILP connector to CBC and other solvers)
- [Run-time monitoring] dd (binary decision diagram implemented using python)

# Related publications

- [Metrics & test case generation] https://arxiv.org/abs/1806.02338
- [Static analysis & formal verification] https://arxiv.org/abs/1705.01040 
- [Runtime verification] 

