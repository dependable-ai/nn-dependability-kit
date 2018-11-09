# nn-dependability-kit

nn-dependability kit is a research prototype to assist engineering neural networks for safety-critical domains. 

## License

GNU Affero General Public License (AGPL) Version 3

## Trying the tool

Examples are presented as jupyter notebooks to allow step-by-step understanding over the concepts. 

- [Formal verification] TargetVehicleProcessingNetwork_FormalVerification.ipynb
- [Metrics & test case generation] GTSRB_Neuron2ProjectionCoverage_TestGen.ipynb, or MNIST_Neuron2ProjectionCoverage_TestGen.ipynb, or SSD_InterpretationPrecision.ipynb
- [Runtime verification] GTSRB_RuntimeMonitoring.ipynb, or MNIST_RuntimeMonitoring.ipynb  

## Structure

There are four packages under nndependability, namely
- metrics: compute dependability metrics for neural networks
- atg: automatic test case generation to improve the metrics
- formal: formal verification (static analysis, constraint solving) of neural networks
- rv: runtime monitoring of neural networks

## Important python packages as requirements

- PyTorch 0.4.x + Numpy + matplotlib + jupyter
- [Test case generation] Google optimization research tools (https://developers.google.com/optimization/introduction/installing/binary)
- [Verification / static analysis] PuLP (python-based MILP connector to CBC and other solvers)
    - [For Ubuntu users] CBC solver pre-shipped with PuLP may crash in solving some problems. Therefore, please additionally install [GNU GLPK](http://www.gnu.org/software/glpk/). The static analysis engine assumes that the GLPK solver is installed in the default directory "/usr/local/bin/glpsol". Whenever CBC crashes, GLPK is automatically triggered as a replacement.
- [Run-time verification] dd (binary decision diagram implemented using python)

## Related publications

- [Metrics & test case generation] https://arxiv.org/abs/1806.02338
- [Static analysis & formal verification] https://arxiv.org/abs/1705.01040 
- [Runtime verification] https://arxiv.org/abs/1809.06573
- [SSD-example] The example uses a few pictures from the VOC2012 dataset: The PASCAL Visual Object Classes (VOC) Challenge
Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A.
International Journal of Computer Vision, 88(2), 303-338, 2010

## Metrics related to intrepretation precision and occlusion sensitivity

### 1. Additional package to be installed
- [Metrics] saliency (https://github.com/PAIR-code/saliency)
  Use it in the following way:
```
# init submodule for saliency
cd nndependability/metrics/saliency-source/
git submodule init
git submodule update
cd ..
ln -s saliency-source/saliency saliency
cd ../../
```

### 2. Preparation for SSD example
```
cd models/SSD-Tensorflow/
git submodule init
git submodule update

# prepare weights
cd checkpoints/
unzip ssd_300_vgg.ckpt.zip
cd ../

# install custom changes to module SSD-Tensorflow that allows using saliency
git apply ../ssd_tensorflow_diff.diff
cd ../../
```
