# CalibrationPaper

This repository accompanies the paper ["Calibration tests in multi-class
classification: A unifying framework" by Widmann, Lindsten, and Zachariah](http://papers.nips.cc/paper/9392-calibration-tests-in-multi-class-classification-a-unifying-framework),
which was presented at NeurIPS 2019.

2021-05-04: [**We extended the calibration errors and tests to general probabilistic predictive models in our paper "Calibration tests beyond classification" presented at ICLR 2021**](https://openreview.net/forum?id=-bxf89v3Nx)

## Structure

The folder `paper` contains the LaTeX source code of the paper.

The folder `experiments` contains the source code and the results of our
experiments.

The folder `src` contains common implementations such as the definition of the
generative models, which are used for generating the figures in our paper and
for some experiments.

## Reproducibility

You can rerun our experiments and recompile our paper. Every folder contains
instructions for how to build and run the files therein.

## Software

We published software packages for the proposed calibration errors and calibration tests.

### Julia packages

- [CalibrationErrors.jl](https://github.com/devmotion/CalibrationErrors.jl) and [CalibrationErrorsDistributions.jl](https://github.com/devmotion/CalibrationErrorsDistributions.jl) for estimating calibration errors from data sets of predictions and targets, including general probabilistic predictive models.
- [CalibrationTests.jl](https://github.com/devmotion/CalibrationTests.jl) for statistical hypothesis tests of calibration.

### Python and R interface

- [pycalibration](https://github.com/devmotion/pycalibration) is a Python interface for CalibrationErrors.jl, CalibrationErrorsDistributions.jl, and CalibrationTests.jl.
- [rcalibration](https://github.com/devmotion/rcalibration) is an R interface for CalibrationErrors.jl, CalibrationErrorsDistributions.jl, and CalibrationTests.jl.
