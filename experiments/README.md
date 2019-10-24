# Experiments

This folder contains the implementation and the results of the experiments in
the paper "Calibration tests in multi-class classification: A unifying
framework" by Widmann, Lindsten, and Zachariah, which is going to be
presented at NeurIPS 2019.

## Structure

The subfolder `scripts` contains the scripts
- `errors.jl` for evaluating calibration error estimators for different
  calibrated and uncalibrated models,
- `pvalues.jl` for evaluating bounds and approximation of the p-value of
  calibration error estimates under the null hypothesis of the model being
  calibrated for different calibrated and uncalibrated models,
- `nn.jl` for evaluating calibration error estimators and p-value approximations
  for different neural networks pre-trained on the CIFAR-10 image data set,
- `timings.jl` for benchmarking different calibration error estimations.

These scripts are written in a format of the literate programming tool
[Weave](https://github.com/JunoLab/Weave.jl). The subfolder `html` contains HTML
files that are generated from these scripts and illustrate the results that
are saved as CSV files in the subfolder `data`.

## View HTML files

You can use the HTML preview feature of Github to display the HTML version of
the experiments [`errors.jl`](http://htmlpreview.github.io/?https://github.com/devmotion/CalibrationPaper/blob/master/experiments/html/errors.html),
[`pvalues.jl`](http://htmlpreview.github.io/?https://github.com/devmotion/CalibrationPaper/blob/master/experiments/html/pvalues.html),
[`nn.jl`](http://htmlpreview.github.io/?https://github.com/devmotion/CalibrationPaper/blob/master/experiments/html/nn.html),
and [`timings.jl`](http://htmlpreview.github.io/?https://github.com/devmotion/CalibrationPaper/blob/master/experiments/html/timings.html)
online.

## Reproducibility

If you want to rerun the experiments, make sure to delete the relevant CSV files
with our results in the subfolder `data`. Open a terminal in this folder and
install the required Julia packages by running
```shell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
You can run the desired experiment `EXPERIMENT.jl` in the subfolder `scripts`
with
```shell
cd scripts
julia --project=.. EXPERIMENT.jl
```
Note that the experiments `errors.jl`, `pvalues.jl`, and `nn.jl` take multiple
hours or even days to complete, if you run them from scratch. Hence these scripts
are heavily parallelized and make use of multiple cores, if possible. It is
recommended to run them on a dedicated server and to use multi-core
processing with
```shell
julia --project=.. -p=n EXPERIMENT.jl
```
where `n` is the number of additional local worker processes. By specifying
`auto`, as many workers as the number of local CPU threads (logical cores) are
launched.

The corresponding HTML files can be regenerated and updated by running
```shell
cd scripts
julia --project=.. -e 'using Weave; weave("EXPERIMENT.jl"; out_path = joinpath("..", "html"))'
```
It is recommended to perform the experiments and to make sure that the results
are saved to the subfolder `data` before generating the HTML file.
