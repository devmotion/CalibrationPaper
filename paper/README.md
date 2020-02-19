# Paper

This folder contains the LaTeX source code of the paper "Calibration tests in
multi-class classification: A unifying framework" by Widmann, Lindsten, and
Zachariah.

## Generate the figures

Open a terminal in the current directory and install all required Julia packages
by running
```shell
julia --project=. -e "using Pkg; Pkg.instantiate()"
```
Afterwards start a Julia REPL
```shell
julia --project=.
```
and include the file `figures.jl` with

``` julia
julia> include("figures.jl")
```

You can regenerate the figures in the paper with the following commands:
| Figure | Command |
| ------ | ------- |
| 1 | `errors_comparison()` |
| 2 | `pvalues_comparison()` |
| 3-34 | `errors()` |
| 35-82| `pvalues()` |
| 83 | `cifar10_errors_comparison()` |
| 84 | `cifar10_pvalues_comparison()` |
| 85 | `timings()` |

## Compile the paper

The paper can be compiled as a PDF file by running
```shell
arara main.tex
```
