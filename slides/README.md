# Slides

This folder contains the source code of the slides summarizing the paper
"Calibration tests in multi-class classification: A unifying framework"
by Widmann, Lindsten, and Zachariah.

## Generate the figure

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
You can regenerate the figure by running
``` julia
julia> errors_ece()
```

## Compile the slides

The slides can be compiled as a PDF file by running
```shell
arara spotlight.tex
```
