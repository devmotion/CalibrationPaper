# Poster

This folder contains the source code of the poster for the paper
"Calibration tests in multi-class classification: A unifying framework"
by Widmann, Lindsten, and Zachariah which is going to be presented
at NeurIPS 2019.

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

## Fonts

If available, the non-free font Berling Antikva is used which is part
of the official design of Uppsala University. The corresponding OTF
fonts have to be available in the subfolder `fonts` as `Berling.otf`,
`Berling-Italic.otf`, `Berling-Bold.otf`, and `Berling-BoldItalic.otf`.
Unfortunately, the official fonts that can be
[downloaded](https://mp.uu.se/documents/432512/911262/18350_Typsnitt_PC+v2.zip/24599197-7993-f965-e58d-0e375939d144)
from Uppsala University's staff portal can not be used with older
version of LuaLaTeX (the problem is explained in a
[question on StackExchange](https://tex.stackexchange.com/questions/430837/why-does-lualatex-have-problem-with-a-ttf-that-xelatex-accepts/430872#comment1161846_430837).
However, suitable OTF fonts can be generated from these files according
to the
[instructions provided on StackExchange](https://tex.stackexchange.com/a/430872).

## Compile the poster

The poster can be compiled as a PDF file by running
```shell
arara neurips.tex
```
