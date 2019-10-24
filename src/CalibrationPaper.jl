module CalibrationPaper

using CalibrationErrors
using Distances
using Parameters
using Query
using SpecialFunctions
using StatsFuns

using Statistics

export CalibrationPaperModel
export median_TV_kernel, mean_TV_kernel
export collect_estimates

include("model.jl")
include("kernels.jl")

end # module
