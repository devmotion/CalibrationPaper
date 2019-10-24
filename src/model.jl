struct CalibrationPaperModel{A<:Real,P<:Real}
    m::Int
    αᵢ::A
    π::P
    only_firstclass::Bool

    function CalibrationPaperModel{A,P}(m::Int, αᵢ::A, π::P,
                                        only_firstclass::Bool) where {A,P}
        # check arguments
        m > 0 || throw(ArgumentError("number of classes must be positive"))
        αᵢ > zero(αᵢ) ||
            throw(ArgumentError("parameter αᵢ must be positive"))
        zero(π) ≤ π ≤ one(π) ||
            throw(ArgumentError("probability π must be between 0 and 1"))

        new{A,P}(m, αᵢ, π, only_firstclass)
    end
end

"""
    CalibrationPaperModel(m::Int, αᵢ::Real, π::Real, only_firstclass::Bool)

Create a generative model
```math
\\begin{aligned}
    g(X) &\\sim \\textrm{Dir}(\\alpha),\\\\
    Z &\\sim \\textrm{Ber}(\\pi),\\\\
    Y \\,|\\, g(X) = \\gamma, Z = 1 &\\sim \\textrm{Categorical}(\\beta),\\\\
    Y \\,|\\, g(X) = \\gamma, Z = 0 &\\sim \\textrm{Categorical}(\\gamma),
\\end{aligned}
```
where ``\\alpha = (\\alpha_i, \\ldots, \\alpha_i) \\in \\mathbb{R}_{>0}^m`` determines the
distribution of predictions ``g(X)``, ``\\pi > 0`` determines the degree of miscalibration,
and ``\\beta`` defines a fixed categorical distribution with ``\\beta = (1, 0, \\ldots, 0)``
if `only_firstclass = true`, and ``\\beta = (1/m, \\ldots, 1/m)`` otherwise.
"""
CalibrationPaperModel(m::Int, αᵢ::Real, π::Real, only_firstclass::Bool) =
    CalibrationPaperModel{typeof(αᵢ),typeof(π)}(m, αᵢ, π, only_firstclass)

function analytic_ece(model::CalibrationPaperModel)
    @unpack m = model

    if model.only_firstclass
        model.π * (m - 1) * inv(m)
    else
        αᵢ = model.αᵢ
        model.π * inv(αᵢ) * exp(αᵢ * (xlogx(m - 1) - xlogx(m)) - lbeta(αᵢ, (m - 1) * αᵢ))
    end
end


function collect_estimates(data, model::CalibrationPaperModel)
    @from i in data begin
        @where i.m == model.m && i.αᵢ == model.αᵢ && i.π == model.π && i.only_firstclass == model.only_firstclass
        @select i.estimate
        @collect
    end
end
