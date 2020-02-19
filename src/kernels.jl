median_TV_distance(predictions) = median(_pairwise(TotalVariation(), predictions))

function mean_TV_distance(m, αᵢ)
    α₀ = m * αᵢ
    r = (m - 1) * αᵢ

    2 * inv(αᵢ) * exp(logabsbeta(α₀, α₀)[1] - logabsbeta(αᵢ, αᵢ)[1] - logabsbeta(r, r)[1])
end

function median_TV_kernel(predictions)
    # compute median TV distance
    γ = inv(median_TV_distance(predictions))

    # create kernel
    UniformScalingKernel(ExponentialKernel(γ, TotalVariation()))
end

function mean_TV_kernel(m, αᵢ)
    # compute mean TV distance
    γ = inv(mean_TV_distance(m, αᵢ))

    # create kernel
    UniformScalingKernel(ExponentialKernel(γ, TotalVariation()))
end

mean_TV_kernel(model::CalibrationPaperModel) = mean_TV_kernel(model.m, model.αᵢ)

# pairwise evaluation for vector of vectors
function _pairwise(metric::SemiMetric, a::AbstractVector{<:AbstractVector{<:Real}})
    n = length(a)
    n > 0 || error("at least one sample required")

    T = eltype(a[1])
    r = Matrix{Distances.result_type(metric, T, T)}(undef, n, n)
    @inbounds for j in 1:n
        aj = a[j]
        for i in (j + 1):n
            r[i, j] = metric(a[i], aj)
        end
        r[j, j] = 0
        for i in 1:(j - 1)
            r[i, j] = r[j, i]   # leveraging the symmetry of SemiMetric
        end
    end
    r
end

_pairwise(metric::SemiMetric, a::AbstractMatrix{<:Real}) = pairwise(metric, a; dims = 2)