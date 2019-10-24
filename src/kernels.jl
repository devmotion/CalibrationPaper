median_TV_distance(predictions::AbstractMatrix{<:Real}) =
    median(pairwise(TotalVariation(), predictions, dims=2))

function mean_TV_distance(m, αᵢ)
  α₀ = m * αᵢ
  r = (m - 1) * αᵢ

  2 * inv(αᵢ) * exp(lbeta(α₀, α₀) - lbeta(αᵢ, αᵢ) - lbeta(r, r))
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
