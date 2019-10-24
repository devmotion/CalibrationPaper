#' ---
#' title: Calibration error estimates and p-value approximations for modern neural networks
#' author: David Widmann
#' ---

#' # Intro
#'
#' In the following experiments we download a set of pretrained modern neural networks for
#' the image data set CIFAR-10. We estimate the expected calibration error (ECE) with
#' respect to the total variation distance and the squared kernel calibration error of
#' these models.

#' # Packages
#'
#' We perform distributed computing to speed up our computations.

using Distributed

#' First we have to activate the local package environment on all cores.

@everywhere begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

#' Then we load the packages that are required on all cores.

@everywhere begin
    using CalibrationErrors
    using CalibrationPaper
    using CalibrationTests

    using DelimitedFiles
    using Random
end

#' The following packages are only required on the main process.

using Conda
using CSV
using DataDeps
using DataFrames
using PyCall

using LibGit2

#' # Experiments
#'
#' ## Pretrained neural networks
#'
#' As a first step we download a set of pretrained neural networks for CIFAR-10 from
#' [PyTorch-CIFAR10](https://github.com/huyvnphan/PyTorch-CIFAR10). We extract the
#' predictions of these models on the validation data set together with the correct labels.
#'
#' We start by registering the required data. If you want to rerun this experiment from
#' scratch, please download the pretrained weights of the models.

register(ManualDataDep(
    "PyTorch-CIFAR10",
    """
    Please go to
        https://drive.google.com/drive/folders/15jBlLkOFg0eK-pwsmXoSesNDyDb_HOeV
    and download the pretrained weights.
    Note that this must be done manually since Google requires to confirm the download of
    large files and I have not automated the confirmation process (yet).
    """
))

#' First we install PyTorch.

Conda.add("cpuonly=1.0"; channel = "pytorch")
Conda.add("pytorch=1.3.0"; channel = "pytorch")
Conda.add("torchvision=0.4.1"; channel = "pytorch")

mktempdir() do dir
    # create directory for results
    datadir = joinpath(@__DIR__, "..", "data", "PyTorch-CIFAR10")
    isdir(datadir) || mkpath(datadir)

    # check if predictions exist
    modelnames = ["densenet121", "densenet161", "densenet169", "googlenet", "inception_v3",
                  "mobilenet_v2", "resnet_orig", "resnet18", "resnet34", "resnet50",
                  "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]
    let datadir = datadir
        filter!(modelnames) do name
            !isfile(joinpath(datadir, "$name.csv"))
        end
    end

    # exit if predictions and labels exist
    isempty(modelnames) && isfile(joinpath(datadir, "labels.csv")) && return

    # clone repository
    repodir = mkdir(joinpath(dir, "PyTorch-CIFAR10"))
    LibGit2.clone("https://github.com/huyvnphan/PyTorch-CIFAR10.git", repodir)
    LibGit2.checkout!(LibGit2.GitRepo(repodir), "90325333f4da099b3a795693cfa18e64490dffe9")

    # copy pretrained weights to the correct directory
    weightsdir = joinpath(repodir, "models", "state_dicts")
    for name in modelnames
        cp(joinpath(datadep"PyTorch-CIFAR10", "$name.pt"), joinpath(weightsdir, "$name.pt"))
    end

    # load Python packages
    torch = pyimport("torch")
    F = pyimport("torch.nn.functional")
    torchvision = pyimport("torchvision")
    transforms = pyimport("torchvision.transforms")

    # import local models
    pushfirst!(PyVector(pyimport("sys")."path"), repodir)
    models = pyimport("models")

    # define transformation
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                         [0.2023, 0.1994, 0.2010])])

    # download CIFAR-10 validation data set
    dataset = torchvision.datasets.CIFAR10(root=joinpath(dir, "CIFAR10"), train=false,
                                           transform=transform, download=true)

    # extract and save labels
    if !isfile(joinpath(datadir, "labels.csv"))
        @info "extracting labels..."

        # extract labels of the validation data set
        _, labels_py = iterate(torch.utils.data.DataLoader(dataset, batch_size=10_000, shuffle=false))[1]

        # save labels (+1 since we need classes 1,...,n)
        labels = pycall(labels_py."numpy", PyArray) .+ 1
        writedlm(joinpath(datadir, "labels.csv"), labels)
    end

    # extract predictions
    if !isempty(modelnames)
        # create data loader with batches of 250 images
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=250, shuffle=false)

        cd(repodir) do
            @pywith torch.no_grad() begin
                for name in modelnames
                    @info "extracting predictions of model $name..."

                    # load model with pretrained weights
                    model = getproperty(models, name)(pretrained=true)

                    # save all predictions
                    open(joinpath(datadir, "$name.csv"), "w") do f
                        for (images_py, _) in dataloader
                            predictions = pycall(F.softmax(model(images_py), dim=1)."numpy", PyArray)
                            writedlm(f, predictions, ',')
                        end
                    end
                end
            end
        end
    end

    nothing
end

#' ## Calibration error estimates
#'
#' For all pretrained neural networks we compute a set of different calibration error
#' estimates and save them in a CSV file `errors.csv`. More concretely, we evaluate the
#' expected calibration error estimators with 10 uniform bins per dimension and with data
#' dependent bins, and the biased and the unbiased quadratic estimator of the squared
#' kernel calibration error as well as the unbiased linear estimator for a uniformly scaled
#' exponential kernel for which the bandwidth is set with the median heuristic.

function calibration_errors()
    # do not recompute the calibration errors if a file with results exists
    datadir = joinpath(@__DIR__, "..", "data", "PyTorch-CIFAR10")
    file = joinpath(datadir, "errors.csv")
    isfile(file) && return

    # load labels
    labels = vec(readdlm(joinpath(datadir, "labels.csv"), ',', Int))

    # define the studied models
    models = ["densenet121", "densenet161", "densenet169", "googlenet", "inception_v3",
              "mobilenet_v2", "resnet_orig", "resnet18", "resnet34", "resnet50",
              "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]

    # define arrays for the estimates
    n = length(models)
    ece_uniform = Vector{Float64}(undef, n)
    ece_dynamic = Vector{Float64}(undef, n)
    skceb_median = Vector{Float64}(undef, n)
    skceuq_median = Vector{Float64}(undef, n)
    skceul_median = Vector{Float64}(undef, n)

    # analyze every model
    @inbounds for i in 1:n
        model = models[i]
        @info "evaluating calibration error estimators for $model..."

        # load predictions
        predictions = copy(readdlm(joinpath(datadir, "$model.csv"), ',')')
        data = (predictions, labels)

        # evaluate ECE estimators
        ece_uniform[i] = calibrationerror(ECE(UniformBinning(10)), data)
        ece_dynamic[i] = calibrationerror(ECE(MedianVarianceBinning(10)), data)

        # compute kernel based on the median heuristic
        kernel = median_TV_kernel(predictions)

        # evaluate SKCE estimators
        skceb_median[i] = calibrationerror(BiasedSKCE(kernel), data)
        skceuq_median[i] = calibrationerror(QuadraticUnbiasedSKCE(kernel), data)
        skceul_median[i] = calibrationerror(LinearUnbiasedSKCE(kernel), data)
    end

    # save results
    df = DataFrame(model = models, ECE_uniform = ece_uniform, ECE_dynamic = ece_dynamic,
                   SKCEb_median = skceb_median, SKCEuq_median = skceuq_median,
                   SKCEul_median = skceul_median)
    CSV.write(file, df)

    nothing
end

calibration_errors()

#' ## Calibration tests
#'
#' Additionally we compute different p-value approximations for each model. More concretely,
#' we estimate the p-value by consistency resampling of the two ECE estimators mentioned
#' above, by distribution-free bounds of the three SKCE estimators discussed above, and by
#' the asymptotic approximations for the unbiased quadratic and linear SKCE estimators
#' used above. The results are saved in a CSV file `pvalues.csv`.

@everywhere function calibration_pvalues(rng::AbstractRNG, predictions, labels)
    data = (predictions, labels)

    # evaluate consistency resampling based estimators
    Random.seed!(rng, 1234)
    ece_uniform =
        pvalue(ConsistencyTest(ECE(UniformBinning(10)), data; rng = rng))
    Random.seed!(rng, 1234)
    ece_dynamic =
        pvalue(ConsistencyTest(ECE(MedianVarianceBinning(100)), data; rng = rng))

    # compute kernel based on the median heuristic
    kernel = median_TV_kernel(predictions)

    # evaluate distribution-free bounds
    skceb_median_distribution_free =
        pvalue(DistributionFreeTest(BiasedSKCE(kernel), data))
    skceuq_median_distribution_free =
        pvalue(DistributionFreeTest(QuadraticUnbiasedSKCE(kernel), data))
    skceul_median_distribution_free =
        pvalue(DistributionFreeTest(LinearUnbiasedSKCE(kernel), data))

    # evaluate asymptotic bounds
    Random.seed!(rng, 1234)
    skceuq_median_asymptotic =
        pvalue(AsymptoticQuadraticTest(kernel, data; rng = rng))
    skceul_median_asymptotic = pvalue(AsymptoticLinearTest(kernel, data))

    ece_uniform, ece_dynamic, skceb_median_distribution_free,
    skceuq_median_distribution_free, skceul_median_distribution_free,
    skceuq_median_asymptotic, skceul_median_asymptotic
end

function calibration_pvalues()
    # do not recompute the p-values if a file with results exists
    datadir = joinpath(@__DIR__, "..", "data", "PyTorch-CIFAR10")
    file = joinpath(datadir, "pvalues.csv")
    isfile(file) && return

    # load labels
    labels = vec(readdlm(joinpath(datadir, "labels.csv"), Int))

    # define the studied models
    models = ["densenet121", "densenet161", "densenet169", "googlenet", "inception_v3",
              "mobilenet_v2", "resnet_orig", "resnet18", "resnet34", "resnet50",
              "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]

    # analyze every model
    wp = CachingPool(workers())
    estimates = let rng = Random.GLOBAL_RNG, datadir = datadir, labels = labels
        pmap(wp, models) do model
            # load predictions
            predictions = copy(readdlm(joinpath(datadir, "$model.csv"), ',')')

            @info "evaluating p-value approximations for $model..."
            estimates = calibration_pvalues(deepcopy(rng), predictions, labels)

            (model, estimates...)
        end
    end

    # save results
    df = DataFrame(map(idx -> getindex.(estimates, idx), eachindex(first(estimates))),
                   [:model, :ECE_uniform, :ECE_dynamic, :SKCEb_median_distribution_free,
                    :SKCEuq_median_distribution_free, :SKCEul_median_distribution_free,
                    :SKCEuq_median_asymptotic, :SKCEul_median_asymptotic])
    CSV.write(file, df)

    nothing
end

calibration_pvalues()
