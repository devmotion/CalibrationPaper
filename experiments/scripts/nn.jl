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
using ProgressMeter
using PyCall

using LibGit2

#' # Experiments
#'
#' ## Pretrained neural networks
#'
#' As a first step we download a set of pretrained neural networks for CIFAR-10 from
#' [PyTorch-CIFAR10](https://github.com/huyvnphan/PyTorch-CIFAR10). We extract the
#' predictions of these models on the validation data set together with the correct labels.
#' First we check if the predictions and labels are already extracted.

# create directory for results
const DATADIR = joinpath(@__DIR__, "..", "data", "PyTorch-CIFAR10")
isdir(DATADIR) || mkpath(DATADIR)

# check if predictions exist
const ALL_MODELS = ["densenet121", "densenet161", "densenet169", "googlenet", "inception_v3",
                    "mobilenet_v2", "resnet_orig", "resnet18", "resnet34", "resnet50",
                    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]
const MISSING_MODELS = filter(ALL_MODELS) do name
    !isfile(joinpath(DATADIR, "$name.csv"))
end

#' If the data does not exist, we start by loading all missing packages and
#' registering the required data. If you want to rerun this experiment from
#' scratch, please download the pretrained weights of the models.

if !isempty(MISSING_MODELS) || !isfile(joinpath(DATADIR, "labels.csv"))   
    # register the data source for the pretrained models
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

    # install PyTorch
    Conda.add("cpuonly=1.0"; channel = "pytorch")
    Conda.add("pytorch=1.3.0"; channel = "pytorch")
    Conda.add("torchvision=0.4.1"; channel = "pytorch")

    mktempdir() do dir
        # clone the repository
        repodir = mkdir(joinpath(dir, "PyTorch-CIFAR10"))
        LibGit2.clone("https://github.com/huyvnphan/PyTorch-CIFAR10.git", repodir)
        LibGit2.checkout!(LibGit2.GitRepo(repodir), "90325333f4da099b3a795693cfa18e64490dffe9")

        # copy pretrained weights to the correct directory
        weightsdir = joinpath(repodir, "models", "state_dicts")
        for name in MISSING_MODELS
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
        if !isfile(joinpath(DATADIR, "labels.csv"))
            @info "extracting labels..."

            # extract labels of the validation data set
            _, labels_py = iterate(torch.utils.data.DataLoader(dataset, batch_size=10_000, shuffle=false))[1]

            # save labels (+1 since we need classes 1,...,n)
            labels = pycall(labels_py."numpy", PyArray) .+ 1
            writedlm(joinpath(datadir, "labels.csv"), labels)
        end

        # extract predictions
        if !isempty(MISSING_MODELS)
            # create data loader with batches of 250 images
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=250, shuffle=false)

            cd(repodir) do
                @pywith torch.no_grad() begin
                    for name in MISSING_MODELS
                        @info "extracting predictions of model $name..."

                        # load model with pretrained weights
                        model = getproperty(models, name)(pretrained=true)

                        # save all predictions
                        open(joinpath(DATADIR, "$name.csv"), "w") do f
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
end

#' We load the true labels since they are the same for every model.

const LABELS = CSV.read(joinpath(DATADIR, "labels.csv");
    header = false, delim = ',', type = Int) |> Matrix{Int} |> vec

#' ## Calibration error estimates
#'
#' For all pretrained neural networks we compute a set of different calibration error
#' estimates and save them in a CSV file `errors.csv`. More concretely, we evaluate the
#' expected calibration error estimators with 10 uniform bins per dimension and with data
#' dependent bins, and the biased and the unbiased quadratic estimator of the squared
#' kernel calibration error as well as the unbiased linear estimator for a uniformly scaled
#' exponential kernel for which the bandwidth is set with the median heuristic.

@everywhere function calibration_errors(rng::AbstractRNG, predictions, labels, channel)
    # evaluate ECE estimators
    ece_uniform = calibrationerror(ECE(UniformBinning(10)), predictions, labels)
    put!(channel, true)
    ece_dynamic = calibrationerror(ECE(MedianVarianceBinning(100)), predictions, labels)
    put!(channel, true)

    # compute kernel based on the median heuristic
    kernel = median_TV_kernel(predictions)

    # evaluate SKCE estimators
    skceb_median = calibrationerror(BiasedSKCE(kernel), predictions, labels)
    put!(channel, true)
    skceuq_median = calibrationerror(QuadraticUnbiasedSKCE(kernel), predictions, labels)
    put!(channel, true)
    skceul_median = calibrationerror(LinearUnbiasedSKCE(kernel), predictions, labels)
    put!(channel, true)

    (
        ECE_uniform = ece_uniform,
        ECE_dynamic = ece_dynamic,
        SKCEb_median = skceb_median,
        SKCEuq_median = skceuq_median,
        SKCEul_median = skceul_median
    )
end

# do not recompute the calibration errors if a file with results exists
if isfile(joinpath(DATADIR, "errors.csv"))
    @info "skipping calibration error estimation: output file $(joinpath(DATADIR, "errors.csv")) exists"
else
    # define the pool of workers, the progress bar, and its update channel
    wp = CachingPool(workers())
    n = length(ALL_MODELS)
    p = Progress(5 * n, 1, "computing calibration error estimates...")
    channel = RemoteChannel(() -> Channel{Bool}(5 * n))

    local estimates
    @sync begin
        # update the progress bar
        @async while take!(channel)
            next!(p)
        end

        # compute the p-value approximations for all models
        estimates = let rng = Random.GLOBAL_RNG, datadir = DATADIR, labels = LABELS, channel = channel
            pmap(wp, ALL_MODELS) do model
                # load predictions
                rawdata = CSV.read(joinpath(datadir, "$model.csv");
                                   header = false, transpose = true, delim = ',',
                                   type = Float64) |> Matrix{Float64}
                predictions = [rawdata[:, i] for i in axes(rawdata, 2)]

                # copy random number generator and set seed
                _rng = deepcopy(rng)
                Random.seed!(_rng, 1234)

                # compute approximations
                errors = calibration_errors(_rng, predictions, labels, channel)
                merge((model = model,), errors)
            end
        end

        # stop progress bar
        put!(channel, false)
    end

    @info "saving calibration error estimates..."
    CSV.write(joinpath(DATADIR, "errors.csv"), estimates)
end

#' ## Calibration tests
#'
#' Additionally we compute different p-value approximations for each model. More concretely,
#' we estimate the p-value by consistency resampling of the two ECE estimators mentioned
#' above, by distribution-free bounds of the three SKCE estimators discussed above, and by
#' the asymptotic approximations for the unbiased quadratic and linear SKCE estimators
#' used above. The results are saved in a CSV file `pvalues.csv`.

@everywhere function calibration_pvalues(rng::AbstractRNG, predictions, labels, channel)
    # evaluate consistency resampling based estimators
    ece_uniform = ConsistencyTest(ECE(UniformBinning(10)), predictions, labels)
    pvalue_ece_uniform = pvalue(ece_uniform; rng = rng)
    put!(channel, true)
    ece_dynamic = ConsistencyTest(ECE(MedianVarianceBinning(100)), predictions, labels)
    pvalue_ece_dynamic = pvalue(ece_dynamic; rng = rng)
    put!(channel, true)

    # compute kernel based on the median heuristic
    kernel = median_TV_kernel(predictions)

    # evaluate distribution-free bounds
    skceb_median_distribution_free = DistributionFreeTest(BiasedSKCE(kernel), predictions, labels)
    pvalue_skceb_median_distribution_free = pvalue(skceb_median_distribution_free)
    put!(channel, true)
    skceuq_median_distribution_free = DistributionFreeTest(QuadraticUnbiasedSKCE(kernel), predictions, labels)
    pvalue_skceuq_median_distribution_free = pvalue(skceuq_median_distribution_free)
    put!(channel, true)
    skceul_median_distribution_free = DistributionFreeTest(LinearUnbiasedSKCE(kernel), predictions, labels)
    pvalue_skceul_median_distribution_free = pvalue(skceul_median_distribution_free)
    put!(channel, true)

    # evaluate asymptotic bounds
    skceuq_median_asymptotic = AsymptoticQuadraticTest(kernel, predictions, labels)
    pvalue_skceuq_median_asymptotic = pvalue(skceuq_median_asymptotic; rng = rng)
    put!(channel, true)
    skceul_median_asymptotic = AsymptoticLinearTest(kernel, predictions, labels)
    pvalue_skceul_median_asymptotic = pvalue(skceul_median_asymptotic)
    put!(channel, true)

    (
        ECE_uniform = pvalue_ece_uniform,
        ECE_dynamic = pvalue_ece_dynamic,
        SKCEb_median_distribution_free = pvalue_skceb_median_distribution_free,
        SKCEuq_median_distribution_free = pvalue_skceuq_median_distribution_free,
        SKCEul_median_distribution_free = pvalue_skceul_median_distribution_free,
        SKCEuq_median_asymptotic = pvalue_skceuq_median_asymptotic,
        SKCEul_median_asymptotic = pvalue_skceul_median_asymptotic
    )
end

# do not recompute the p-values if a file with results exists
if isfile(joinpath(DATADIR, "pvalues.csv"))
    @info "skipping p-value approximations: output file $(joinpath(DATADIR, "pvalues.csv")) exists"
else
    # define the pool of workers, the progress bar, and its update channel
    wp = CachingPool(workers())
    n = length(ALL_MODELS)
    p = Progress(7 * n, 1, "computing p-value approximations...")
    channel = RemoteChannel(() -> Channel{Bool}(7 * n))

    local estimates
    @sync begin
        # update the progress bar
        @async while take!(channel)
            next!(p)
        end

        # compute the p-value approximations for all models
        estimates = let rng = Random.GLOBAL_RNG, datadir = DATADIR, labels = LABELS, channel = channel
            pmap(wp, ALL_MODELS) do model
                # load predictions
                rawdata = CSV.read(joinpath(datadir, "$model.csv");
                                   header = false, transpose = true, delim = ',',
                                   type = Float64) |> Matrix{Float64}
                predictions = [rawdata[:, i] for i in axes(rawdata, 2)]

                # copy random number generator and set seed
                _rng = deepcopy(rng)
                Random.seed!(_rng, 1234)

                # compute approximations
                pvalues = calibration_pvalues(_rng, predictions, labels, channel)
                merge((model = model,), pvalues)
            end
        end

        # stop progress bar
        put!(channel, false)
    end

    # save estimates
    @info "saving p-value approximations..."
    CSV.write(joinpath(DATADIR, "pvalues.csv"), estimates)
end