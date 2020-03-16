using CalibrationPaper
using CSV
using DataFrames
using PGFPlotsX
using Query
using StatsBase

using Statistics

function errors_comparison()
    # initialize group plot
    @pgf plt = GroupPlot(
    {
        group_style =
        {
            group_size = "3 by 4",
            xlabels_at = "edge bottom",
            ylabels_at = "edge left",
            horizontal_sep = raw"0.1\linewidth",
            vertical_sep = raw"0.05\linewidth",
        },
        no_markers,
        tick_label_style = { font = raw"\tiny" },
        grid = "major",
        title_style = { align = "center" },
        width = raw"0.3\linewidth",
        height = raw"0.2\linewidth",
        "every x tick scale label/.style" = {at = "{(1,0)}", anchor = "west"},
        ylabel_style = { font = raw"\small" }
    })

    # define displayed experiments and corresponding labels
    experiments = ["ECE_uniform", "SKCEb_median", "SKCEuq_median", "SKCEul_median"]
    labels = [raw"$\widehat{\ECE}$", raw"$\biasedestimator$",
              raw"$\unbiasedestimator$", raw"$\linearestimator$"]

    # define displayed models
    models = [CalibrationPaperModel(10, 0.1, 0.0, false),
              CalibrationPaperModel(10, 0.1, 0.5, true),
              CalibrationPaperModel(10, 0.1, 1.0, false)]

    # directory with all experimental results
    datadir = joinpath(@__DIR__, "..", "experiments", "data", "errors")

    # for all studied experiments and models
    for (i, experiment) in enumerate(experiments), (j, model) in enumerate(models)
        # obtain full path
        file = joinpath(datadir, "$experiment.csv")

        # load estimates
        estimates = collect_estimates(CSV.read(file), model)

        # compute histogram
        hist = fit(Histogram, estimates, closed = :left)

        # create axis object with histogram
        @pgf ax = Axis(PlotInc({ ybar_interval, fill = "Dark2-A!30!white" }, Table(hist)),
                       VLine({ solid, thick, "Dark2-B" }, mean(estimates)))

        # add titles
        if i == 1
            ax["title"] = "\$\\mathbf{M$j}\$"
        end

        # add labels
        if j == 1
            ax["ylabel"] = labels[i]
        end

        # compute true value
        if startswith(experiment, "ECE")
            analytic = CalibrationPaper.analytic_ece(model)
        else
            # take mean of quadratic unbiased estimates
            if startswith(experiment, "SKCEuq")
                analytic = mean(estimates)
            else
                file_uq = replace(file, r"(SKCEb|SKCEul)" => "SKCEuq")
                estimates_uq = collect_estimates(CSV.read(file_uq), model)
                analytic = mean(estimates_uq)
            end
        end

        # plot true value
        @pgf push!(ax, VLine({ dashed, thick, "Dark2-C" }, analytic))
        # hack so that limits are updated as well
        @pgf push!(ax, PlotInc({ draw = "none" }, Coordinates([analytic], [0])))

        # add axis to group plot
        push!(plt, ax)
    end

    # add labels
    picture = TikzPicture(plt,
        raw"\node[anchor=north] at ($(group c1r4.west |- group c1r4.outer south)!0.5!(group c3r4.east |- group c3r4.outer south)$){calibration error estimate};",
        raw"\node[anchor=south, rotate=90] at ($(group c1r1.north -| group c1r1.outer west)!0.5!(group c1r4.south -| group c1r4.outer west)$){\# runs};")

    # save histogram
    figuresdir = joinpath(@__DIR__, "figures")
    isdir(figuresdir) || mkdir(figuresdir)
    pgfsave(joinpath(figuresdir, "errors_comparison.tex"), picture; include_preamble = false)

    nothing
end

function pvalues_comparison()
    # initialize group plot
    @pgf plt = GroupPlot(
    {
        group_style =
        {
            group_size = "3 by 6",
            xlabels_at = "edge bottom",
            ylabels_at = "edge left",
            horizontal_sep = raw"0.1\linewidth",
            vertical_sep = raw"0.023\linewidth",
            xticklabels_at = "edge bottom",
        },
        no_markers,
        tick_label_style = { font = raw"\tiny" },
        grid = "major",
        title_style = { align = "center" },
        width = raw"0.3\linewidth",
        height = raw"0.17\linewidth",
        "every x tick scale label/.style" = {at = "{(1,0)}", anchor = "west"},
        ylabel_style = { font = raw"\small" },
        xmin = 0, xmax = 1, ymin=-0.1, ymax=1.1
    })

    # define displayed experiments and corresponding labels
    experiments = ["ECE_uniform", "SKCEb_median_distribution_free",
                   "SKCEuq_median_distribution_free",
                   "SKCEul_median_distribution_free",
                   "SKCEuq_median_asymptotic",
                   "SKCEul_median_asymptotic"]
    labels = [raw"$\mathbf{C}$", raw"$\mathbf{D}_{\mathrm{b}}$",
              raw"$\mathbf{D}_{\mathrm{uq}}$", raw"$\mathbf{D}_{\mathrm{l}}$",
              raw"$\mathbf{A}_{\mathrm{uq}}$",
              raw"$\mathbf{A}_{\mathrm{l}}$"]

    # define displayed models
    models = [CalibrationPaperModel(10, 0.1, 0.0, false),
              CalibrationPaperModel(10, 0.1, 0.5, true),
              CalibrationPaperModel(10, 0.1, 1.0, false)]

    # define range of significance levels
    αs = 0:0.01:1

    # directory with all experimental results
    datadir = joinpath(@__DIR__, "..", "experiments", "data", "pvalues")

    # for all studied experiments and models
    for (i, experiment) in enumerate(experiments), (j, model) in enumerate(models)
        # obtain full path
        file = joinpath(datadir, "$experiment.csv")

        # load p-value estimates
        pvalues = collect_estimates(CSV.read(file), model)

        # compute empirical CDF
        empiricalCDF = ecdf(pvalues)

        if iszero(model.π)
            # if the model is calibrated we plot the empirical estimate of
            # P[p(T) < \alpha | H_0] together with the diagonal of the unit square
            @pgf ax = Axis(PlotInc({ thick }, Table(αs, empiricalCDF.(αs))),
                           PlotInc({ dashed, thick }, Coordinates([0, 1], [0, 1])))
        else
            # otherwise we plot the empirical estimate for P[p(T) > \alpha | H_1]
            @pgf ax = Axis(PlotInc({ thick }, Table(αs, 1 .- empiricalCDF.(αs))))
        end

        # add titles
        if i == 1
            ax["title"] = "\$\\mathbf{M$j}\$"
        end

        # add labels
        if j == 1
            ax["ylabel"] = labels[i]
        end

        # add axis to group plot
        push!(plt, ax)
    end

    # add labels
    picture = TikzPicture(plt,
        raw"\node[anchor=north] at ($(group c1r6.west |- group c1r6.outer south)!0.5!(group c3r6.east |- group c3r6.outer south)$){significance level};",
        raw"\node[anchor=south, rotate=90] at ($(group c1r1.north -| group c1r1.outer west)!0.5!(group c1r6.south -| group c1r6.outer west)$){empirical test error};")

    # save histogram
    figuresdir = joinpath(@__DIR__, "figures")
    isdir(figuresdir) || mkdir(figuresdir)
    pgfsave(joinpath(figuresdir, "pvalues_comparison.tex"), picture; include_preamble = false)

    nothing
end

function errors(experiment, αᵢ, only_firstclass)
    # initialize group plot
    @pgf plt = GroupPlot(
    {
        group_style =
        {
            group_size = "5 by 4",
            xlabels_at = "edge bottom",
            ylabels_at = "edge left",
            horizontal_sep = raw"0.07\linewidth",
            vertical_sep = raw"0.05\linewidth",
        },
        no_markers,
        grid = "major",
        tick_label_style = { font = raw"\tiny" },
        width = raw"0.23\linewidth",
        height = raw"0.2\linewidth",
        xticklabel_style = "/pgf/number format/fixed",
        ylabel_style = { font = raw"\small" },
        "every x tick scale label/.style" = {at = "{(1,0)}", anchor = "west"}
    })

    # load file with all experimental results
    file = joinpath(@__DIR__, "..", "experiments", "data", "errors",
                    "$experiment.csv")
    df = CSV.read(file)

    # for all parameter settings
    for m in (2, 10, 100, 1000), π in 0:0.25:1
        # define model
        model = CalibrationPaperModel(m, αᵢ, π, only_firstclass)

        # load estimates
        estimates = collect_estimates(df, model)

        # compute histogram
        hist = fit(Histogram, estimates, closed = :left)

        # create axis object with histogram
        @pgf ax = Axis(PlotInc({ ybar_interval, fill = "Dark2-A!30!white" }, Table(hist)),
                       VLine({ solid, "Dark2-B", thick }, mean(estimates)))

        # add titles
        if m == 2
            ax["title"] = "\$\\pi = $π\$"
        end

        # add labels
        if π == 0
            ax["ylabel"] = "\$m = $m\$"
        end

        # compute true value
        if startswith(experiment, "ECE")
            analytic = CalibrationPaper.analytic_ece(model)
        else
            # take mean of quadratic unbiased estimates
            if startswith(experiment, "SKCEuq")
                analytic = mean(estimates)
            else
                file_uq = replace(file, r"(SKCEb|SKCEul)" => "SKCEuq")
                estimates_uq = collect_estimates(CSV.read(file_uq), model)
                analytic = mean(estimates_uq)
            end
        end

        # plot true value
        @pgf push!(ax, VLine({ dashed, "Dark2-C", thick }, analytic))

        # hack so that limits are updated as well
        @pgf push!(ax, PlotInc({ draw = "none" }, Coordinates([analytic], [0])))

        # add axis to group plot
        push!(plt, ax)
    end

    # create picture
    picture = TikzPicture(plt,
        raw"\node[anchor=north] at ($(group c1r4.west |- group c1r4.outer south)!0.5!(group c5r4.east |- group c5r4.outer south)$){calibration error estimate};",
        raw"\node[anchor=south, rotate=90] at ($(group c1r1.north -| group c1r1.outer west)!0.5!(group c1r4.south -| group c1r4.outer west)$){\# runs};")

    # save histograms
    figuresdir = joinpath(@__DIR__, "figures", "errors")
    isdir(figuresdir) || mkpath(figuresdir)
    output = joinpath(figuresdir,
                      "$(experiment)_alpha_i=$(αᵢ)_only_firstclass=$(only_firstclass).tex")
    pgfsave(output, picture; include_preamble = false)

    nothing
end

function errors()
    for only_firstclass in (true, false), αᵢ in (0.1, 1.0),
        experiment in ("ECE_uniform", "ECE_dynamic", "SKCEuq_mean", "SKCEuq_median",
                       "SKCEul_mean", "SKCEul_median", "SKCEb_mean", "SKCEb_median")

        errors(experiment, αᵢ, only_firstclass)
    end

    nothing
end

function pvalues(experiment, αᵢ, only_firstclass)
    # initialize group plot
    @pgf plt = GroupPlot(
    {
        group_style =
        {
            group_size = "3 by 2",
            xlabels_at = "edge bottom",
            ylabels_at = "edge left",
            horizontal_sep = raw"0.1\linewidth",
            vertical_sep = raw"0.05\linewidth",
            xticklabels_at = "edge bottom",
        },
        no_markers,
        grid = "major",
        tick_label_style = { font = raw"\tiny" },
        width = raw"0.3\linewidth",
        height = raw"0.2\linewidth",
        xticklabel_style = "/pgf/number format/fixed",
        ylabel_style = { font = raw"\small" },
        "every x tick scale label/.style" = {at = "{(1,0)}", anchor = "west"},
        xmin = 0, xmax = 1, ymin = -0.1, ymax = 1.1
    })

    # define range of significance levels
    αs = 0:0.01:1

    # load file with all experimental results
    file = joinpath(@__DIR__, "..", "experiments", "data", "pvalues",
                    "$experiment.csv")
    df = CSV.read(file)

    # for all parameter settings
    for m in (2, 10), π in 0:0.5:1
        # define model
        model = CalibrationPaperModel(m, αᵢ, π, only_firstclass)

        # load p-value estimates
        pvalues = collect_estimates(df, model)

        # compute empirical CDF
        empiricalCDF = ecdf(pvalues)

        if iszero(model.π)
            # if the model is calibrated we plot the empirical estimate of
            # P[p(T) < \alpha | H_0] together with the diagonal of the unit square
            @pgf ax = Axis(PlotInc({ thick }, Table(αs, empiricalCDF.(αs))),
                           PlotInc({ dashed, thick }, Coordinates([0, 1], [0, 1])))
        else
            # otherwise we plot the empirical estimate for P[p(T) > \alpha | H_1]
            @pgf ax = Axis(PlotInc({ thick }, Table(αs, 1 .- empiricalCDF.(αs))))
        end

        # add titles
        if m == 2
            ax["title"] = "\$\\pi = $π\$"
        end

        # add labels
        if π == 0
            ax["ylabel"] = "\$m = $m\$"
        end

        # add axis to group plot
        push!(plt, ax)
    end

    # create picture
    picture = TikzPicture(plt,
        raw"\node[anchor=north] at ($(group c1r2.west |- group c1r2.outer south)!0.5!(group c3r2.east |- group c3r2.outer south)$){significance level};",
        raw"\node[anchor=south, rotate=90] at ($(group c1r1.north -| group c1r1.outer west)!0.5!(group c1r2.south -| group c1r2.outer west)$){empirical test error};")

    # save histograms
    figuresdir = joinpath(@__DIR__, "figures", "pvalues")
    isdir(figuresdir) || mkpath(figuresdir)
    output = joinpath(figuresdir,
                      "$(experiment)_alpha_i=$(αᵢ)_only_firstclass=$(only_firstclass).tex")
    pgfsave(output, picture; include_preamble = false)

    nothing
end

function pvalues()
    for only_firstclass in (true, false), αᵢ in (0.1, 1.0),
        experiment in ("ECE_uniform", "ECE_dynamic",
                       "SKCEuq_mean_distribution_free", "SKCEuq_median_distribution_free",
                       "SKCEul_mean_distribution_free", "SKCEul_median_distribution_free",
                       "SKCEb_mean_distribution_free", "SKCEb_median_distribution_free",
                       "SKCEuq_mean_asymptotic", "SKCEuq_median_asymptotic",
                       "SKCEul_mean_asymptotic", "SKCEul_median_asymptotic")
        pvalues(experiment, αᵢ, only_firstclass)
    end

    nothing
end

function cifar10_errors_comparison()
    # dictionary of labels
    labels = Dict("densenet121" => "DenseNet121", "densenet161" => "DenseNet161",
                  "densenet169" => "DenseNet169", "googlenet" => "GoogLeNet",
                  "inception_v3" => "Inception", "mobilenet_v2" => "MobileNet",
                  "resnet_orig" => "ResNet", "resnet18" => "ResNet18",
                  "resnet34" => "ResNet34", "resnet50" => "ResNet50", "vgg11_bn" => "VGG11",
                  "vgg13_bn" => "VGG13", "vgg16_bn" => "VGG16", "vgg19_bn" => "VGG19")

    # load all experimental data
    file = joinpath(@__DIR__, "..", "experiments", "data", "PyTorch-CIFAR10", "errors.csv")
    df = CSV.read(file)

    # create group plot
    @pgf plt = GroupPlot(
    {
        group_style =
        {
            group_size = "1 by 2",
            xlabels_at = "edge bottom",
            vertical_sep = raw"0.15\linewidth",
        },
        ybar="0pt",
        ymajorgrids,
        width = raw"0.8\linewidth",
        height = raw"0.3\linewidth",
        xlabel = "model",
        xticklabels = [labels[model] for model in df[!, :model]],
        xtick = "data",
        xticklabel_style = { rotate = 45, anchor = "east", font = raw"\tiny" },
        scale_ticks_below_exponent = 0,
        legend_style =
        {
            cells = { anchor = "west" },
            legend_pos = "outer north east",
            font = raw"\small",
        },
        cycle_list =
        {
            "{Dark2-A,fill=Dark2-A!30!white,mark=none}",
            "{Dark2-B,fill=Dark2-B!30!white,mark=none}",
            "{Dark2-C,fill=Dark2-C!30!white,mark=none}"
        },
    },
        # plot ECE estimates
        Axis(
        {
            title = raw"$\widehat{\ECE}$",
            bar_width = "7pt",
        },
            Plot(Table({ x_expr = raw"\coordindex", y = "ECE_uniform" }, raw"\datatable")),
            Plot(Table({ x_expr = raw"\coordindex", y = "ECE_dynamic" }, raw"\datatable")),
            Legend(["uniform size", "data-dependent"])),
        # plot SKCE estimates
        Axis(
        {
            title = raw"$\widehat{\squaredkernelmeasure}$",
            bar_width = "5pt",
        },
            Plot(Table({ x_expr = raw"\coordindex", y = "SKCEb_median" }, raw"\datatable")),
            Plot(Table({ x_expr = raw"\coordindex", y = "SKCEuq_median" }, raw"\datatable")),
            Plot(Table({ x_expr = raw"\coordindex", y = "SKCEul_median" }, raw"\datatable")),
            Legend([raw"$\biasedestimator$", raw"$\unbiasedestimator$",
                    raw"$\linearestimator$"])))

    # create picture
    picture = TikzPicture(
        raw"\pgfplotstableread[col sep=comma, header=true]{" * file * raw"}\datatable",
        plt,
        raw"\node[anchor=south, rotate=90, yshift=1em] at ($(group c1r1.north -| group c1r1.outer west)!0.5!(group c1r2.south -| group c1r2.outer west)$){calibration error estimate};")

    # save plot
    figuresdir = joinpath(@__DIR__, "figures", "PyTorch-CIFAR10")
    isdir(figuresdir) || mkpath(figuresdir)
    pgfsave(joinpath(figuresdir, "errors_comparison.tex"), picture; include_preamble = false)

    nothing
end

function cifar10_pvalues_comparison()
    # dictionary of labels
    labels = Dict("densenet121" => "DenseNet121", "densenet161" => "DenseNet161",
                  "densenet169" => "DenseNet169", "googlenet" => "GoogLeNet",
                  "inception_v3" => "Inception", "mobilenet_v2" => "MobileNet",
                  "resnet_orig" => "ResNet", "resnet18" => "ResNet18",
                  "resnet34" => "ResNet34", "resnet50" => "ResNet50", "vgg11_bn" => "VGG11",
                  "vgg13_bn" => "VGG13", "vgg16_bn" => "VGG16", "vgg19_bn" => "VGG19")

    # load all experimental data
    file = joinpath(@__DIR__, "..", "experiments", "data", "PyTorch-CIFAR10", "pvalues.csv")
    df = CSV.read(file)

    # create group plot
    @pgf plt = GroupPlot(
    {
        group_style =
        {
            group_size = "1 by 2",
            xlabels_at = "edge bottom",
            vertical_sep = raw"0.1\linewidth",
        },
        ybar="0pt",
        ymajorgrids,
        width = raw"0.9\linewidth",
        height = raw"0.3\linewidth",
        xlabel = "model",
        xtick = "data",
        xticklabel_style = { rotate = 45, anchor = "east", font = raw"\tiny" },
        legend_style =
        {
            legend_columns = -1,
            "/tikz/every even column/.append style" = { column_sep = raw"0.05\linewidth" },
            font = raw"\small",
        },
        cycle_list =
        {
            "{Dark2-A,fill=Dark2-A!30!white,mark=none}",
            "{Dark2-B,fill=Dark2-B!30!white,mark=none}",
            "{Dark2-C,fill=Dark2-C!30!white,mark=none}",
            "{Dark2-D,fill=Dark2-D!30!white,mark=none}",
            "{Dark2-E,fill=Dark2-E!30!white,mark=none}",
            "{Dark2-F,fill=Dark2-F!30!white,mark=none}",
            "{Dark2-G,fill=Dark2-G!30!white,mark=none}"
        },
    },
        # plot estimates of models 1 to 7
        Axis(
        {
            xticklabels = [labels[model] for model in df[1:7, :model]],
            skip_coords_between_index = "{7}{14}",
            legend_to_name = "cifar10_pvalues_legend",
            bar_width = "5pt",
        },
            Plot(Table({ x_expr = raw"\coordindex", y = "ECE_uniform" }, raw"\datatable")),
            Plot(Table({ x_expr = raw"\coordindex", y = "ECE_dynamic" }, raw"\datatable")),
            Plot(Table({ x_expr = raw"\coordindex", y = "SKCEb_median_distribution_free" }, raw"\datatable")),
            Plot(Table({ x_expr = raw"\coordindex", y = "SKCEuq_median_distribution_free" }, raw"\datatable")),
            Plot(Table({ x_expr = raw"\coordindex", y = "SKCEul_median_distribution_free" }, raw"\datatable")),
            Plot(Table({ x_expr = raw"\coordindex", y = "SKCEuq_median_asymptotic" }, raw"\datatable")),
            Plot(Table({ x_expr = raw"\coordindex", y = "SKCEul_median_asymptotic" }, raw"\datatable")),
            Legend([raw"$\mathbf{C}_{\mathrm{uniform}}$",
                    raw"$\mathbf{C}_{\mathrm{data-dependent}}$",
                    raw"$\mathbf{D}_{\mathrm{b}}$", raw"$\mathbf{D}_{\mathrm{uq}}$",
                    raw"$\mathbf{D}_{\mathrm{l}}$", raw"$\mathbf{A}_{\mathrm{uq}}$",
                    raw"$\mathbf{A}_{\mathrm{l}}$"])),
        # plot estimates of models 8 to 14
        Axis(
        {
            xticklabels = [labels[model] for model in df[8:14, :model]],
            skip_coords_between_index = "{0}{7}",
            bar_width = "5pt",
        },
            Plot(Table({ x_expr = raw"\coordindex", y = "ECE_uniform" }, raw"\datatable")),
            Plot(Table({ x_expr = raw"\coordindex", y = "ECE_dynamic" }, raw"\datatable")),
            Plot(Table({ x_expr = raw"\coordindex", y = "SKCEb_median_distribution_free" }, raw"\datatable")),
            Plot(Table({ x_expr = raw"\coordindex", y = "SKCEuq_median_distribution_free" }, raw"\datatable")),
            Plot(Table({ x_expr = raw"\coordindex", y = "SKCEul_median_distribution_free" }, raw"\datatable")),
            Plot(Table({ x_expr = raw"\coordindex", y = "SKCEuq_median_asymptotic" }, raw"\datatable")),
            Plot(Table({ x_expr = raw"\coordindex", y = "SKCEul_median_asymptotic" }, raw"\datatable"))))

    # create picture
    picture = TikzPicture(
        raw"\pgfplotstableread[col sep=comma, header=true]{" * file * raw"}\datatable",
        plt,
        raw"\node[anchor=north] at ($(group c1r2.west |- group c1r2.outer south)!0.5!(group c1r2.east |- group c1r2.outer south)$){\pgfplotslegendfromname{cifar10_pvalues_legend}};",
        raw"\node[anchor=south, rotate=90] at ($(group c1r1.north -| group c1r1.outer west)!0.5!(group c1r2.south -| group c1r2.outer west)$){bound/approximation of p-value};")

    # save plot
    figuresdir = joinpath(@__DIR__, "figures", "PyTorch-CIFAR10")
    isdir(figuresdir) || mkpath(figuresdir)
    pgfsave(joinpath(figuresdir, "pvalues_comparison.tex"), picture; include_preamble = false)

    nothing
end

function timings()
    # load all experimental data
    file = joinpath(@__DIR__, "..", "experiments", "data", "timings.csv")
    df = CSV.read(file)

    # initialize group plot
    @pgf plt = GroupPlot(
    {
        group_style =
        {
            group_size = "2 by 2",
            horizontal_sep = raw"0.1\linewidth",
            vertical_sep = raw"0.15\linewidth",
        },
        grid = "major",
        width = raw"0.45\linewidth",
        height = raw"0.3\linewidth",
        legend_style =
        {
            legend_columns = -1,
            font = raw"\small",
        },
    })

    # define estimators and labels
    estimators = ["ECE_uniform", "ECE_dynamic", "SKCEb_median", "SKCEuq_median",
                  "SKCEul_median"]
    labels = [raw"$\widehat{\ECE}$ (uniform)", raw"$\widehat{\ECE}$ (data-dependent)",
              raw"$\biasedestimator$", raw"$\unbiasedestimator$", raw"$\linearestimator$"]

    # plot timings for different number of classes
    for nclasses in (2, 10, 100, 1000)
        # initialize next axis
        @pgf ax = LogLogAxis({ title = raw"$m = " * string(nclasses) * raw"$" })

        # plot all estimators
        for estimator in estimators
            # extract data
            df_subset = @from i in df begin
                @where i.estimator == estimator && i.nclasses == nclasses
                @select {i.nsamples, i.time}
                @collect DataFrame
            end

            @pgf push!(ax, PlotInc({ x = "nsamples", y = "time" }, Table(df_subset)))
        end

        # add legend to the first axis
        if nclasses == 2
            ax["legend to name"] = "timings_legend"
            @pgf push!(ax, Legend(labels))
        end

        # add axis to group plot
        @pgf push!(plt, ax)
    end

    # create picture
    picture = TikzPicture(
        plt,
        raw"\node[anchor=north, align=center] at ($(group c1r2.west |- group c1r2.outer south)!0.5!(group c2r2.east |- group c2r2.outer south)$){number of samples \\[0.05\linewidth] \pgfplotslegendfromname{timings_legend}};",
        raw"\node[anchor=south, rotate=90] at ($(group c1r1.north -| group c1r1.outer west)!0.5!(group c1r2.south -| group c1r2.outer west)$){time [sec]};")

    # save plot
    figuresdir = joinpath(@__DIR__, "figures")
    isdir(figuresdir) || mkpath(figuresdir)
    pgfsave(joinpath(figuresdir, "timings.tex"), picture; include_preamble = false)

    nothing
end
