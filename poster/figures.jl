using CalibrationPaper
using CSV
using DataFrames
using PGFPlotsX
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
        width = raw"0.23\linewidth",
        height = raw"0.155\linewidth",
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
            ax["title"] = "\$\\symbf{M$j}\$"
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
        raw"\node[anchor=south, rotate=90, yshift=1ex] at ($(group c1r1.north -| group c1r1.outer west)!0.5!(group c1r4.south -| group c1r4.outer west)$){\# runs};")

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
            vertical_sep = raw"0.015\linewidth",
            xticklabels_at = "edge bottom",
        },
        no_markers,
        tick_label_style = { font = raw"\tiny" },
        grid = "major",
        title_style = { align = "center" },
        width = raw"0.23\linewidth",
        height = raw"0.155\linewidth",
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
    labels = [raw"$\symbf{C}$", raw"$\symbf{D}_{\mathrm{b}}$",
              raw"$\symbf{D}_{\mathrm{uq}}$", raw"$\symbf{D}_{\mathrm{l}}$",
              raw"$\symbf{A}_{\mathrm{uq}}$",
              raw"$\symbf{A}_{\mathrm{l}}$"]

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
            ax["title"] = "\$\\symbf{M$j}\$"
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
