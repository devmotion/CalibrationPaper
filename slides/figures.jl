using CalibrationPaper
using CSV
using DataFrames
using PGFPlotsX
using StatsBase

function errors_ece()
    # initialize group plot
    @pgf plt = GroupPlot(
    {
        group_style =
        {
            group_name = "group",
            group_size = "3 by 1",
            horizontal_sep = raw"0.02\textwidth",
            vertical_sep = "0pt",
            ylabels_at = "edge left",
            yticklabels_at = "edge left"
        },
        ylabel = raw"\# runs",
        no_markers,
        label_style = { font = raw"\small" },
        tick_label_style = { font = raw"\tiny" },
        grid = "major",
        width = raw"0.19\textwidth",
        height = raw"0.08\textwidth",
        "every x tick scale label/.style" = { at = "{(1,0)}", anchor = "west" },
        "scale only axis",
        ymin = 0, ymax = 3500,
        legend_cell_align = "left",
        legend_style =
        {
            fill = "none",
            draw = "none",
            font = raw"\small",
            inner_sep = "0pt",
            at = "({1.1, 1})",
            anchor = "north west" }
    })

    # define displayed models
    models = [CalibrationPaperModel(10, 0.1, 0.0, true),
              CalibrationPaperModel(10, 0.1, 0.5, true),
              CalibrationPaperModel(10, 0.1, 1.0, false)]

    # load experimental results
    datadir = joinpath(@__DIR__, "..", "experiments", "data", "errors")
    df = CSV.read(joinpath(datadir, "ECE_uniform.csv"))

    # for all studied experiments and models
    for (i, model) in enumerate(models)
        # load estimates
        estimates = collect_estimates(df, model)

        # compute histogram
        hist = fit(Histogram, estimates, closed = :left)

        # create axis object with histogram
        @pgf ax = Axis(PlotInc(
            {
                ybar_interval,
                fill = "Dark2-A!30!white",
                forget_plot
            },
            Table(hist)))

        # add mean of estimates
        @pgf push!(ax, VLine({ solid, thick, "Dark2-B" }, mean(estimates)))
        if i == 3
            push!(ax,
                  raw"\addlegendimage{solid, thick, Dark2-B, no markers};",
                  LegendEntry("mean estimate"))
        end

        # compute true value
        analytic = CalibrationPaper.analytic_ece(model)

        # plot true value
        @pgf push!(ax, VLine({ dashed, thick, "Dark2-C" }, analytic))
        if i == 3
            push!(ax, raw"\addlegendimage{dashed, thick, Dark2-C, no markers};",
                  LegendEntry(raw"$\mathrm{ECE}$"))
        end

        # hack so that limits are updated as well
        @pgf push!(ax, PlotInc({ draw = "none" }, Coordinates([analytic], [0])))

        # add axis to group plot
        push!(plt, ax)
    end

    # save histogram
    figuresdir = joinpath(@__DIR__, "figures")
    isdir(figuresdir) || mkdir(figuresdir)
    picture = TikzPicture(plt,
                          raw"\node[anchor=north, font=\small] at ($(group c1r1.west |- group c1r1.outer south)!0.5!(group c3r1.east |- group c3r1.outer south)$){$\mathrm{ECE}$ estimate};")
    pgfsave(joinpath(figuresdir, "errors_ece.tex"), picture; include_preamble = false)

    nothing
end
