import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt


def move_legend_to_right(ax, scale=0.8):
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * scale, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))


def move_legend_to_top(ax, scale=0.8, ncol=None, lower_anchor=0.5, upper_anchor=1.1):
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * scale])

    # Put a legend to the top of the current axis
    ax.legend(
        loc="upper center", bbox_to_anchor=(lower_anchor, upper_anchor), ncol=ncol
    )


# https://stackoverflow.com/questions/43214978/seaborn-barplot-displaying-values
def show_values_on_bars(axs, h_v="v", space=0.4, formatter="{:.2f}"):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = formatter.format(p.get_height())
                ax.text(_x, _y, value, ha="center")
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = formatter.format(p.get_width())
                ax.text(_x, _y, value, ha="left", va="bottom")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def plot_stackedbar_p(df, labels, colors, title, subtitle):
    """
    Source: https://towardsdatascience.com/stacked-bar-charts-with-pythons-matplotlib-f4020e4eb4a7

    Parameters
    ----------
    df
    labels
    colors
    title
    subtitle

    Returns
    -------

    """
    fields = df.columns.tolist()

    # figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 10))

    # plot bars
    left = len(df) * [0]
    for idx, name in enumerate(fields):
        plt.barh(df.index, df[name], left=left, color=colors[idx])
        left = left + df[name]

    # title and subtitle
    plt.title(title, loc="left")
    plt.text(0, ax.get_yticks()[-1] + 0.75, subtitle)

    # legend
    plt.legend(labels, bbox_to_anchor=([0.58, 1, 0, 0]), ncol=4, frameon=False)

    # remove spines
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # format x ticks
    xticks = np.arange(0, 1.1, 0.1)
    xlabels = ["{}%".format(i) for i in np.arange(0, 101, 10)]
    plt.xticks(xticks, xlabels)

    # adjust limits and draw grid lines
    plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
    ax.xaxis.grid(color="gray", linestyle="dashed")

    plt.show()


def discrete_cmap_with_manual_colors(
    cmap_type="Blues", n_classes=10, colour_replacements=None
):
    # default setting: replace color of first cbar segment with white
    if colour_replacements is None:
        colour_replacements = {0: "white"}

    cmap = plt.get_cmap(cmap_type, n_classes)

    # extract all colors from the cmap
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # update with manual color changs
    for k, v in colour_replacements.items():
        cmaplist[k] = v

    # create the new map
    cmap_new = mpl.colors.LinearSegmentedColormap.from_list(
        "Custom cmap", cmaplist, cmap.N
    )
    return cmap_new
