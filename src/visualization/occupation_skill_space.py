import os
import scipy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from src import dictionaries, utils
from src.data.framework import Esco
from src.plotting_utils import discrete_cmap_with_manual_colors

from mpl_toolkits.axes_grid1 import make_axes_locatable

useful_paths = utils.UsefulPaths()
esco = Esco()


def plot_cooc_matrix_ordered_by_isco(
    version="unweighted",
    n_classes=10,
    cmap_type="Blues",
    annotate_isco=1,
    line_alpha=0.3,
    linewidth=0.5,
    figsize=(16, 9),
    despine=False,
    zoom=False,
    zooming_margins=(0.1, 0.05),
    save_fig=True,
    dpi=600,
    out_path=os.path.join(
        useful_paths.figure_dir,
        "occupation_skill_space",
        "occ_sim_matrix_cooc_isco{isco_groups}_{version}_cmap_{cmap_type}{n_classes}.png",
    ),
):
    """
    Plot occupation similarity matrix based on skills overlap. Based on ESCO v.1.1.

    Parameters
    ----------
    version : str
        Version of the COOC matrix.
    n_classes : int
        N classes of discrete cmap.
    cmap_type : str
        Cmap name.
    annotate_isco : int
        Whether ISCO-08 1-digit categories should be indicated.
    figsize : tuple
        Figure size.
    despine : boolean
        Whether to despine all axis
    save_fig : boolean
        Whether to save the figure.
    out_path : str
        Target path for saved plots.

    Returns
    -------

    """
    occ = esco.occupations

    # read bipartite adjecency matrix for occupations and skills
    osm = esco.read_occ_skills_matrix(return_version=version)
    osm_sparse = scipy.sparse.csr_matrix(osm.values)

    # Co-occurrence matrix
    sim_matrix_sparse = osm_sparse.dot(osm_sparse.T)
    sim_matrix = sim_matrix_sparse.toarray()
    np.fill_diagonal(sim_matrix, 0)

    # reindex matrix by isco level
    reindex_by_isco4 = occ.sort_values("iscoGroup").index.values
    x, y = np.meshgrid(reindex_by_isco4, reindex_by_isco4)

    # plot
    cmap_new = discrete_cmap_with_manual_colors(cmap_type, n_classes)

    fig, ax = plt.subplots(figsize=figsize)

    # enables zooming
    ax.use_sticky_edges = False

    im = plt.imshow(sim_matrix[x, y], cmap=cmap_new, vmin=0, vmax=n_classes)
    plt.colorbar(
        im,
        fraction=0.046,
        pad=0.03,
        shrink=0.6,
        label="Skills overlap [-]",
        orientation="vertical",
        extend="max",
    )

    # cbar.ax.set_ylabel()
    plt.xlabel("Target occupations")
    plt.ylabel("Source occupations")

    if despine:
        sns.despine(left=True, bottom=True)

    # indicate isco major groups
    if annotate_isco is not None:
        isco_level = annotate_isco
        rdict = {110: 0000, 210: 0000, 310: 0000}
        reindex_isco_lvl = (
            occ.sort_values("iscoGroup")
            .iscoGroup.replace(rdict)
            .astype(str)
            .str.slice(0, isco_level)
            .astype(int)
        )

        # how many unique occupations do we need to consider?
        unique_occs = reindex_isco_lvl.unique()

        # where do we see a change in occupation category?
        df_reindex_isco_lvl = reindex_isco_lvl.diff().reset_index()

        # where do we see jumps in the occupation id?
        lvl_boundaries = df_reindex_isco_lvl.loc[df_reindex_isco_lvl.iscoGroup > 0]

        # determine locations of tick boundaries
        tick_locs = np.insert(
            lvl_boundaries.index.values, [0, len(unique_occs) - 1], [0, len(occ)]
        )

        # determine locations of center ticks
        center_ticks = []
        for i in range(len(tick_locs) - 1):
            c = (tick_locs[i] + tick_locs[i + 1]) / 2
            center_ticks.append(c)

        # plot ticks and labels
        # plt.xticks(
        #     center_ticks, list(dictionaries.isco_lvl1_mapping.values()), rotation=90
        # )
        # plt.yticks(center_ticks, list(dictionaries.isco_lvl1_mapping.values()))

        # annotate lines
        for i in df_reindex_isco_lvl.index.values:
            if df_reindex_isco_lvl.loc[i, "iscoGroup"] > 0:
                plt.axvline(
                    x=i,
                    color="grey",
                    linewidth=linewidth,
                    alpha=line_alpha,
                    linestyle="--",
                )
                plt.axhline(
                    y=i,
                    color="grey",
                    linewidth=linewidth,
                    alpha=line_alpha,
                    linestyle="--",
                )

        if zoom:
            ax.margins(x=zooming_margins[0], y=zooming_margins[1])

    plt.tight_layout()

    if save_fig:
        plt.savefig(
            out_path.format(
                isco_groups=annotate_isco,
                version=version,
                cmap_type=cmap_type,
                n_classes=n_classes,
            ),
            bbox_inches="tight",
            dpi=dpi,
        )
    else:
        plt.show()


def plot_occ_sim_matrix_ordered_by_isco(
    version="weighted",
    sim_metric="excess",
    n_classes=10,
    vmax=100,
    cmap_type="Reds",
    annotate_isco1d=True,
    figsize=(16, 9),
    despine=False,
    save_fig=True,
    grid_color="Grey",
    out_path=os.path.join(
        useful_paths.figure_dir,
        "occupation_skill_space",
        "occ_sim_matrix_skill_{sim_metric}_isco_groups_{version}_cmap_{cmap_type}{n_classes}.png",
    ),
):
    """
    Plot occupation similarity matrix based on skills shortage. Based on ESCO v.1.1.

    Parameters
    ----------
    version : str
        Version of the OSM matrix.
    n_classes : int
        N classes of discrete cmap.
    cmap_type : str
        Cmap name.
    annotate_isco1d : bool
        Whether ISCO-08 1-digit categories should be indicated.
    figsize : tuple
        Figure size.
    despine : boolean
        Whether to despine all axis
    save_fig : boolean
        Whether to save the figure.
    out_path : str
        Target path for saved plots.

    Returns
    -------

    """
    if version == "weighted" and sim_metric == "cooc":
        # read bipartite adjacency matrix for occupations and skills
        occ_skills_mat = esco.read_occ_skills_matrix(return_version=version)

        # esco-level similarity
        sim_matrix = np.dot(occ_skills_mat.values, occ_skills_mat.values.transpose())
    elif version == "weighted" and sim_metric == "shortage":
        sim_matrix = pd.read_pickle(
            os.path.join(
                useful_paths.data_processed,
                "esco",
                "occ_sim_matrix_weighted_skill_shortage.pkl",
            )
        ).values
    elif version == "weighted" and sim_metric == "excess":
        sim_matrix = pd.read_pickle(
            os.path.join(
                useful_paths.data_processed,
                "esco",
                "occ_sim_matrix_weighted_skill_excess.pkl",
            )
        ).values
    elif version == "weighted" and sim_metric == "shortage_excess_avg":
        sim_matrix = pd.read_pickle(
            os.path.join(
                useful_paths.data_processed,
                "esco",
                "occ_sim_matrix_weighted_skill_shortage_excess_avg.pkl",
            )
        ).values

    # reindex matrix by isco level
    occ = esco.occupations
    reindex_by_isco4 = occ.sort_values("iscoGroup").index.values
    x, y = np.meshgrid(reindex_by_isco4, reindex_by_isco4)

    # plot
    cmap_new = discrete_cmap_with_manual_colors(cmap_type, n_classes)
    cmap_new.set_over("k")

    plt.figure(figsize=figsize)
    plt.imshow(sim_matrix[x, y], cmap=cmap_new, vmin=0, vmax=vmax)
    cbar = plt.colorbar(fraction=0.046, pad=0.04, orientation="vertical", extend="max")
    cbar.ax.set_ylabel("Skill {} [-]".format(sim_metric))
    plt.xlabel("Target occupations")
    plt.ylabel("Source occupations")

    if despine:
        sns.despine(left=True, bottom=True)

    # indicate isco major groups
    if annotate_isco1d:
        rdict = {110: 0000, 210: 0000, 310: 0000}
        reindex_isco_lvl1 = (
            occ.sort_values("iscoGroup")
            .iscoGroup.replace(rdict)
            .astype(str)
            .str.slice(0, 1)
            .astype(int)
        )
        df_reindex_isco_lvl1 = reindex_isco_lvl1.diff().reset_index()
        lvl1_boundaries = df_reindex_isco_lvl1.loc[df_reindex_isco_lvl1.iscoGroup == 1]
        tick_locs = np.insert(lvl1_boundaries.index.values, [0, 9], [0, len(occ)])

        center_ticks = []
        for i in range(len(tick_locs) - 1):
            c = (tick_locs[i] + tick_locs[i + 1]) / 2
            center_ticks.append(c)

        plt.xticks(
            center_ticks, list(dictionaries.isco_lvl1_mapping.values()), rotation=90
        )
        plt.yticks(center_ticks, list(dictionaries.isco_lvl1_mapping.values()))

        for i in df_reindex_isco_lvl1.index.values:
            if df_reindex_isco_lvl1.loc[i, "iscoGroup"] == 1:
                plt.axvline(
                    x=i, color=grid_color, linewidth=1, alpha=0.5, linestyle="--"
                )
                plt.axhline(
                    y=i, color=grid_color, linewidth=1, alpha=0.5, linestyle="--"
                )

    plt.tight_layout()

    if save_fig:
        plt.savefig(
            out_path.format(
                sim_metric=sim_metric,
                version=version,
                cmap_type=cmap_type,
                n_classes=n_classes,
            ),
            bbox_inches="tight",
            dpi=300,
        )
    else:
        plt.show()


if __name__ == "__main__":
    with sns.axes_style("ticks"), sns.plotting_context("notebook", font_scale=1.2):

        # plot_cooc_matrix_ordered_by_isco(version="unweighted", despine=True,
        #                                  cmap_type="Blues", n_classes=20)
        plot_cooc_matrix_ordered_by_isco(
            version="weighted",
            despine=True,
            cmap_type="Blues",
            n_classes=20,
            annotate_isco=3,
            dpi=600,
            zooming_margins=(-0.45, -0.45),
            zoom=True,
        )

        # plot_occ_sim_matrix_ordered_by_isco(sim_metric="shortage", cmap_type="Reds")
        # plot_occ_sim_matrix_ordered_by_isco(sim_metric="excess", cmap_type="Blues")
        # plot_occ_sim_matrix_ordered_by_isco(
        #     sim_metric="shortage_excess_avg", cmap_type="Blues"
        # )
