import os
import copy
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
import seaborn as sns
import matplotlib.pyplot as plt

from src import UsefulPaths, utils
from src.data.lfs import EulfsDs, EuLfs
from src.data.framework import Classifications
from src.stats_utils import correlation_matrix

useful_paths = UsefulPaths()


def correlation_matrix_plot(df, significance_level=0.05, cbar_levels=8, figsize=(6, 6)):
    """Plot corrmat considering p-vals."""
    corr, pvals = correlation_matrix(df)

    # create triangular mask for heatmap
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # mask corrs based on p-values
    pvals_plot = np.where(pvals > significance_level, np.nan, corr)

    # plot
    # -------------------------------------------------------------------------
    # define correct cbar height and pass to sns.heatmap function
    fig, ax = plt.subplots(figsize=figsize)
    cbar_kws = {"fraction": 0.046, "pad": 0.04}
    sns.heatmap(
        corr,
        mask=mask,
        cmap=sns.diverging_palette(20, 220, n=cbar_levels),
        square=True,
        vmin=-1,
        center=0,
        vmax=1,
        annot=pvals_plot,
        cbar_kws=cbar_kws,
    )
    plt.title("p < {:.3f}".format(significance_level))
    plt.tight_layout()
    return fig, ax


def eulfs_maps(
    year,
    input_file=os.path.join(
        useful_paths.data_processed, "eulfs", "eulfs_{}_by_NUTS_ID_NACE1D_label.pkl"
    ),
    out_dir=os.path.join(useful_paths.results_dir, "eulfs"),
    slice_by_industry=False,
    figsize=(10, 10),
    legend=True,
    vmin=0,
    cbar_label="Employment fraction [-]",
):
    # read input data
    df = pd.read_pickle(input_file.format(year))
    print(df)


# TODO: make data_panel assignable by path or similar (currently hardcoded)
class EulfsVis(EulfsDs):
    """Class for visualising EU-LFS data."""

    def __init__(self, fn_config_path, fn_config_data, fn_vis_config, year=2019):
        """

        Parameters
        ----------
        fn_config_path : str
        Name of yml file containing path configurations.
        fn_config_data : str
            Name of yml file containing data configurations.
        fn_vis_config : str
            Name of yml file containing visualisation configurations.
        year : int
            Year subset of eu lfs data
        """
        # inheritance
        EulfsDs.__init__(
            self=self, fn_config_data=fn_config_data, fn_config_path=fn_config_path
        )

        self.year = year

        # parse config file and unpack
        self.config_vis = utils.load_config(
            os.path.join(self.config_dir, fn_vis_config)
        )

        self.epsg = str(self.config_vis["EU_MAPS"]["epsg"])
        self.bbox_eu_epsg_3035 = self.config_vis["EU_MAPS"]["bbox_eu_epsg_3035"]
        self.missing_kwds = self.config_vis["EU_MAPS"]["missing_kwds"]
        self.edgecolor = self.config_vis["EU_MAPS"]["edgecolor"]
        self.linewidth = self.config_vis["EU_MAPS"]["linewidth"]
        self.robust_cmap_percentiles = self.config_vis["EU_MAPS"][
            "robust_cmap_percentiles"
        ]
        self.cmap = self.config_vis["EU_MAPS"]["cmap"]
        self.countries_for_maps = self.config_vis["EU_MAPS"]["countries_for_maps"]

        # read country borders
        self.gdf_nuts_0 = self.gdf_nuts[self.epsg].loc[
            self.gdf_nuts[self.epsg]["LEVL_CODE"] == 0
        ]

        # read data panel
        self.data_panel = self.read(year=self.year)

        # todo: needs to become less hardcoded. what about wanting to plot other
        #  variables than the GBN employment shares?
        # retrieve cols that should be aggregated
        self.cols_to_calc = self.data_panel.columns[
            self.data_panel.columns.str.startswith("COEFF_")
        ].to_list()
        self.cols_to_calc_all = copy.copy(self.cols_to_calc)
        self.cols_to_calc_all.insert(0, "COEFF")

        self.cols_to_aggregate = [
            "COEFF",
            "COEFF_share_green",
            "COEFF_share_brown_sl",
            "COEFF_share_brown_slt",
        ]

        self.cols_to_plot = [
            "COEFF_share_green_rel",
            "COEFF_share_brown_sl_rel",
            "COEFF_share_brown_slt_rel",
        ]

    # todo: potentially move to new class in modelling submodule
    def aggregate_by_region(self, industry=None, epsg="3035"):
        """
        Aggregate variables by NUTS regions.

        Parameters
        ----------
        epsg : str
            EPSG code of map projection

        Returns
        -------
        gdf_sub : gpd.GeoDataFrame
            Variables aggregated by NUTS regions. Ready to plot maps from.
        """
        target_fdir = os.path.join(self.path_eulfs_processed, "aggregated_by_region")
        target_fname = "eulfs_vars_by_nuts_{}_{}.shp".format(
            self.n_digits_nuts, self.year
        )
        target_fpath = os.path.join(target_fdir, target_fname)
        utils.ccdir(target_fdir)

        # get correct shapefile
        gdf_digits_match = self.gdf_nuts[epsg].loc[
            self.gdf_nuts[epsg]["LEVL_CODE"] == self.n_digits_nuts
        ]
        # gdf_digits_match = self.gdf_nuts[epsg]

        # todo (!): move to visualisation instead of aggregation function
        # subset countries to plot
        data_panel = self.data_panel[
            self.data_panel["COUNTRYW"].isin(self.countries_for_maps)
        ]

        # optional: subset by industries
        if industry is not None:
            industry_mask = data_panel.loc[:, "NACE1D_label"] == industry
            data_panel = data_panel.loc[industry_mask]

        # aggregate over regions
        df_all_agg = (
            data_panel.groupby("NUTS_ID")[self.cols_to_calc_all].sum().reset_index()
        )

        # combine
        gdf_sub = pd.merge(gdf_digits_match, df_all_agg, on="NUTS_ID", how="left")
        gdf_sub = gpd.GeoDataFrame(gdf_sub, crs="EPSG:{}".format(self.epsg))

        # calculate relative shares at regional level
        for col in self.cols_to_calc:
            gdf_sub["{}_relative".format(col)] = gdf_sub[col] / gdf_sub["COEFF"]

        # todo: work around colname length restriction stemming from fiona when saving
        # gdf_sub.to_file(target_fpath)
        return gdf_sub

    def aggregate_by_industry(self, cols_to_aggregate=None, grouping=None):
        """
        Aggregate variables by NACE industrial sectors.

        Returns
        -------
        df_all_agg_by_grouping : pd.DataFrame
            Variables aggregated by grouping.
        """
        if grouping is None:
            grouping = ["NACE1D_label", "COUNTRYW"]
        target_fdir = os.path.join(self.path_eulfs_processed, "aggregated_by_industry")
        target_fname = "eulfs_vars_by_nace_{}_{}".format(self.n_digits_nace, self.year)
        utils.ccdir(target_fdir)
        target_fpath = os.path.join(target_fdir, target_fname + ".pkl")

        # aggregate over industries and countries
        # todo: make more flexible to work with different NACE digit levels
        df_all_agg_by_grouping = (
            self.data_panel.groupby(grouping)[cols_to_aggregate].sum().reset_index()
        )

        for col in cols_to_aggregate:
            df_all_agg_by_grouping["{}_relative".format(col)] = (
                df_all_agg_by_grouping[col] / df_all_agg_by_grouping["COEFF"]
            )

        utils.save_df_to_files(df_all_agg_by_grouping, target_fdir, target_fname)
        return df_all_agg_by_grouping

    def aggregate_by_occupation(self):
        """
        Aggregate variables by ISCO occupations.

        Returns
        -------
        df_all_agg_by_cntr_occ : pd.DataFrame
            Variables aggregated by occupation-country pairs.
        """
        target_fdir = os.path.join(
            self.path_eulfs_processed, "aggregated_by_occupation"
        )
        target_fname = "eulfs_vars_by_isco_{}_{}".format(
            self.n_digits_isco08, self.year
        )
        utils.ccdir(target_fdir)
        target_fpath = os.path.join(target_fdir, target_fname + ".pkl")

        # aggregate over occupations and countries
        df_all_agg_by_cntr_occ = (
            self.data_panel.groupby(
                [self.fmt_string_isco_label.format(self.n_digits_isco08), "COUNTRYW"]
            )[self.cols_to_calc_all]
            .sum()
            .reset_index()
        )

        for col in self.cols_to_aggregate:
            df_all_agg_by_cntr_occ["{}_relative".format(col)] = (
                df_all_agg_by_cntr_occ[col] / df_all_agg_by_cntr_occ["COEFF"]
            )

        utils.save_df_to_files(df_all_agg_by_cntr_occ, target_fdir, target_fname)

        return df_all_agg_by_cntr_occ

    def create_maps(
        self,
        slice_by_industry=False,
        figsize=(10, 10),
        legend=True,
        vmin=0,
        vmax=None,
        cmap="Greys",
        n_cats=None,
        cbar_label="Employment fraction [-]",
        out_dir=os.path.join(
            useful_paths.figure_dir,
            "03_eulfs",
            "2019",
            "employment_shares_by_region",
        ),
    ):
        """
        Create EU maps based on LFS data.

        Parameters
        ----------
        out_dir
        base_dir_name
        slice_by_industry
        figsize
        legend
        vmin
        cbar_label

        Returns
        -------
        Saves plots in dedicated folder.

        """
        logger = logging.getLogger(__name__)
        logger.info("GBN employment shares by region.")

        # read aggregated data
        if not slice_by_industry:
            # todo: should be read from an existing file
            gdf_sub = self.aggregate_by_region()

            # select cols that should be plotted
            # todo: fix hardcoded retrieval of column name identifiers
            cols_to_plot = gdf_sub.columns[
                gdf_sub.columns.str.startswith("COEFF_")
                # & gdf_sub.columns.str.endswith("relative")
            ].to_list()

            self.plot_map(
                gdf_sub=gdf_sub,
                cols_to_plot=cols_to_plot,
                out_dir=out_dir,
                figsize=figsize,
                cbar_label=cbar_label,
                legend=legend,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                n_cats=n_cats,
            )
        else:
            industry_labels = self.df_nace.loc[:, "NACE1D_label"]
            for i, industry in enumerate(industry_labels):
                print(industry)
                out_dir_industry = os.path.join(out_dir, "{}_{}".format(i, industry))

                # subset by industry before aggregating by region
                gdf_sub = self.aggregate_by_region(industry=industry)

                # select cols that should be plotted
                # todo: fix hardcoded retrieval of column name identifiers
                cols_to_plot = gdf_sub.columns[
                    gdf_sub.columns.str.startswith("COEFF_")
                    # & gdf_sub.columns.str.endswith("relative")
                ].to_list()

                self.plot_map(
                    gdf_sub=gdf_sub,
                    cols_to_plot=cols_to_plot,
                    out_dir=out_dir_industry,
                    figsize=figsize,
                    cbar_label=cbar_label,
                    legend=legend,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    n_cats=n_cats,
                    additional_title_info=industry,
                )
        return None

    def plot_map(
        self,
        gdf_sub,
        cols_to_plot,
        out_dir,
        figsize,
        cbar_label,
        legend,
        vmin,
        vmax=None,
        cmap="Greys",
        cmap_set_over=None,
        n_cats=None,
        max_folder_name_length=50,
        additional_title_info=None,
        save_data=True,
    ):
        # parse plotting params
        legend_kwds = {"label": cbar_label, "fraction": 0.03, "extend": "max"}
        p_low, p_high = self.robust_cmap_percentiles
        xmin, xmax, ymin, ymax = self.bbox_eu_epsg_3035

        for column in cols_to_plot:
            print(column)

            # create figure
            fig, ax = plt.subplots(figsize=(10, 10))

            # dynamically set n categories and vmax
            q_high, q_low = np.nanpercentile(gdf_sub[column].values, q=[p_high, p_low])
            n_cats_auto = np.ceil(q_high * 100)
            # vmax = np.ceil(gdf_sub[column].max() * 100) / 100

            # optionally discretize cmap
            if n_cats == "auto":
                cmap = plt.get_cmap(cmap, n_cats_auto)
            elif isinstance(n_cats, int):
                cmap = plt.get_cmap(cmap, n_cats)
            else:
                cmap = plt.get_cmap(cmap)

            # extend max, color for above triangle
            if cmap_set_over is not None:
                cmap.set_over(cmap_set_over)

            # plot data
            gdf_sub.plot(
                ax=ax,
                column=column,
                figsize=figsize,
                legend=legend,
                cmap=cmap,
                vmax=q_high if vmax is None else vmax,
                vmin=q_low if vmin is None else vmin,
                missing_kwds=self.missing_kwds,
                legend_kwds=legend_kwds,
                edgecolor=self.edgecolor,
                linewidth=self.linewidth,
            )

            # add country borders (NUTS 0)
            self.gdf_nuts_0.geometry.boundary.plot(
                ax=ax, color=None, edgecolor="black", linewidth=0.5
            )

            # labelling
            title_snippet = (
                additional_title_info if additional_title_info is not None else ""
            )
            ax.set_title("{} ({}) \n {}".format(column, self.year, title_snippet))
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            # layout
            ax.axis("off")
            plt.tight_layout()

            # save
            # todo: needs a fix to save industry slices
            split_path = out_dir.split(os.sep)
            folder_name = split_path[-1]
            if len(folder_name) >= max_folder_name_length:
                folder_name_short = folder_name[:max_folder_name_length]
                split_path[-1] = folder_name_short
                out_dir = "{}".format(os.sep).join(split_path)

            utils.ccdir(out_dir)

            try:
                plt.savefig(
                    os.path.join(out_dir, "{}.png".format(column)),
                    dpi=150,
                    bbox_inches="tight",
                )
            except FileNotFoundError as e:
                print(e)
                continue

            plt.cla()
            plt.close(fig)
        plt.close()

        if save_data:
            gdf_sub.to_csv(
                os.path.join(out_dir, "{}.csv".format(additional_title_info)),
                encoding="utf-8",
            )

    def create_industry_boxplots(
        self,
        year=2019,
        grouping=None,
        cols_to_aggregate=None,
        cols_to_plot=None,
        x_lims=(0, 0.4),
        out_dir=os.path.join(
            useful_paths.figure_dir,
            "03_eulfs",
            "{year}",
            "employment_shares_by_industry",
        ),
    ):

        logger = logging.getLogger(__name__)
        logger.info("GBN employment shares by industry.")

        # decide over variable selection
        cols_to_aggregate = (
            cols_to_aggregate
            if cols_to_aggregate is not None
            else self.cols_to_aggregate
        )
        cols_to_plot = cols_to_plot if cols_to_plot is not None else self.cols_to_plot

        # aggregate
        if grouping is None:
            grouping = ["NACE1D_label", "COUNTRYW"]

        if len(grouping) > 1:
            df_plotting = self.aggregate_by_industry(
                cols_to_aggregate=cols_to_aggregate, grouping=grouping
            )
        else:
            df_plotting = self.data_panel

            for col in cols_to_aggregate:
                df_plotting["{}_relative".format(col)] = (
                    df_plotting[col] / df_plotting["COEFF"]
                )

        # plot
        y_var = "NACE{}D_label".format(self.n_digits_nace)

        for x_var in cols_to_plot:
            print("plotting variable: {}".format(x_var))

            # infer order (sorted by median in descending order)
            plotting_order = (
                df_plotting.loc[:, (y_var, x_var)]
                .groupby(y_var)
                .median()
                .sort_values(by=x_var, ascending=False)
                .index.values
            )

            # create plot
            fig, ax = plt.subplots(figsize=(20, 10))
            print(df_plotting.info())
            sns.boxplot(
                data=df_plotting,
                x=x_var,
                y=y_var,
                color=".5",
                fliersize=0,
                order=plotting_order,
            )

            # if there is a second group variable, define as hue and overlay scatter
            if len(grouping) > 1:
                hue_var = grouping[1]
                n_hue_colors = len(df_plotting[hue_var].unique())
                palette = sns.color_palette("cubehelix", n_colors=n_hue_colors)

                sns.scatterplot(
                    data=df_plotting.set_index([y_var, hue_var])
                    .reindex(plotting_order, level=0)
                    .reset_index(),
                    x=x_var,
                    y=y_var,
                    hue=hue_var,
                    style=hue_var,
                    palette=palette,
                    zorder=5,
                    edgecolor="black"
                    # order=plotting_order,
                )

            # labelling
            if x_lims is not None:
                ax.set_xlim(x_lims)

            # Shrink current axis by x %
            shrinkage_factor = 0.2
            box = ax.get_position()
            ax.set_position(
                [box.x0, box.y0, box.width * (1 - shrinkage_factor), box.height]
            )

            # Put a legend to the right of the current axis
            if len(grouping) > 1:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            ax.grid(axis="x", linestyle="--", zorder=0)
            ax.set_xlabel("{} ({})".format(x_var, self.year))
            sns.despine()

            # save
            out_dir = out_dir.format(year=year)
            utils.ccdir(out_dir)

            plt.savefig(
                os.path.join(out_dir, "{}.png".format(x_var)),
                dpi=150,
                bbox_inches="tight",
            )

            plt.close(fig)

    # NOTE: occupations have the same GBN shares across countries, so this function does
    #   not make a whole lot of sense
    def create_occupation_barplots(self, n_occ="all"):
        logger = logging.getLogger(__name__)
        logger.info("GBN employment shares by occupation.")

        df_all_agg_by_cntr_occ = self.aggregate_by_occupation()

        y_var = self.fmt_string_isco_label.format(self.n_digits_isco08)
        hue_var = "COUNTRYW"

        n_colors = len(df_all_agg_by_cntr_occ[hue_var].unique())
        palette = sns.color_palette("cubehelix", n_colors=n_colors)

        for x_var in self.cols_to_plot:
            print("plotting variable: {}".format(x_var))

            # infer order (sorted by median in descending order)
            plotting_order = (
                df_all_agg_by_cntr_occ.loc[:, (y_var, x_var)]
                .groupby(y_var)
                .median()
                .sort_values(by=x_var, ascending=False)
                .index.values
            )

            # extract and reorder data
            plotting_data = (
                df_all_agg_by_cntr_occ.set_index([y_var, hue_var])
                .reindex(plotting_order, level=0)
                .reset_index()
            )

            if n_occ != "all":
                plotting_data = plotting_data.head(n_occ * len(self.countries))
                figsize = (10, n_occ / 2)
                x_min = 0
            else:
                figsize = (10, 20)
                x_min = None

            fig, ax = plt.subplots(figsize=figsize)

            # sns.boxplot(
            #     data=df_all_agg_by_cntr_occ,
            #     x=x_var,
            #     y=y_var,
            #     color=".5",
            #     fliersize=5,
            #     order=plotting_order,
            # )

            # sns.pointplot(
            #     data=df_all_agg_by_cntr_occ.set_index([y_var, hue_var])
            #         .reindex(plotting_order, level=0)
            #         .reset_index(),
            #     x=x_var,
            #     y=y_var,
            #     join=False,
            #     scale=0.5,
            #     estimator=np.median,
            # )

            sns.scatterplot(
                data=plotting_data,
                x=x_var,
                y=y_var,
                hue=hue_var,
                style=hue_var,
                palette=palette,
                zorder=5,
                edgecolor="black"
                # order=plotting_order,
            )

            # labelling

            # Shrink current axis by x %
            shrinkage_factor = 0.2
            box = ax.get_position()
            ax.set_position(
                [box.x0, box.y0, box.width * (1 - shrinkage_factor), box.height]
            )

            # Put a legend to the right of the current axis
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

            ax.grid(axis="x", linestyle="--", zorder=0)
            ax.set_xlabel("{} ({})".format(x_var, self.year))
            ax.set_xlim(xmin=x_min)
            sns.despine()

            # save
            out_dir = os.path.join(
                self.figure_dir,
                "03_eulfs",
                str(self.year),
                "employment_shares_by_occupation",
            )
            utils.ccdir(out_dir)

            plt.savefig(
                os.path.join(out_dir, "{}_n_{}.png".format(x_var, n_occ)),
                dpi=150,
                bbox_inches="tight",
            )

            plt.close(fig)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    config_paths = "paths_config.yml"
    config_data = "data_config.yml"
    config_vis = "vis_config.yml"

    # EU-LFS
    eulfs_visualiser = EulfsVis(
        fn_config_data=config_data,
        fn_config_path=config_paths,
        fn_vis_config=config_vis,
        year=2019,
    )

    # eulfs_visualiser.create_maps(slice_by_industry=True)
    eulfs_visualiser.create_industry_boxplots()

    # eulfs_visualiser.create_occupation_barplots(n_occ="all")
    # eulfs_visualiser.create_occupation_barplots(n_occ=10)
    # eulfs_visualiser.create_occupation_barplots(n_occ=20)
