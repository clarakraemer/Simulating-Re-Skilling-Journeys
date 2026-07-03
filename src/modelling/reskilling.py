import os
import zlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
import pickle

import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src import utils, plotting_utils
from data.framework import Esco
from data.lfs import EuLfs
from src.modelling import occupation_distance

# init objects used by classes below
esco = Esco()
useful_paths = utils.UsefulPaths()

# static parameters
bbox_eu_epsg_3035 = [2500000.0, 6000000.0, 1300000.0, 5500000.0]


class ReskillingPathways:
    """
    Class for simulating occupation transition pathways and re/upskilling interventions
    for EU-LFS labour force survey data.
    """

    def __init__(
        self, osm_version="weighted", sim_metric="cooc", lfs_data=None, year=2023
    ):
        """

        Parameters
        ----------
        osm_version : str
            Version of occupation skill matrix. Whether optional skills are weighted.
        sim_metric : str
            Version of occupation similarity metric. One of self.sim_metrics
        lfs_data : pd.DataFrame
            Input EU-LFS data.
        """

        # -----------------------------------------------------------------------------
        # Parameters
        # -----------------------------------------------------------------------------

        # load params from init
        self.osm_version = osm_version
        self.sim_metric = sim_metric
        self.lfs_data = lfs_data
        self.year = year

        # Correctness fix (revision): the centrality "not-yet-held" filter tracks skills
        # acquired during a worker's journey, so a skill already learned is never
        # re-offered. The former baseline-only behaviour (filter reads only the baseline
        # occupation-skill matrix) is a deviation from the intended model and is retained
        # behind this switch for the regression test and the SI robustness comparison.
        # Set rp.journey_aware = False to reproduce the pre-correction (baseline-only) runs.
        self.journey_aware = True

        # Task B (employment-share weighting of destinations). regC-only (only the
        # regional path carries the per-target NUTS-2 employment count). Shipping default
        # is the parameter-free "above-current wage floor": among targets paying at least
        # the worker's current wage -- the income-preference assumption already stated in
        # the paper -- draw by NUTS-2 employment share; fall back to all feasible if none
        # qualify. Chosen over a band rule (no unmotivated "why 25%?" parameter) and
        # validated on income-eligible countries (preserves the baseline income-loss
        # share while delivering ~the reviewer-requested employment-realism gain).
        #   "off"  -> top-income argmax (deterministic) == Phase-1            [disable switch]
        #   "share_income_acceptable" with acceptability=
        #        "above_current" (DEFAULT, parameter-free): share-draw among targets >= current wage
        #        "band"          (SI robustness):           share-draw within income_band of best income
        #   "share_only" -> share-draw, income ignored (upper bound, not shipped)
        self.destination_weighting = "share_income_acceptable"
        self.acceptability = "above_current"
        self.income_band = 0.25   # used only when acceptability=="band" (SI robustness sweep)
        self.share_seed = 42

        # phaseout scenario implementations
        #self.phaseout_scenarios = ["coal", "brown_techchange", "brown"] # OLD
        self.phaseout_scenarios = ["at_risk", "high_carbon", "shortage"] # NEW

        # coefficient weights for each scenario
        self.transition_pool_weights = {
            # "coal": "COEFFY",
            # "brown_techchange": "COEFFY_share_brown_slt",
            # "brown": "COEFFY_share_brown_sl",
            "at_risk": "COEFFY_share_unviable_to_decarbonize",
            "high_carbon": "COEFFY_share_high_carbon",
            "shortage": "COEFFY_share_shortage",
        }

        # criteria for coal phase-out scenario
        self.coal_occupations = {
            "Mining, manufacturing and construction supervisors": "312",
            "Mining and mineral processing plant operators": "811",
            "Manufacturing, mining, construction, and distribution managers": "132",
            "Mining and construction labourers": "931",
        }
        self.coal_industries = {
            "Electricity, Gas, Steam and Air Conditioning Supply": "D",
            "Mining and Quarrying": "B",
        }

        # occupation similarity thresholds
        self.trans_thresh = None
        self.trans_thresh_pc_approach = None

        # desirable target categories
        # self.target_cats = ["neutral", "green"] # OLD
        self.target_cats_at_risk = ["viable_to_decarbonize", "neutral", "low_carbon"]  # NEW
        self.target_cats_high_carbon = ["neutral", "low_carbon"] # NEW
        self.target_cats_shortage = ["low_carbon"]  # NEW

        # phase-out scenario specific list version for assigning GBN categories
        self.category_versions = {
            # TODO: discuss categories for coal case
            #"coal": "category_sl",
            #"brown_techchange": "category_slt",
            #"brown": "category_sl",
            "at_risk": "category",
            "high_carbon": "category",
            "shortage": "category",
        }

        # OSM versions
        self.osm_versions = ["weighted", "unweighted"]

        # occupation similarity metrics
        self.sim_metrics = ["cooc", "shortage_excess_avg", "shortage", "excess"]

        # whether a high or low occupation similarity score is better depends on metric
        self.best_choice = {
            "cooc": "max",
            "shortage_excess_avg": "min",
            "shortage": "min",
            "excess": "min",
        }

        # granularity levels in osim
        self.level_dict = {1: "esco_5_digit", 2: "isco_4_digit", 3: "isco_3_digit"}

        # mapping to threshold categories
        self.threshold_categories = {
            "esco_5_digit": "viable_high",
            "isco_4_digit": "viable",
            "isco_3_digit": "viable_low",
        }

        # avoid transitions to same occupation
        # note: actually this should always be set to False
        self.osim_diag_zeros = False

        self.dirname_out = "{}_{}_opt_{}"
        self.dirname_out_reg = "{sim_version}_{opt_target}-opt_{reg_constraint}_{year}"

        # possible simulation options and corresponding folder names
        self.simulation_name = {
            None: "baseline",
            "coreness_weighted": "reskill-coreWeight",
            "coreness_ranked": "reskill-coreRanked",
            "optimal": "reskill-optimal",
            "digital": "reskill-digital",
            "green": "reskill-green"
        }

        # checks
        assert self.sim_metric in self.sim_metrics
        assert self.osm_version in self.osm_versions

        # -----------------------------------------------------------------------------
        # DATA
        # -----------------------------------------------------------------------------
        # load similarity and annotate granularity levels (baseline version)
        self.df_occ_sim = occupation_distance.occ_sim_matrix_by_levels(
            sim_metric=self.sim_metric,
            osm_version=self.osm_version,
            diagonal_zeros=self.osim_diag_zeros,
            upskilling_ids=None,
        )

        # read bipartite adjacency matrix for occupations and skills
        self.occ_skills_mat = esco.read_occ_skills_matrix(
            return_version=self.osm_version
        )

        # create 3D version
        self.occ_skills_mat.index = occupation_distance.create_multiindex_for_esco_occs(
            esco.occupations
        )
        self.occ_skills_mat_3d = self.occ_skills_mat.groupby(level=3).mean()

        self.n_occs, self.n_skills = self.occ_skills_mat_3d.shape

        # Read geodata with NUTS regions
        self.gdf_all_levels = gpd.read_file(
            os.path.join(
                useful_paths.data_raw,
                "geodata",
                "NUTS_RG_03M_2024_3035",
                "NUTS_RG_03M_2024_3035.shp",
            )
        )
        self.gdf = self.gdf_all_levels[self.gdf_all_levels["LEVL_CODE"] == 2]

        # nace mapping
        # shorten NACE labels
        self.nace_labels = pd.read_csv(
            os.path.join(
                useful_paths.data_raw,
                "classifications",
                "NACE_REV2_1d_section_codes_short_names.csv",
            ),
            # index_col=0,
        )

        self.nace_mapping = dict(
            zip(
                self.nace_labels["NACE2_1D_label"], self.nace_labels["NACE2_1D_label_short"]
            )
        )

        ##########################
        #### Reskilling modes ####
        ##########################

        # 1) Optimal/ tailored reskilling: occupation-specific skill that unlocks most new transitions
        self.df_optimal_upskilling_per_occ = pd.read_csv(
            os.path.join(
                useful_paths.data_interim,
                "upskilling_analysis",
                "upskilling_best_100_skills_per_isco3d_occupation_merged.csv",
            ),
            index_col=0,
        )
        self.df_optimal_upskilling_per_occ.rename(
            columns={"ISCO3D_label": "ISCO08_3D_label"},
            inplace=True
        )

        # 2) Broad/ transferable reskilling: coreness measure of all ESCO skills
        self.df_coreness = pd.read_pickle(
            os.path.join(
                useful_paths.data_processed, "esco", "skills_network_metrics.pkl"
            )
        )

        # 3) Digital reskilling: random draws from digital skill list
        df_digital_skills = pd.read_csv(  #NEW
            os.path.join(
                useful_paths.data_raw,
                "esco",
                "v1.1.0",
                "digCompSkillsCollection_en.csv",
            ),
            index_col=0,
        )
        self.digital_skills = df_digital_skills["conceptUri"].tolist()  #NEW

        # 4. Green reskilling: random draws from green skill list
        df_green_skills = pd.read_csv(  #NEW
            os.path.join(
                useful_paths.data_raw,
                "esco",
                "v1.1.0",
                "greenSkillsCollection_en.csv",
            ),
            index_col=0,
        )
        self.green_skills = df_green_skills["conceptUri"].tolist()  # NEW

        #NEW
        print(
            "→ Digital URIs:", len(self.digital_skills),
            "| present in matrix:",
            sum(u in self.occ_skills_mat_3d.columns for u in self.digital_skills)
        )
        print(
            "→ Green   URIs:", len(self.green_skills),
            "| present in matrix:",
            sum(u in self.occ_skills_mat_3d.columns for u in self.green_skills)
        )

        # based on full, unaggregated matrix
        self.trans_thresh_pc_approach = np.percentile(
            self.df_occ_sim.values.flatten(), q=(96, 99)
        )


    def get_occs(self, level="isco_3_digit", lfs_country_subset=None):
        """
        Obtain list of occupations for given ISCO granularity level that are available in
        the occupation similarity matrix.

        Parameters
        ----------
        level : str
            ISCO occupation granularity level. One of (isco_3_digit, isco_4_digit).
        lfs_country_subset : pd.DataFrame
            Note: currently only works for ISCO 3D data
        Returns
        -------
        pd.DataFrame
            List of available occupations with index numbers matching those of the
            occ sim matrix.
        """
        lvl_code = utils.reverse_dict(self.level_dict)[level]
        ndigs = {"isco_4_digit": 4, "isco_3_digit": 3}
        occs_all = esco.isco_groups[
            esco.isco_groups.code.str.len() == ndigs[level]
        ].loc[:, ("conceptUri", "preferredLabel", "code")]
        occs_available = (
            self.df_occ_sim.index.get_level_values(lvl_code)
            .unique()
            .sort_values()
            .values
        )

        # subset
        occs_at_level = occs_all[occs_all.code.isin(occs_available)].reset_index(
            drop=True
        )

        # optionally join country-level means from lfs data set
        if lfs_country_subset is not None:
            agg_dict = {
                #"share_green": np.nanmean,
                #"share_brown_sl": np.nanmean,
                #"share_brown_slt": np.nanmean,
                #"share_neutral_sl": np.nanmean,
                #"share_neutral_slt": np.nanmean,
                "share_viable_to_decarbonize": "mean",
                "share_unviable_to_decarbonize": "mean",
                "share_high_carbon": "mean",
                "share_neutral": "mean",
                "share_low_carbon": "mean",
                "INCDECIL_imputed": "median",  # keep median as in original code
                "annual_earnings": "mean",
            }

            isco_avg = ( # Resolves panda warnings
                lfs_country_subset
                .groupby(["ISCO", "ISCO08_3D_label"], as_index=False)
                .agg(agg_dict)
            )

            cat_cols = [
                "share_low_carbon",
                "share_viable_to_decarbonize",
                "share_unviable_to_decarbonize",
                "share_neutral",
            ]
            isco_avg["category"] = (
                isco_avg[cat_cols]
                .idxmax(axis=1)
                .replace({
                    "share_low_carbon": "low_carbon",
                    "share_viable_to_decarbonize": "viable_to_decarbonize",
                    "share_unviable_to_decarbonize": "unviable_to_decarbonize",
                    "share_neutral": "neutral",
                })
            )

            # join
            occs_at_level["code"] = occs_at_level["code"].astype(str)
            isco_avg["ISCO"] = isco_avg["ISCO"].astype(str)

            occs_at_level = pd.merge(
                occs_at_level, isco_avg, left_on="code", right_on="ISCO", how="left"
            )

        return occs_at_level

    def sim_matrix_at_level(
        self,
        level="isco_3_digit",
        agg_func="mean",
        upskilling_ids=None,
        occ_skills_mat=None,
        mask_diagonal=True,
    ):
        """
        Average the original occupation similarity matrix at a specific occupation
        group level.

        Note: first aggregating the occupation-skills matrix at a specific level and
         then calculating the co-occurrence yields the same result.

        Parameters
        ----------
        level : str
            Occupation granularity level. One of self.level_dict.values()
        agg_func : str or np.func
            Function to be used for aggregating occupation similarities
        upskilling_ids
        occ_skills_mat
        mask_diagonal

        Returns
        -------
        sim_matrix_agg : pd.DataFrame
            Aggregated occupation similarity matrix.
        """
        lvl_code = utils.reverse_dict(self.level_dict)[level]

        if upskilling_ids is None:
            df_occ_sim = self.df_occ_sim
        else:
            df_occ_sim = occupation_distance.occ_sim_matrix_by_levels(
                occ_skills_mat=occ_skills_mat,
                osm_version=self.osm_version,
                sim_metric=self.sim_metric,
                upskilling_ids=upskilling_ids,
            )

        df_sim_matrix_agg = (
            df_occ_sim.groupby(level=lvl_code)
            .aggregate(agg_func)
            .T
            .groupby(level=lvl_code)
            .aggregate(agg_func)
        )

        if mask_diagonal:
            sim_matrix_agg = df_sim_matrix_agg.values
            np.fill_diagonal(sim_matrix_agg, 0)
            df_sim_matrix_agg = pd.DataFrame(
                data=sim_matrix_agg,
                index=df_sim_matrix_agg.index,
                columns=df_sim_matrix_agg.columns,
            )

        return df_sim_matrix_agg

    def plot_sim_matrix_at_level(
        self,
        level="isco_3_digit",
        agg_func="mean",
        classify=False,
        vmax=10,
        cmap_type="Blues",
        over_vmax_color="pink",
        mask_upper=True,
        save_fig=True,
        out_dir=os.path.join(useful_paths.figure_dir, "reskilling_simulation"),
    ):
        """
        Plot numeric or classified occupation similarity matrix for a given granularity
        level.

        Parameters
        ----------
        level : str
            Occupation granularity level.
        agg_func : str
            Aggregation function
        classify : bool
            Whether to classifiy numeric matrix into non-viable/viable/highly viable
        vmax : int
            Max value to show and number of colors in cbar.
        cmap_type : str
            Colormap
        over_vmax_color : str
            Color of tip if cbar is extended.
        mask_upper : bool
            Whether to mask the upper triangle.
        save_fig : bool
            Whether to save the plot.
        out_dir : os.Path
            Path to output directory.

        Returns
        -------
        ax : mpl ax object
            Figure axis.
        """

        # read mat
        sim_mat = self.sim_matrix_at_level(level=level, agg_func=agg_func).values
        if mask_upper:
            sim_mat = np.tril(sim_mat)

        if classify:
            q_viable, q_highly_viable = self.trans_thresh_pc_approach

            sim_mat[sim_mat < q_viable] = 0
            sim_mat[(sim_mat >= q_viable) & (sim_mat < q_highly_viable)] = 1
            sim_mat[sim_mat >= q_highly_viable] = 2

        # define cmap
        if not classify:
            cmap = plotting_utils.discrete_cmap_with_manual_colors(cmap_type, vmax)
            cmap.set_over(over_vmax_color)
            norm = None
            label = "Skills {} [-]".format(self.sim_metric.capitalize())
            extend = "max"
        else:
            cmap = colors.ListedColormap(["white", "lightblue", "darkblue"])
            bounds = [0, 1, 2, 3]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            vmax = None
            label = "Transition viability"
            extend = None

        # plot
        fig, ax = plt.subplots()
        im = ax.imshow(sim_mat, cmap=cmap, vmax=vmax, norm=norm)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical", label=label, extend=extend)

        ax.set_ylabel("Source occupations ({})".format(level))
        ax.set_xlabel("Target occupations ({})".format(level))
        sns.despine()

        if save_fig:
            out_folder = "{}_metric_thresh_cal_diagzero_{}".format(
                self.sim_metric, self.osim_diag_zeros
            )
            utils.ccdir(os.path.join(out_dir, out_folder))

            plt.savefig(
                os.path.join(
                    out_dir,
                    out_folder,
                    "occ_sim_{}_classified_{}.png".format(level, classify),
                ),
                dpi=300,
            )

        return ax

    def calc_sim_means_by_level(self):
        """
        Calculate mean and sd of group-wise occupation similarity scores for different
        granularity levels.

        Returns
        -------
        df_sim_container : pd.DataFrame
            Mean and sd of occ_sim grouped by level and occupation
        """

        # iterate over groups
        sim_container = {}

        for level in [1, 2, 3]:
            group_res = {}
            for group in self.df_occ_sim.index.get_level_values(level).values:
                # subset row
                row_subset = self.df_occ_sim.loc[
                    self.df_occ_sim.index.get_level_values(level) == group
                ]

                # subset col
                combined_subset = row_subset.loc[
                    :, row_subset.columns.get_level_values(level) == group
                ].values.flatten()

                # stats
                mean = combined_subset.mean()
                sd = combined_subset.std()
                n = int(np.sqrt(len(combined_subset)))

                # append
                group_res[group] = [mean, sd, n]

            # append
            sim_container[level] = group_res

        # to dfs
        df_sim_container = pd.concat(
            {k: pd.DataFrame(v).T for k, v in sim_container.items()}, axis=0
        ).reset_index()

        df_sim_container = df_sim_container.rename(
            columns={
                "level_0": "level",
                "level_1": "occupation",
                0: "sim_mean",
                1: "sim_sd",
                2: "n",
            }
        )

        # rename levels
        df_sim_container = df_sim_container.replace(
            to_replace={"level": self.level_dict}
        )

        return df_sim_container

    def calc_transition_thresholds(
        self,
        save_plots_by_level=True,
        save_plot_overview=True,
        show_plots=True,
        save_data=True,
        out_dir=os.path.join(useful_paths.figure_dir, "reskilling_simulation"),
    ):
        """
        Calculate average transition numbers within different occupation granularity
        levels. This is a crucial parameter for the calibration of a threshold
        constraining occupation transition & reskilling simulations.

        Parameters
        ----------
        save_plots_by_level : bool
            Whether plots by granularity level should be saved.
        save_plot_overview : bool
            Whether overview plot should be saved.
        show_plots : bool
            Whether plots should be shown.
        save_data : bool
            Whether numeric data should be stored
        out_dir : os.Path
            Path to output directory.

        Returns
        -------
        df_within_group_results : pd.DataFrame
            Mean and sd of within-group occupation similarities.
        """

        # load data
        df_sim_container = self.calc_sim_means_by_level()

        # create output folder
        if save_plots_by_level or save_data:
            out_folder = "{}_metric_thresh_cal_diagzero_{}".format(
                self.sim_metric, self.osim_diag_zeros
            )
            utils.ccdir(os.path.join(out_dir, out_folder))

        within_group_results = {}
        for level in list(self.level_dict.values()):
            plot_data = df_sim_container[df_sim_container.level == level]

            # calc stats
            within_group_mean = plot_data.sim_mean.mean()
            within_group_sd = plot_data.sim_mean.std()
            within_group_results[level] = [
                within_group_mean,
                within_group_sd,
                within_group_mean - within_group_sd,
                within_group_mean + within_group_sd,
            ]

            if save_plots_by_level:
                # plot
                fig = plt.figure()
                plt.hist(plot_data.sim_mean, bins=20, color="lightgrey")

                plt.axvline(within_group_mean, linestyle="--", zorder=1)
                plt.text(
                    within_group_mean,
                    0,
                    "$\mu = {:.2f}$".format(within_group_mean),
                    rotation=0,
                    va="bottom",
                    ha="left",
                    fontsize=8,
                )

                plt.grid(linestyle=":")
                plt.xlabel("Skills {} [-]".format(self.sim_metric))
                plt.ylabel("Number of {} groups [-]".format(level))

                sns.despine()
                plt.tight_layout()

                plt.savefig(
                    os.path.join(out_dir, out_folder, "{}.png".format(level)),
                    dpi=300,
                )

            if not show_plots:
                plt.cla()
                plt.clf()

        # combine and save means & sd
        df_within_group_results = pd.DataFrame(
            data=within_group_results, index=["mean", "sd", "mean-sd", "mean+sd"]
        ).T

        # annotate categories
        df_within_group_results["threshold_category"] = [
            self.threshold_categories[lvl]
            for lvl in df_within_group_results.index.values
        ]

        if save_data:
            df_within_group_results.to_csv(
                os.path.join(
                    out_dir,
                    out_folder,
                    "within_group_similarities.csv",
                )
            )

        if save_plot_overview:
            # overview
            plt.figure()
            plt.hist(
                self.df_occ_sim.values.flatten(), bins=50, log="y", color="lightgrey"
            )

            plt.grid(linestyle=":")
            plt.xlabel("Skills overlap [-]")
            plt.ylabel("Number of occupations, logged [-]")

            # annotate
            viable = df_within_group_results.loc["isco_4_digit", "mean-sd"]
            plt.axvline(viable, linestyle="--", zorder=1, color="blue")
            plt.text(
                viable,
                10e6,
                "viable = {:.1f}".format(viable),
                rotation=0,
                va="top",
                ha="left",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="blue", alpha=1),
            )

            highly_viable = df_within_group_results.loc["esco_5_digit", "mean-sd"]
            plt.axvline(highly_viable, linestyle="--", zorder=1, color="darkblue")
            plt.text(
                highly_viable,
                10e5,
                "highly viable = {:.1f}".format(highly_viable),
                rotation=0,
                va="top",
                ha="left",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="darkblue", alpha=1),
            )

            sns.despine()
            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    out_dir,
                    "{}_metric_thresh_cal_diagzero_{}".format(
                        self.sim_metric, self.osim_diag_zeros
                    ),
                    "overview_of_thresholds.png",
                ),
                dpi=300,
            )

            if not show_plots:
                plt.cla()
                plt.clf()

        return df_within_group_results

    def define_transition_pool(self, scenario, country):
        """
        Extract a subset of the LFS data based on phase-out scenario and country.

        Parameters
        ----------
        scenario : str
            Phase-out scenario
        country : str
            Country code (ISO)

        Returns
        -------
        transition_pool : pd.DataFrame
            Subset of LFS data specific to scenario and country.
        """
        transition_pool = None

        # subset LFS data
        lfs_data_country = self.lfs_data[self.lfs_data["COUNTRYW"] == country]

        # get averages
        isco_grp_avg = self.get_occs(
            level="isco_3_digit", lfs_country_subset=lfs_data_country
        )

        means = {
            "share_low_carbon": isco_grp_avg["share_low_carbon"].mean(),
            "share_viable_to_decarbonize": isco_grp_avg["share_viable_to_decarbonize"].mean(),
            "share_unviable_to_decarbonize": isco_grp_avg["share_unviable_to_decarbonize"].mean(),
            "share_neutral": isco_grp_avg["share_neutral"].mean(),
        }

        # job pools
        # 1) at‐risk: unviable‐to‐decarbonize above average
        if scenario == "at_risk":
            transition_pool = lfs_data_country.loc[
                lfs_data_country["share_unviable_to_decarbonize"]
                > isco_grp_avg["share_unviable_to_decarbonize"].mean()
                ]

        # 2) high‐carbon: viable OR unviable above their averages
        elif scenario == "high_carbon":
            thr_v = isco_grp_avg["share_viable_to_decarbonize"].mean()
            thr_u = isco_grp_avg["share_unviable_to_decarbonize"].mean()
            transition_pool = lfs_data_country.loc[
                (lfs_data_country["share_viable_to_decarbonize"] > thr_v)
                | (lfs_data_country["share_unviable_to_decarbonize"] > thr_u)
                ]

        # 3) shortage: both high‐carbon AND neutral above their averages
        elif scenario == "shortage":
            thr_hc = isco_grp_avg["share_high_carbon"].mean()
            thr_n = isco_grp_avg["share_neutral"].mean()
            transition_pool = lfs_data_country.loc[
                (lfs_data_country["share_high_carbon"] > thr_hc)
                & (lfs_data_country["share_neutral"] > thr_n)
                ]

        else:
            raise NotImplementedError()

        # Combine COEFFY_share_shortage
        if "COEFFY_share_neutral" in lfs_data_country.columns and "COEFFY_share_high_carbon" in lfs_data_country.columns:
            # compute on the transition_pool slice (avoid touching the original lfs_data_country)
            transition_pool["COEFFY_share_shortage"] = (
                    transition_pool.get("COEFFY_share_neutral", 0).fillna(0)
                    + transition_pool.get("COEFFY_share_high_carbon", 0).fillna(0)
            )
        else:
            raise NotImplementedError()

        return transition_pool

    def jobs_by_country_and_region(self):
        """
       Calculate the number of jobs per country, region and occupation.

        Returns
        -------
        jobs_by_regions_countries : pd.DataFrame
            Dataframe where employment numbers are grouped by country (ISO),
            region (NUTS2) and occupation (ISCO 3D).
        """
        occ_number_by_nuts2_and_isco3 = self.lfs_data.groupby(
            ["NUTS_ID", "ISCO08_3D"]
        )["COEFFY"].sum().reset_index()

        # beautify
        jobs_by_regions_countries = occ_number_by_nuts2_and_isco3.reset_index()
        jobs_by_regions_countries["COUNTRYW"] = jobs_by_regions_countries[
            "NUTS_ID"
        ].str.slice(0, 2)

        return jobs_by_regions_countries


    def simulate(
        self,
        level="isco_3_digit",
        countries=None,
        scenarios=None,
        reskilling=None,
        reskilling_journey_length=1,
        transition_optimisation="wage",
        mask_diagonal=True,
        transition_thresholds=None,
        threshold="viable",
        verbose=False,
        out_dir=os.path.join(useful_paths.figure_dir, "reskilling_simulation"),
    ):
        """
        Simulate occupation transitions for given phase-out scenarios without
        regional constraints (baseline or with reskilling).

        Parameters
        ----------
        level : str
            Granularity level
        countries : list of str
            Countries to rerun_simulations.
        scenarios : list of str
            Phase-out scenarios to rerun_simulations.
        reskilling : str
            Mode of reskilling (See documentation of self.reskill). One of
            None, "coreness_weighted","optimal"
            ].
        transition_optimisation : str
            Which variable to optimise in the simulation of occupation transitions.
            Either 'wage' (minimise wage loss) or 'skill' (maximise skill overlap).
        out_dir : os.Path
            Path to output directory

        Returns
        -------
        simulation_results : dict of dicts of pd.DataFrame's
            Nested dictionary containing simulation results by (1) scenario and (2)
            country. Simulation results are stored as pd.DataFrame and contain worker-
            level information about: number of viable transitions, target occupation,
            earnings changes, etc.
        """
        # define transition pool based on scenario
        if scenarios is None:
            scenarios = self.phaseout_scenarios
        if countries is None:
            countries = ["DE"]

        # Always load baseline similarity matrix (for step=0 and beyond)
        similarity_matrix = self.sim_matrix_at_level(
            level=level, mask_diagonal=mask_diagonal
        ).values

        # define transition thresholds
        if transition_thresholds is None:
            q_viable, q_highly_viable = self.trans_thresh_pc_approach
        else:
            q_viable, q_highly_viable = transition_thresholds
        print("Viability thresholds:", q_viable, q_highly_viable)

        # create output dir and fnames
        dirname = self.dirname_out.format(
            self.simulation_name[reskilling], transition_optimisation, self.year
        )
        fname = "{}.pkl".format(dirname)
        target_dir = os.path.join(out_dir, dirname)
        utils.ccdir(target_dir)

        # -----------------------------------------------------------------------------
        # Scenario loop
        # -----------------------------------------------------------------------------
        simulation_results = {}
        for scenario in scenarios:
            print("scenario: {}".format(scenario))
            # select scenario-specific weighting coefficient
            coeffy_weight = self.transition_pool_weights[scenario]

            # scenario-specific categories of occupations
            category_version = self.category_versions[scenario]

            # -------------------------------------------------------------------------
            # Country loop
            # -------------------------------------------------------------------------
            scenario_results = {}
            for country in tqdm(countries):
                if verbose:
                    print("  country: {}".format(country))

                # select scenario- and country-specific transition pool
                df_transition_pool = self.define_transition_pool(
                    scenario=scenario, country=country
                )

                # Ensure coeffy is numeric only, no ceil in simulate (ceil is only used in simulate_regional)
                coeffy = coeffy_weight  # e.g. "COEFFY_share_unviable_to_decarbonize"
                if coeffy not in df_transition_pool.columns:
                    df_transition_pool[coeffy] = 0
                df_transition_pool = df_transition_pool.copy()  # avoid SettingWithCopyWarning
                df_transition_pool.loc[:, coeffy] = pd.to_numeric(
                    df_transition_pool.loc[:, coeffy], errors="coerce"
                ).fillna(0.0)

                # Fail early if there are negative values (data bug)
                if (df_transition_pool[coeffy] < 0).any():
                    sample_bad = df_transition_pool.loc[
                        df_transition_pool[coeffy] < 0, ["COUNTRYW", "NACE2_1D", coeffy]].head(10)
                    raise ValueError(
                        f"Negative values found in {coeffy} for {country} in simulate(); sample:\n{sample_bad}")

                # impute missing region of work with region of home & update NUTS codes
                df_transition_pool.loc[
                    df_transition_pool["REGION_2DW"].isna(), "NUTS_ID"
                ] = df_transition_pool.loc[
                    df_transition_pool["REGION_2DW"].isna(), "COUNTRYW"
                ].astype(
                    str
                ) + df_transition_pool.loc[
                    df_transition_pool["REGION_2DW"].isna(), "REGION_2D"
                ].astype(
                    str
                )

                # read occupation list coherent with sim matrix and enriched by means
                # across several LFS variables
                df_occs = self.get_occs(
                    level=level,
                    lfs_country_subset=self.lfs_data[
                        self.lfs_data["COUNTRYW"] == country
                    ],
                )

                # Country-level job availability (reuse regional helper)
                jobs_country = (
                    self.jobs_by_country_and_region()
                    .query("COUNTRYW == @country")
                    .groupby("ISCO08_3D")["COEFFY"].sum()
                )
                available_occs = jobs_country[jobs_country > 0].index.astype(str)

                # Pre-compute country mean shares (same cols as simulate_regional)
                share_cols = [
                    "share_viable_to_decarbonize",
                    "share_unviable_to_decarbonize",
                    "share_high_carbon",
                    "share_neutral",
                    "share_low_carbon",
                ]
                country_share_means = {c: df_occs[c].mean() for c in share_cols}

                # populate transition pool dict
                transition_pool_dict = {}
                for i, s in df_transition_pool.iterrows():
                    transition_pool_dict[s.ISCO08_3D_label] = s.ISCO08_3D


                # ---------------------------------------------------------------------
                # Worker-level transition simulation
                # ---------------------------------------------------------------------
                transition_number_data = []
                if verbose:
                    print(
                        "    transition pool: {} workers (n={})".format(
                            df_transition_pool[coeffy_weight].sum(),
                            df_transition_pool.shape[0],
                        )
                    )

                # One clean baseline per worker similar to simulate_regional
                rng = np.random.RandomState(42)
                df_transition_pool_shuffled = df_transition_pool.sample(frac=1, random_state=rng).reset_index(drop=True)

                for i, search_obs in df_transition_pool_shuffled.iterrows():
                    occ_skills_mat = self.occ_skills_mat_3d.copy()
                    added_skill = None

                    # local sim matrix (keeps global baseline untouched)
                    sim_mat = similarity_matrix

                    # loop cumulatively through reskilling steps
                    for step in range(0, reskilling_journey_length + 1):
                        search_obs["reskilling_step"] = step
                        search_label = search_obs.ISCO08_3D_label
                        idx_occ = df_occs.loc[
                            df_occs["preferredLabel"] == search_label
                            ].index.values[0]

                        # perform upskilling for this worker & step
                        if step > 0 and reskilling is not None:
                            sim_mat, occ_skills_mat, added_skill = self.reskill( #NEW: sim_mat
                                occ_skills_mat=occ_skills_mat,
                                idx_occ=idx_occ,
                                search_label=search_label,
                                reskilling_mode=reskilling,
                                mask_diagonal=mask_diagonal,
                                skill_rank=step,
                                prev_sim_mat=sim_mat,  # incremental M_oo update
                            )

                        # stamp onto this search_obs row
                        if reskilling == "optimal" and added_skill is not None:
                            search_obs[f"added_skill_idx_step_{step}"] = added_skill
                            sel = self.df_optimal_upskilling_per_occ[
                                (self.df_optimal_upskilling_per_occ["ISCO08_3D_label"] == search_label)
                                & (self.df_optimal_upskilling_per_occ["idx_skill"] == added_skill)
                                ]
                            search_obs[f"added_skill_label_step_{step}"] = sel["skill_label"].iat[0]

                        # find closest target occupations
                        target_occs = occupation_distance.find_closest(
                            i=idx_occ, similarity_matrix=sim_mat, df=df_occs
                        )

                        # VIABILITY THRESHOLD
                        target_occs_filtered = target_occs.loc[
                            target_occs["similarity"] > q_viable
                        ]
                        target_occs_filtered_hv = target_occs.loc[
                            target_occs["similarity"] > q_highly_viable
                        ]

                        # drop occupations not present in this country
                        target_occs_filtered = target_occs_filtered.loc[
                            target_occs_filtered["code"].astype(str).isin(available_occs)
                        ]
                        target_occs_filtered_hv = target_occs_filtered_hv.loc[
                            target_occs_filtered_hv["code"].astype(str).isin(available_occs)
                        ]

                        # apply the same share‑filter map from simulate_regional
                        share_filter_by_scenario = {
                            "at_risk": ["share_unviable_to_decarbonize", "share_neutral", "share_low_carbon"],
                            "high_carbon": ["share_neutral", "share_low_carbon"],
                            "shortage": ["share_low_carbon"],
                        }
                        cols_to_check = share_filter_by_scenario[scenario]
                        mask = False
                        for sc in cols_to_check:
                            mask |= (target_occs_filtered[sc] > country_share_means[sc])
                        target_occs_filtered = target_occs_filtered.loc[mask]
                        mask_hv = False
                        for sc in cols_to_check:
                            mask_hv |= (target_occs_filtered_hv[sc] > country_share_means[sc])
                        target_occs_filtered_hv = target_occs_filtered_hv.loc[mask_hv]

                        # earnings delta to next closest occupation
                        target_occs_filtered["wage_diff"] = (
                            target_occs_filtered["annual_earnings"]
                            - search_obs["annual_earnings"]
                        )
                        target_occs_filtered_hv["wage_diff"] = (
                            target_occs_filtered_hv["annual_earnings"]
                            - search_obs["annual_earnings"]
                        )

                        # if I know the wage of the source occupation, choose transition
                        # that minimises wage losses. else, choose transition with highest
                        # target wage.
                        if transition_optimisation == "wage":
                            #if search_obs["annual_earnings"] != np.nan: #OLD
                            if pd.notna(search_obs["annual_earnings"]): #NEW
                                target = target_occs_filtered.sort_values("wage_diff").tail(
                                    1
                                )
                                target_hv = target_occs_filtered_hv.sort_values(
                                    "wage_diff"
                                ).tail(1)
                            else:
                                target = target_occs_filtered.sort_values(
                                    "annual_earnings"
                                ).tail(1)
                                target_hv = target_occs_filtered_hv.sort_values(
                                    "annual_earnings"
                                ).tail(1)
                        # choose occupation with highest skills overlap
                        elif transition_optimisation == "skill":
                            target = target_occs_filtered.sort_values("similarity").tail(1)
                            target_hv = target_occs_filtered_hv.sort_values(
                                "similarity"
                            ).tail(1)
                        else:
                            raise NotImplementedError()

                        # check if viable transition exists
                        # compute base delta
                        if not target.empty:
                            base_delta = target["annual_earnings"].values[0] - search_obs["annual_earnings"]

                            # step-suffixed columns
                            search_obs[f"earnings_delta_closest_switch_step_{step}"] = base_delta
                            search_obs[f"earnings_delta_closest_switch_sum_step_{step}"] = (
                                    base_delta * search_obs[coeffy_weight]
                            )
                            search_obs[f"earnings_delta_closest_switch_sum_mio_step_{step}"] = (
                                    search_obs[f"earnings_delta_closest_switch_sum_step_{step}"] / 1e6
                            )

                            # ADD: per-worker % change (mean-aggregated later)
                            if pd.notna(search_obs["annual_earnings"]) and search_obs["annual_earnings"] != 0:
                                search_obs[f"earnings_delta_closest_switch_pct_step_{step}"] = (
                                        base_delta / search_obs["annual_earnings"]
                                )
                            else:
                                search_obs[f"earnings_delta_closest_switch_pct_step_{step}"] = np.nan
                            search_obs[f"n_viable_transitions_step_{step}"] = target_occs_filtered.shape[0]
                            search_obs[f"n_viable_transitions_sum_step_{step}"] = (
                                    target_occs_filtered.shape[0] * search_obs[coeffy_weight]
                            )

                            search_obs[f"transition_viable_step_{step}"] = True
                            search_obs[f"transition_target_step_{step}"] = target["preferredLabel"].values[0]
                            # transition_target_code not used in simulate but present in simulate_regional; include for parity
                            search_obs[f"transition_target_code_step_{step}"] = target["code"].values[0]

                            # target category (same logic as before)
                            share_cols = [
                                "share_low_carbon",
                                "share_viable_to_decarbonize",
                                "share_unviable_to_decarbonize",
                                "share_neutral",
                            ]
                            share_labels = [
                                "low_carbon",
                                "viable_to_decarbonize",
                                "unviable_to_decarbonize",
                                "neutral",
                            ]
                            if all(col in target.columns for col in share_cols):
                                target_shares = target[share_cols].values[0]
                                tgt_cat = share_labels[np.argmax(target_shares)]
                                search_obs[f"target_category_step_{step}"] = tgt_cat
                            else:
                                search_obs[f"target_category_step_{step}"] = None

                        else:
                            # No viable (non-HV) target exists
                            if scenario == "shortage": # Shortage scenario: worker stays → keep wage (Δ = 0)
                                base_delta = 0.0  # keep wage
                            else:
                                base_delta = -1 * search_obs["annual_earnings"]  # full loss

                                # BASE per-worker delta & totals
                            search_obs[f"earnings_delta_closest_switch_step_{step}"] = base_delta
                            search_obs[f"earnings_delta_closest_switch_sum_step_{step}"] = (
                                    base_delta * search_obs[coeffy_weight]
                            )
                            search_obs[f"earnings_delta_closest_switch_sum_mio_step_{step}"] = (
                                    search_obs[f"earnings_delta_closest_switch_sum_step_{step}"] / 1e6
                            )
                            if pd.notna(search_obs["annual_earnings"]) and search_obs["annual_earnings"] != 0:
                                search_obs[f"earnings_delta_closest_switch_pct_step_{step}"] = (
                                        base_delta / search_obs["annual_earnings"]
                                )
                            else:
                                search_obs[f"earnings_delta_closest_switch_pct_step_{step}"] = np.nan

                            # Flags
                            search_obs[f"n_viable_transitions_step_{step}"] = 0
                            search_obs[f"n_viable_transitions_sum_step_{step}"] = 0
                            search_obs[f"transition_viable_step_{step}"] = False
                            search_obs[f"transition_target_step_{step}"] = None
                            search_obs[f"transition_target_code_step_{step}"] = None
                            search_obs[f"target_category_step_{step}"] = None

                            # (optional) also stamp the HV fields for diagnostics
                            base_delta_hv = base_delta
                            search_obs[f"earnings_delta_closest_switch_hv_step_{step}"] = base_delta_hv
                            search_obs[f"earnings_delta_closest_switch_sum_hv_step_{step}"] = (
                                    base_delta_hv * search_obs[coeffy_weight]
                            )
                            search_obs[f"earnings_delta_closest_switch_sum_hv_mio_step_{step}"] = (
                                    search_obs[f"earnings_delta_closest_switch_sum_hv_step_{step}"] / 1e6
                            )
                            if pd.notna(search_obs["annual_earnings"]) and search_obs["annual_earnings"] != 0:
                                search_obs[f"earnings_delta_closest_switch_hv_pct_step_{step}"] = (
                                        base_delta_hv / search_obs["annual_earnings"]
                                )
                            else:
                                search_obs[f"earnings_delta_closest_switch_hv_pct_step_{step}"] = np.nan
                            search_obs[f"n_hv_transitions_step_{step}"] = 0
                            search_obs[f"n_hv_transitions_sum_step_{step}"] = 0
                            search_obs[f"transition_hv_step_{step}"] = False
                            search_obs[f"transition_target_hv_step_{step}"] = None
                            search_obs[f"target_category_hv_step_{step}"] = None

                        if not target_hv.empty:
                            # Every worker transitions to target job based on switching
                            # logic. We evaluate the wage difference.
                            # wage diff to highly viable target
                            base_delta_hv = target_hv["annual_earnings"].values[0] - search_obs["annual_earnings"]

                            # per-worker delta (HV)
                            search_obs[f"earnings_delta_closest_switch_hv_step_{step}"] = base_delta_hv

                            # totals (HV)
                            search_obs[f"earnings_delta_closest_switch_sum_hv_step_{step}"] = (
                                    base_delta_hv * search_obs[coeffy_weight]
                            )
                            # per-million (HV)
                            search_obs[f"earnings_delta_closest_switch_sum_hv_mio_step_{step}"] = (
                                    search_obs[f"earnings_delta_closest_switch_sum_hv_step_{step}"] / 1e6
                            )

                            # % change (HV)
                            if pd.notna(search_obs["annual_earnings"]) and search_obs["annual_earnings"] != 0:
                                search_obs[f"earnings_delta_closest_switch_hv_pct_step_{step}"] = (
                                        base_delta_hv / search_obs["annual_earnings"]
                                )
                            else:
                                search_obs[f"earnings_delta_closest_switch_hv_pct_step_{step}"] = np.nan

                            # mirror HV into BASE so downstream plots see the realized outcome
                            search_obs[f"earnings_delta_closest_switch_step_{step}"] = base_delta_hv
                            search_obs[f"earnings_delta_closest_switch_sum_step_{step}"] = (
                                    base_delta_hv * search_obs[coeffy_weight]
                            )
                            search_obs[f"earnings_delta_closest_switch_sum_mio_step_{step}"] = (
                                    search_obs[f"earnings_delta_closest_switch_sum_step_{step}"] / 1e6
                            )
                            if pd.notna(search_obs["annual_earnings"]) and search_obs["annual_earnings"] != 0:
                                search_obs[f"earnings_delta_closest_switch_pct_step_{step}"] = (
                                        base_delta_hv / search_obs["annual_earnings"]
                                )
                            else:
                                search_obs[f"earnings_delta_closest_switch_pct_step_{step}"] = np.nan

                            search_obs[f"n_hv_transitions_step_{step}"] = target_occs_filtered_hv.shape[0]
                            search_obs[f"n_hv_transitions_sum_step_{step}"] = (
                                    target_occs_filtered_hv.shape[0] * search_obs[coeffy_weight]
                            )
                            search_obs[f"transition_hv_step_{step}"] = True
                            search_obs[f"transition_target_hv_step_{step}"] = target_hv["preferredLabel"].values[0]
                            search_obs[f"transition_target_code_hv_step_{step}"] = target_hv["code"].values[0]

                            # target_category_hv logic (same as before) into step-suffixed
                            if all(col in target_hv.columns for col in share_cols):
                                target_shares_hv = target_hv[share_cols].values[0]
                                search_obs[f"target_category_hv_step_{step}"] = share_labels[np.argmax(target_shares_hv)]
                            else:
                                search_obs[f"target_category_hv_step_{step}"] = None

                        else:
                            # No highly-viable target exists
                            if scenario == "shortage":
                                # Shortage scenario: worker stays → keep wage (Δ = 0)
                                base_delta = 0.0
                                search_obs[f"earnings_delta_closest_switch_step_{step}"] = base_delta
                                search_obs[f"earnings_delta_closest_switch_sum_step_{step}"] = 0.0
                                search_obs[f"earnings_delta_closest_switch_sum_mio_step_{step}"] = 0.0

                                # percentage change also 0 (if earnings known), else NaN
                                if pd.notna(search_obs["annual_earnings"]) and search_obs["annual_earnings"] != 0:
                                    search_obs[f"earnings_delta_closest_switch_pct_step_{step}"] = 0.0
                                else:
                                    search_obs[f"earnings_delta_closest_switch_pct_step_{step}"] = np.nan

                                search_obs[f"n_viable_transitions_step_{step}"] = 0
                                search_obs[f"n_viable_transitions_sum_step_{step}"] = 0
                                search_obs[f"transition_viable_step_{step}"] = False
                                search_obs[f"transition_target_step_{step}"] = None
                                search_obs[f"transition_target_code_step_{step}"] = None
                                search_obs[f"target_category_step_{step}"] = None
                            else:
                                # At-risk scenarios: full wage loss
                                base_delta = -1 * search_obs["annual_earnings"]
                                search_obs[f"earnings_delta_closest_switch_step_{step}"] = base_delta
                                search_obs[f"earnings_delta_closest_switch_sum_step_{step}"] = (
                                        base_delta * search_obs[coeffy_weight]
                                )
                                search_obs[f"earnings_delta_closest_switch_sum_mio_step_{step}"] = (
                                        search_obs[f"earnings_delta_closest_switch_sum_step_{step}"] / 1e6
                                )
                                if pd.notna(search_obs["annual_earnings"]) and search_obs["annual_earnings"] != 0:
                                    search_obs[f"earnings_delta_closest_switch_pct_step_{step}"] = (
                                            base_delta / search_obs["annual_earnings"]
                                    )
                                else:
                                    search_obs[f"earnings_delta_closest_switch_pct_step_{step}"] = np.nan

                                search_obs[f"n_viable_transitions_step_{step}"] = 0
                                search_obs[f"n_viable_transitions_sum_step_{step}"] = 0
                                search_obs[f"transition_viable_step_{step}"] = False
                                search_obs[f"transition_target_step_{step}"] = None
                                search_obs[f"transition_target_code_step_{step}"] = None
                                search_obs[f"target_category_step_{step}"] = None

                        transition_number_data.append(search_obs.to_dict())

                # to df
                if len(transition_number_data) > 0:
                    df_transition_numbers = pd.DataFrame(transition_number_data)
                    df_transition_numbers = df_transition_numbers.infer_objects()

                    if "AGE" in df_transition_numbers.columns:
                        df_transition_numbers["AGE"] = pd.to_numeric(df_transition_numbers["AGE"], errors="coerce")

                    mean_group = df_transition_numbers.columns.str.startswith((
                        "n_viable_transitions_step_",  # per-worker count of options
                        "transition_viable",  # flags
                        "AGE",
                        "earnings_delta_closest_switch_step_",
                        # per-worker € deltas by step  ← IMPORTANT: MEAN, not SUM
                        "earnings_delta_closest_switch_hv_step_",
                        "earnings_delta_closest_switch_pct_step_",
                        "earnings_delta_closest_switch_hv_pct_step_"
                    ))
                    sum_group = df_transition_numbers.columns.str.startswith((
                        "earnings_delta_closest_switch_sum_step_",  # totals by step (€)
                        "earnings_delta_closest_switch_sum_hv_step_",
                        "n_viable_transitions_sum_step_",  # totals (counts)
                        "COEFFY", "NOBS"  # all weight and sample count columns
                    ))

                    cols_mean_group = df_transition_numbers.columns[mean_group]
                    cols_sum_group = df_transition_numbers.columns[sum_group]

                    agg_funcs = {}
                    for col in df_transition_numbers.columns:
                        if col in cols_mean_group:
                            agg_funcs[col] = "mean"
                        elif col in cols_sum_group:
                            agg_funcs[col] = "sum"
                        elif col == "annual_earnings":
                            agg_funcs[col] = "mean"
                        # carry any identifier columns if you want them visible post-agg
                        elif col in ("ISCO08_3D", "ISCO08_3D_label"):
                            agg_funcs[col] = "first"
                        # carry chosen transition targets if present
                        elif col.startswith("transition_target_step_") or col.startswith(
                                "transition_target_code_step_"):
                            agg_funcs[col] = (lambda s: s.dropna().iat[0] if s.dropna().any() else None)
                        else:
                            continue

                    df_summary = (
                        df_transition_numbers
                        .groupby(["COUNTRYW", "NACE2_1D"])
                        .aggregate(agg_funcs)
                    )

                    # restore unsuffixed columns from baseline step (0)
                    _base = 0
                    copy_map = {
                        f"n_viable_transitions_step_{_base}": "n_viable_transitions",
                        f"n_viable_transitions_sum_step_{_base}": "n_viable_transitions_sum",
                        f"earnings_delta_closest_switch_step_{_base}": "earnings_delta_closest_switch",
                        f"earnings_delta_closest_switch_sum_step_{_base}": "earnings_delta_closest_switch_sum",
                        f"earnings_delta_closest_switch_pct_step_{_base}": "earnings_delta_closest_switch_pct",
                        f"earnings_delta_closest_switch_hv_step_{_base}": "earnings_delta_closest_switch_hv",
                        f"earnings_delta_closest_switch_sum_hv_step_{_base}": "earnings_delta_closest_switch_sum_hv",
                    }
                    for src, dst in copy_map.items():
                        if src in df_summary.columns:
                            df_summary[dst] = df_summary[src]

                    # visualise expects Mio€ total as well
                    if "earnings_delta_closest_switch_sum" in df_summary.columns:
                        df_summary["earnings_delta_closest_switch_sum_mio"] = (
                                df_summary["earnings_delta_closest_switch_sum"] / 1e6
                        )

                    # store country results
                    scenario_results[country] = df_summary
                else:
                    continue

            # store scenario-country results
            simulation_results[scenario] = scenario_results

        # save as pickle
        with open(os.path.join(target_dir, fname), "wb") as handle:
            pickle.dump(simulation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(">>> ABOUT TO DUMP pickle:", target_dir, fname, "contents:", simulation_results)

        return simulation_results

    def _select_centrality_ranked(self, idx_occ, occ_skills_mat, ranked_positions, skill_rank):
        """Task A: pick a skill from a coreness-ordered list of column positions.

        `ranked_positions` is a list of occ-skill-matrix column indices, sorted by
        descending coreness (e.g. all skills, or only green / DigComp skills).

        The `journey_aware` switch (instance attribute, default False) controls the
        "not-yet-held" filter:
          * journey_aware=True  -> held = skills with weight>0 in the worker's ACCUMULATED
            matrix (so a skill acquired earlier this journey is never re-offered); pick the
            highest-coreness not-yet-held skill greedily.
          * journey_aware=False -> held = skills with weight>0 in the BASELINE matrix
            (occupation's original skills only); pick by rank (preserves a stable,
            occupation-independent walk down the coreness order).
        Returns a column index, or None if nothing remains.
        """
        journey = getattr(self, "journey_aware", False)
        mat = occ_skills_mat if journey else self.occ_skills_mat_3d
        held = set(np.where(mat.iloc[idx_occ].values > 0)[0])
        rem = [pos for pos in ranked_positions if pos not in held]
        if not rem:
            return None
        if journey:
            return rem[0]
        return rem[skill_rank - 1] if (skill_rank - 1) < len(rem) else None

    @staticmethod
    def _draw_seed(base_seed, step, nuts_code, widx):
        """Deterministic per-draw seed. Because each destination draw is seeded from its
        own (base_seed, step, region, worker-index) key — not a single advancing RNG —
        the draw is independent of iteration order, so a live run and a post-hoc
        replay on the captured feasible set produce byte-identical choices. Uses a
        stable hash (not Python's salted hash) so it is reproducible across processes."""
        key = f"{int(base_seed)}|{int(step)}|{nuts_code}|{int(widx)}".encode()
        return zlib.crc32(key) & 0xFFFFFFFF

    def _select_destination(self, targets, src_worker, draw_seed):
        """Task B: choose the destination occupation among feasible targets.

        `targets` is the feasible set, already sorted by annual_earnings descending and
        (regional path only) merged with the NUTS-2 employment count column ``COEFFY``.
        Returns ``(target_row, income_rank)``. Modes (``self.destination_weighting``):
          * ``"off"``  -> the single top-income target (deterministic) == Phase-1.
          * ``"share_only"`` -> draw a target with probability proportional to its NUTS-2
            employment share (``COEFFY``); income is ignored in the choice.
          * ``"share_income_acceptable"`` -> draw proportional to employment share among
            income-acceptable targets (band of best income, or above-current); if none
            qualify, fall back to the full feasible set.
        ``draw_seed`` (from ``_draw_seed``) makes the draw deterministic and order-
        independent. ``income_rank`` is the chosen target's income position (1 = top).
        """
        mode = getattr(self, "destination_weighting", "off")
        if mode == "off" or len(targets) <= 1 or "COEFFY" not in targets.columns:
            return targets.iloc[0], 1
        pool = targets
        if mode == "share_income_acceptable":
            # The acceptable set is the modelling choice (swept on the sample). Two rules:
            #   acceptability="band"          -> earnings >= (1 - income_band) * best feasible
            #                                    (band=0 -> top only ~ off; band=1 -> all ~ share_only)
            #   acceptability="above_current" -> earnings >= the worker's current earnings
            if getattr(self, "acceptability", "band") == "above_current":
                acc = targets[targets["annual_earnings"] >= src_worker["annual_earnings"]]
            else:
                band = getattr(self, "income_band", 0.0)
                thr = (1.0 - band) * float(targets["annual_earnings"].max())
                acc = targets[targets["annual_earnings"] >= thr]
            if len(acc) > 0:
                pool = acc
        w = np.asarray(pool["COEFFY"], dtype=float)
        if not np.isfinite(w).all() or w.sum() <= 0:
            return targets.iloc[0], 1  # degenerate weights -> fall back to top income
        chosen = pool.iloc[int(np.random.RandomState(draw_seed).choice(len(pool), p=w / w.sum()))]
        rank = int((targets["annual_earnings"] > chosen["annual_earnings"]).sum()) + 1
        return chosen, rank

    def reskill(
        self,
        idx_occ,
        occ_skills_mat,
        search_label=None,
        reskilling_mode="optimal",
        skill_rank=1,
        q_coreness=99.9,
        mask_diagonal=True,
        prev_sim_mat=None,
    ):
        """

        Parameters
        ----------
        idx_occ : int
            Index of search occupation.
        search_label : str
            Label of search occupation.
        reskilling_mode : str
            Mode of reskilling. One of:
                coreness_weighted: workers randomly acquire a skill, although with
                    probabilities weighted by a skill's coreness
                optimal: workers acquire the skill that unlocks most new job transition
                    options.
        q_coreness : float
            Percentile of the skills coreness distribution used in coreness_percentile
            reskilling mode.

        Returns
        -------
        occ_sim_mat_3d_updated : np.array
            Updated occupation similarity matrix.
        """

        # Steps dependent on reskilling mode:
        # 1) select skill that should be added to worker's skill set
        # 2) find id/idx of the occupation and skill within the matrix

        #   Optimal reskilling
        if reskilling_mode == "optimal":
            upskilling_data = self.df_optimal_upskilling_per_occ.loc[
                self.df_optimal_upskilling_per_occ["ISCO08_3D_label"] == search_label
            ]
            # subtract 1 from skill rank because python is 0-indexed
            col_loc = upskilling_data.columns.get_loc("idx_skill")
            idx_skill = upskilling_data.iloc[skill_rank - 1, col_loc]

        # Broad/ transferable reskilling:
        # 1) by weights
        elif reskilling_mode == "coreness_weighted":
            idx_skill = self.df_coreness.sample(n=1, weights="coreness").index.values[0]

        # 2) by rank
        elif reskilling_mode == "coreness_ranked":
            if not hasattr(self, "core_ranked_skills"):
                ranked = self.df_coreness.sort_values("coreness", ascending=False)
                self.core_ranked_skills = ranked.index.tolist()
                # Cache each ranked skill's preferredLabel ONCE, aligned with
                # core_ranked_skills, so the per-call filter below needs no pandas
                # .loc lookups (previously ~13,891 .loc calls per reskill() call).
                self._core_ranked_labels = ranked["preferredLabel"].tolist()
                self._rem_cache = {}
            # NOTE (behaviour preserved, do not change here): `have` is read from the
            # BASELINE matrix self.occ_skills_mat_3d, so it depends only on idx_occ and
            # NOT on skills acquired earlier in this worker's journey. Whether the
            # "not-yet-held" filter should track within-journey acquisitions is a real
            # semantic question recorded as an open question in revision/INVESTIGATION.md
            # and decided deliberately in Phase 2 (Task A). Because `rem` is therefore a
            # pure function of idx_occ, memoise it (identical ordered list, computed once
            # per occupation instead of per worker x per step).
            if getattr(self, "journey_aware", False):
                # Task A (journey-aware): exclude skills already held (baseline + acquired
                # this journey) and take the highest-coreness remaining one.
                idx_skill = self._select_centrality_ranked(
                    idx_occ, occ_skills_mat, self.core_ranked_skills, skill_rank
                )
            else:
                # baseline-only (current behaviour, preserved for comparability): the
                # label-vs-URI `have` filter is effectively a no-op, so this walks the
                # global coreness order by rank. Memoised per idx_occ.
                rem = self._rem_cache.get(idx_occ)
                if rem is None:
                    have = set(
                        self.occ_skills_mat_3d.columns[
                            self.occ_skills_mat_3d.iloc[idx_occ] > 0
                            ]
                    )
                    # filter out “have” from global coreness list (order preserved)
                    rem = [
                        idx
                        for idx, lbl in zip(self.core_ranked_skills, self._core_ranked_labels)
                        if lbl not in have
                    ]
                    self._rem_cache[idx_occ] = rem
                idx_skill = rem[skill_rank - 1] if rem else None

        # Digital reskilling (Task A: coreness-ordered over the DigComp list,
        #  replacing the former random draw):
        elif reskilling_mode == "digital":
            if not hasattr(self, "_digital_ranked"):
                self._digital_ranked = (
                    self.df_coreness[self.df_coreness["conceptUri"].isin(set(self.digital_skills))]
                    .sort_values("coreness", ascending=False)
                    .index.tolist()
                )
            idx_skill = self._select_centrality_ranked(
                idx_occ, occ_skills_mat, self._digital_ranked, skill_rank
            )

        # Green reskilling (Task A: coreness-ordered over the ESCO green list,
        #  replacing the former random draw):
        elif reskilling_mode == "green":
            if not hasattr(self, "_green_ranked"):
                self._green_ranked = (
                    self.df_coreness[self.df_coreness["conceptUri"].isin(set(self.green_skills))]
                    .sort_values("coreness", ascending=False)
                    .index.tolist()
                )
            idx_skill = self._select_centrality_ranked(
                idx_occ, occ_skills_mat, self._green_ranked, skill_rank
            )

        # Steps dependent of reskilling mode:
        # 3) add the worker's newly acquired skill (essential = value of 1).
        #    The caller (simulate / simulate_regional) passes a fresh per-worker copy
        #    of occ_skills_mat_3d, so we mutate it in place and avoid copying the full
        #    (125 x 13891) matrix on every skill step.
        occ_skills_mat_3d_updated = occ_skills_mat
        if idx_skill is not None:
            occ_skills_mat_3d_updated.iloc[idx_occ, idx_skill] = 1

        # 4) update the occupation-similarity matrix M_oo = M_os @ M_os.T.
        #    Adding one skill to occupation `idx_occ` changes only row/column idx_occ,
        #    so recompute just that vector instead of the full dense product. This is
        #    equivalent to the full recompute within floating tolerance (~1e-15); the
        #    only row that feeds find_closest (row idx_occ) is recomputed exactly.
        #    Fall back to a full recompute when no prior matrix is carried.
        if prev_sim_mat is None or idx_skill is None:
            occ_sim_mat_3d_updated = np.dot(
                occ_skills_mat_3d_updated.values,
                occ_skills_mat_3d_updated.values.transpose(),
            )
            if mask_diagonal:
                np.fill_diagonal(occ_sim_mat_3d_updated, 0)
        else:
            M = occ_skills_mat_3d_updated.values
            new_vec = M @ M[idx_occ]  # M_oo[idx_occ, :] == M_oo[:, idx_occ] (symmetric)
            occ_sim_mat_3d_updated = np.array(prev_sim_mat, copy=True)
            occ_sim_mat_3d_updated[idx_occ, :] = new_vec
            occ_sim_mat_3d_updated[:, idx_occ] = new_vec
            if mask_diagonal:
                occ_sim_mat_3d_updated[idx_occ, idx_occ] = 0.0

        return occ_sim_mat_3d_updated, occ_skills_mat_3d_updated, idx_skill

    def simulate_regional(
        self,
        level="isco_3_digit",
        countries=None,
        scenarios=None,
        reskilling=None,
        reskilling_journey_length=1,
        transition_optimisation="wage",
        mask_diagonal=True,
        transition_thresholds=None,
        threshold="viable",
        verbose=False,
        region_constraints=True,
        target_job_availability_coeffy="COEFFY_mean+sd",
        out_dir=os.path.join(useful_paths.figure_dir, "reskilling_simulation"),
        optional_weight=0.5,
        symmetric_employment=False,
    ):

        print(f"→ Enter simulate_regional(level={level}, regions={region_constraints}, "
              f"countries={countries}, scenarios={scenarios})")

        """
        Simulate occupation transitions for given phase-out scenarios with
        regional constraints (baseline or with reskilling).

        Parameters
        ----------
        level : str
            Granularity level
        countries : list of str
            Countries to rerun_simulations.
        scenarios : list of str
            Phase-out scenarios to rerun_simulations.
        reskilling : str
            Mode of reskilling (See documentation of self.reskill). One of
            None, "coreness_weighted","optimal"]. See documentation of self.reskill.
        (REMOVED: target_job_availability_coeffy: str
            Availability of jobs in country-region-occupation. Based on long-term
            (1998-2019) positive employment fluctuations.)
        transition_optimisation : str
            Which variable to optimise in the simulation of occupation transitions.
            Either 'wage' (minimise wage loss) or 'skill' (maximise skill overlap).
        out_dir : os.Path
            Path to output directory

        Returns
        -------
        simulation_results : dict of dicts of pd.DataFrame's
            Nested dictionary containing simulation results by (1) scenario and (2)
            country. Simulation results are stored as pd.DataFrame and contain worker-
            level information about: number of viable transitions, target occupation,
            earnings changes, etc.
        """
        # suppress warnings
        pd.options.mode.chained_assignment = None

        # define transition pool based on scenario
        if scenarios is None:
            scenarios = self.phaseout_scenarios
        if countries is None:
            countries = ["DE"]

        # read original sim matrix
        similarity_matrix = self.sim_matrix_at_level(
            level=level, mask_diagonal=mask_diagonal
        ).values

        # define transition thresholds
        if transition_thresholds is None:
            q_viable, q_highly_viable = self.trans_thresh_pc_approach
        else:
            q_viable, q_highly_viable = transition_thresholds
        print("Viability thresholds:", q_viable, q_highly_viable)

        # Task B: destination draws are seeded per-draw (see _draw_seed), so no shared
        # RNG state is needed and the draw is order-independent. _capture is the
        # read-only side-channel for the Task B sample sweep / validation.
        self._capture = []

        # create output dir and fnames
        reg_constraint_str = "regC" if region_constraints else "no-regC"

        # format: "{sim_version}_{opt_target}-opt_{reg_constraint}_{year}"
        dirname = self.dirname_out_reg.format(
            sim_version=self.simulation_name[reskilling],
            opt_target=transition_optimisation,
            reg_constraint=reg_constraint_str,
            year=self.year,
        )
        # Robustness-variant tag: encode any non-default parameter that changes the
        # output, so variants self-organise on disk and never overwrite each other or the
        # Phase-1 baseline. Default (optional weight 0.5, symmetric off) -> NO suffix, so
        # baseline paths are byte-identical and reproducible. (Naming-only; the I/O system
        # itself is untouched — Phase 3.)
        variant = ""
        if abs(float(optional_weight) - 0.5) > 1e-12:
            variant += "_optw{:g}".format(optional_weight)
        if symmetric_employment:
            variant += "_symE"
        dirname = dirname + variant
        fname = "{}.pkl".format(dirname)
        target_dir = os.path.join(out_dir, dirname)
        utils.ccdir(target_dir)

        # Sidecar metadata: make the run self-documenting so a robustness variant's
        # parameters and DERIVED threshold are recoverable from the artifact, not memory
        # (e.g. weight-0's q_viable=1.42). Written next to the pkl.
        try:
            import json
            with open(os.path.join(target_dir, "run_metadata.json"), "w") as _mh:
                json.dump({
                    "program": self.simulation_name[reskilling],
                    "transition_optimisation": transition_optimisation,
                    "region_constraints": region_constraints,
                    "year": self.year,
                    "optional_weight": float(optional_weight),
                    "symmetric_employment": bool(symmetric_employment),
                    "q_viable": float(q_viable),
                    "q_highly_viable": float(q_highly_viable),
                    "journey_aware": getattr(self, "journey_aware", None),
                    "destination_weighting": getattr(self, "destination_weighting", None),
                }, _mh, indent=2)
        except Exception as _e:
            print(f"[warn] could not write run_metadata.json: {_e}")

        # load the table of jobs by NUTS2 & ISCO
        jobs_by_regions_countries = self.jobs_by_country_and_region()

        # -----------------------------------------------------------------------------
        # Scenario loop (#1)
        # -----------------------------------------------------------------------------
        simulation_results = {}
        for scenario in scenarios:
            print("scenario: {}".format(scenario))
            # select scenario-specific weighting coefficient
            coeffy_weight = self.transition_pool_weights[scenario]

            # scenario-specific categories of occupations
            category_version = self.category_versions[scenario]

            # -------------------------------------------------------------------------
            # Country loop (#2)
            # -------------------------------------------------------------------------
            scenario_results = {}

            # NEW: drop countries without granular NUTS2 data from regC
            if region_constraints:
                filtered_countries = []
                for country in countries:
                    # fetch its transition pool
                    df_tp = self.define_transition_pool(scenario=scenario, country=country)
                    # check for any real NUTS2 (not ending in "00")
                    has_real = (
                            df_tp["NUTS_ID"]
                            .astype(str)
                            .loc[lambda s: ~s.str.endswith("00")]
                            .nunique()
                            > 0
                    )
                    if has_real:
                        filtered_countries.append(country)
                    else:
                        print(f"[INFO] dropping {country} from regC run (only XX00 codes)")
            else:
                filtered_countries = countries

            for country in filtered_countries:
                if verbose:
                    print("  country: {}".format(country))

                # select scenario- and country-specific transition pool
                df_transition_pool = self.define_transition_pool(
                    scenario=scenario, country=country
                )

                # Debugging section
                raw_codes = df_transition_pool["NUTS_ID"].astype(str).unique().tolist()
                valid_nuts2 = (
                    self.gdf
                    .loc[self.gdf["CNTR_CODE"] == country, "NUTS_ID"]
                    .astype(str)
                    .unique()
                    .tolist()
                )
                if verbose:
                    print(f"[DEBUG] country={country}  LFS NUTS_IDs: {raw_codes}")
                    print(f"[DEBUG] country={country}  GeoData NUTS_IDs: {valid_nuts2}")

                # restrict jobs_by_regions_countries to this country only
                jobs_by_regions = jobs_by_regions_countries[
                    jobs_by_regions_countries["COUNTRYW"] == country
                    ]

                # nuts codes (number of unique nuts codes depends on scenario)
                nuts_codes = df_transition_pool.NUTS_ID.unique()

                # read occupation list coherent with sim matrix and enriched by means
                # across several LFS variables
                df_occs = self.get_occs(
                    level=level,
                    lfs_country_subset=self.lfs_data[
                        self.lfs_data["COUNTRYW"] == country
                    ],
                )

                share_cols = [
                    "share_viable_to_decarbonize",
                    "share_unviable_to_decarbonize",
                    "share_high_carbon",
                    "share_neutral",
                    "share_low_carbon",
                ]
                country_share_means = {c: df_occs[c].mean() for c in share_cols}

                # check if earnings are still in df
                assert "annual_earnings" in df_occs.columns, (
                    f" Missing annual_earnings in df_occs for country {country}. "
                    "Did the LFS get re-preprocessed without earnings?"
                )
                if verbose:
                    print(f"[DEBUG] annual_earnings present in df_occs ({len(df_occs)} rows)")


                # -------------------------------------------------------------------------
                # Regions loop (#3)
                # -------------------------------------------------------------------------
                results_by_region = {}
                #jobs_by_region_updated = {}
                for nuts_code in tqdm(nuts_codes):
                    # print(nuts_code)

                    jobs_by_region = jobs_by_regions[
                        jobs_by_regions["NUTS_ID"] == nuts_code
                    ]

                    # regional subset of transition pool
                    regional_transition_pool = df_transition_pool.loc[
                        df_transition_pool["NUTS_ID"] == nuts_code
                    ]

                    # regional subset of source occupations
                    # shuffle order to randomise the next loop (randomisation #1)
                    rng = np.random.RandomState(42) # NEW
                    src_occ_groups = list(regional_transition_pool.ISCO08_3D.unique())
                    rng.shuffle(src_occ_groups)
                    #np.random.shuffle(src_occ_groups) # OLD

                    # -----------------------------------------------------------------
                    # Occupation loop (#4): source occupation categories (ISCO 3-digit)
                    # -----------------------------------------------------------------
                    transition_number_data = []
                    for src_occ_group in src_occ_groups:

                        # pool of transitioning workers in region i and occupation j
                        src_workers = regional_transition_pool.loc[
                            regional_transition_pool.ISCO08_3D == src_occ_group
                        ]

                        # ceil number of workers searching new job
                        #src_workers[coeffy_weight] = np.ceil(src_workers[coeffy_weight]) #OLD
                        src_workers.loc[:, coeffy_weight] = np.ceil(src_workers[coeffy_weight]) #NEW

                        # absolute number of transitioning workers in
                        # region i and occupation j
                        n_workers_transitioning = src_workers[coeffy_weight].sum()

                        # find index of search occ and closest target occs
                        idx = df_occs.loc[
                            df_occs["code"] == src_occ_group
                        ].index.values[0]

                        # find occ label for code
                        label_code_mapping = self.get_occs()
                        search_label = label_code_mapping.loc[
                            label_code_mapping["code"] == src_occ_group,
                            "preferredLabel",
                        ].values[0]

                        # -------------------------------------------------------------
                        # Reskilling step (optional)
                        # -------------------------------------------------------------

                        # init baseline occupation-skills matrix (needs to be outside
                        #  reskilling journey loop)
                        if reskilling is not None:
                            occ_skills_mat = self.occ_skills_mat_3d.copy()
                            added_skill = None  # NEW

                        # NEW: local sim matrix
                        sim_mat = similarity_matrix

                        # loop over reskilling journey (for baseline, it has length 1)
                        # note: step = 0 represents the baseline
                        for step in range(0, reskilling_journey_length + 1):
                            # print("reskilling journey step: {}".format(step))

                            if reskilling is not None and step > 0:
                                # iteratively update the occupation-skills matrix
                                #  while storing all previous changes made within
                                #  a given reskilling scenario
                                sim_mat, occ_skills_mat, added_skill = self.reskill( #NEW: added_skill; sim_mat
                                    occ_skills_mat=occ_skills_mat,
                                    idx_occ=idx,
                                    search_label=search_label,
                                    reskilling_mode=reskilling,
                                    skill_rank=int(step),
                                    mask_diagonal=mask_diagonal,
                                    prev_sim_mat=sim_mat,  # incremental M_oo update
                                )

                            # find closest target occupations
                            target_occs = occupation_distance.find_closest(
                                i=idx, similarity_matrix=sim_mat, df=df_occs #NEW: sim_mat
                            )

                            # FILTER CRITERIA 1) : sim > viability threshold
                            target_occs_filtered = target_occs.loc[
                                target_occs["similarity"] > q_viable
                            ]

                            # FILTER CRITERIA 2): target occupation is neutral or low-carbon
                            # logic: use share means instead of category
                            share_filter_by_scenario = {
                                "at_risk": [
                                    "share_viable_to_decarbonize",
                                    "share_neutral",
                                    "share_low_carbon",
                                ],
                                "high_carbon": [
                                    "share_neutral",
                                    "share_low_carbon",
                                ],
                                "shortage": [
                                    "share_low_carbon",
                                ],
                            }
                            cols_to_check = share_filter_by_scenario.get(scenario, [])
                            if not cols_to_check:
                                raise NotImplementedError(f"No share filter defined for scenario: {scenario}")

                            # filter occupations where any of the scenario-relevant shares are above country mean
                            mask_share = False
                            for sc in cols_to_check:
                                if scenario == "shortage":  # NEW: simpler logic for shortage
                                    mask_share = mask_share | (target_occs_filtered[sc] > 0)
                                else:
                                    if sc not in target_occs_filtered.columns or sc not in country_share_means:
                                        raise ValueError(
                                            f"Column '{sc}' missing in target data or country means.")
                                    mask_share = mask_share | (
                                                target_occs_filtered[sc] > country_share_means[sc])

                            target_occs_filtered = target_occs_filtered.loc[mask_share]


                            # how many raw vs. pre-merge filtered targets?
                            if verbose:
                                print(f"[DEBUG1] NUTS2 {nuts_code}: raw targets = {len(target_occs)}, "
                                      f"pre-merge filtered = {len(target_occs_filtered)}")

                            # only keep occupations that actually exist in this NUTS2
                            if region_constraints:
                                target_occs_filtered = target_occs_filtered.merge(
                                    jobs_by_region[["ISCO08_3D", "COEFFY"]],
                                    left_on="code",
                                    right_on="ISCO08_3D",
                                    how="inner",  # switch to left for absorptive constraint
                                ).drop(columns=["ISCO08_3D"])
                                # how many survive the region‐filter merge?
                                if verbose:
                                    print(f"[DEBUG2] NUTS2 {nuts_code}: post-merge filtered = {len(target_occs_filtered)}")

                                # for all countries where wage data is available, potential
                            #  target occupations are ranked by wage. in all other cases,
                            #  occupations are ranked by similarity scores.
                            # note: does this bias our results?
                            if not target_occs_filtered["annual_earnings"].isna().all():
                                target_occs_filtered = target_occs_filtered.sort_values(
                                    "annual_earnings", ascending=False
                                )
                            else:
                                target_occs_filtered = target_occs_filtered.sort_values(
                                    "similarity", ascending=False
                                )

                            # number of available target occs
                            n_targets = len(target_occs_filtered)

                            # shuffle order in which workers change job (randomisation #2)
                            # idea: could also weight by inverse age, but idk how valid
                            # of an assumption that is (job transition probability
                            # might be more gaussian-shaped)
                            src_workers = src_workers.sample(frac=1, random_state=42) #NEW: random_date added

                            # check if viable transitions exists
                            if verbose:
                                print(f"[DBG] region {nuts_code}: src_workers={len(src_workers)}, "
                                      f"raw targets={len(target_occs)}, "
                                      f"filtered targets={len(target_occs_filtered)}, "
                                      f"q_viable={q_viable:.3f}, "
                                      f"sim_min={target_occs['similarity'].min():.3f}, "
                                      f"sim_max={target_occs['similarity'].max():.3f}")

                            # Task B sample capture (read-only side-channel): when enabled
                            # for this step, record the feasible target set (mode-invariant)
                            # plus, per worker, the (deterministic) draw seed and the choice
                            # this run actually made. The post-hoc sweep recomputes choices
                            # from the feasible set + seed; capturing the live choice here
                            # lets the validation assert post-hoc == live exactly.
                            cap_entry = None
                            if (getattr(self, "_capture_at_step", None) == step
                                    and region_constraints and "COEFFY" in target_occs_filtered.columns
                                    and not target_occs_filtered.empty):
                                cap_entry = {
                                    "scenario": scenario, "country": country,
                                    "nuts": nuts_code, "step": step,
                                    "weight_col": coeffy_weight,
                                    "targets": target_occs_filtered[
                                        ["code", "annual_earnings", "COEFFY"]].reset_index(drop=True).copy(),
                                    "choices": [],
                                }
                                self._capture.append(cap_entry)

                            if not target_occs_filtered.empty:
                                if region_constraints:
                                    # -----------------------------------------------------
                                    # Worker loop (#5a): individual, region‐constrained
                                    # -----------------------------------------------------
                                    for widx, (_, src_worker) in enumerate(src_workers.iterrows()):

                                        # NEW: Always stamp the skill just added
                                        if reskilling == "optimal" and added_skill is not None:
                                            src_worker[f"added_skill_idx_step_{step}"] = added_skill
                                            df_opt = self.df_optimal_upskilling_per_occ
                                            sel = (
                                                    (df_opt["ISCO08_3D_label"].str.strip().str.lower()
                                                     == search_label.strip().lower())
                                                    & (df_opt["idx_skill"] == added_skill)
                                            )
                                            src_worker[f"added_skill_label_step_{step}"] = \
                                            df_opt.loc[sel, "skill_label"].iat[0]

                                        rank = 1
                                        while rank <= n_targets:
                                            # Task B: choose the destination. "off" returns
                                            # the top-income target (== Phase-1); the share
                                            # modes draw one weighted by NUTS-2 employment
                                            # share (COEFFY), seeded deterministically per draw.
                                            draw_seed = self._draw_seed(
                                                getattr(self, "share_seed", 42), step, nuts_code, widx
                                            )
                                            target, rank = self._select_destination(
                                                target_occs_filtered, src_worker, draw_seed
                                            )
                                            if cap_entry is not None:
                                                cap_entry["choices"].append({
                                                    "widx": widx,
                                                    "earn": float(src_worker["annual_earnings"]),
                                                    "w": float(src_worker[coeffy_weight]),
                                                    "seed": int(draw_seed),
                                                    "chosen": target["code"],
                                                })

                                                # — everyone takes their top‐ranked viable job —
                                            src_worker[f"transition_viable_step_{step}"] = True
                                            src_worker[f"transition_target_step_{step}"] = target["preferredLabel"]
                                            src_worker[f"transition_target_code_step_{step}"] = target["code"]
                                            src_worker[f"transition_target_rank_step_{step}"] = rank
                                            src_worker[f"target_category_step_{step}"] = target[
                                                self.category_versions[scenario]
                                            ]

                                            # stats & wage deltas
                                            src_worker[f"n_viable_transitions_step_{step}"] = \
                                                target_occs_filtered.shape[0]
                                            src_worker[f"n_viable_transitions_sum_step_{step}"] = (
                                                    src_worker[f"n_viable_transitions_step_{step}"] * src_worker[
                                                coeffy_weight]
                                            )
                                            src_worker[f"earnings_delta_closest_switch_step_{step}"] = (
                                                    target["annual_earnings"] - src_worker["annual_earnings"]
                                            )
                                            src_worker[f"earnings_delta_closest_switch_sum_step_{step}"] = (
                                                    src_worker[f"earnings_delta_closest_switch_step_{step}"] *
                                                    src_worker[coeffy_weight]
                                            )

                                            src_worker[f"earnings_delta_closest_switch_sum_mio_step_{step}"] = (
                                                    src_worker[f"earnings_delta_closest_switch_sum_step_{step}"] / 1e6
                                            )

                                            # ADD: per-worker % change (mean-aggregated later)
                                            if pd.notna(src_worker["annual_earnings"]) and src_worker[
                                                "annual_earnings"] != 0:
                                                src_worker[f"earnings_delta_closest_switch_pct_step_{step}"] = (
                                                        src_worker[f"earnings_delta_closest_switch_step_{step}"] /
                                                        src_worker["annual_earnings"]
                                                )
                                            else:
                                                src_worker[f"earnings_delta_closest_switch_pct_step_{step}"] = np.nan

                                            #transition_number_data.append(src_worker) #OLD
                                            transition_number_data.append(src_worker.to_dict()) #NEW
                                            break  # done with this worker

                                        else:
                                            # if we somehow ran out of ranks (should not happen)
                                            src_worker[f"transition_viable_step_{step}"] = False
                                            src_worker[f"transition_target_rank_step_{step}"] = rank

                                            # Task E (reporting-only): the asymmetry is outward
                                            # non-reachers -> unemployment (full wage loss), inward
                                            # non-reachers -> keep job (Δ=0). symmetric_employment
                                            # applies the keep-job rule to BOTH flows (outward
                                            # workers who cannot transition keep their current job
                                            # rather than becoming unemployed). Default off ->
                                            # Phase-1 behaviour; on -> output dir gets the _symE tag.
                                            if scenario == "shortage" or symmetric_employment:
                                                # Keep wage (Δ = 0)
                                                base_delta = 0.0
                                                src_worker[f"earnings_delta_closest_switch_step_{step}"] = base_delta
                                                src_worker[f"earnings_delta_closest_switch_sum_step_{step}"] = 0.0
                                                src_worker[f"earnings_delta_closest_switch_sum_mio_step_{step}"] = 0.0
                                                if pd.notna(src_worker["annual_earnings"]) and src_worker[
                                                    "annual_earnings"] != 0:
                                                    src_worker[f"earnings_delta_closest_switch_pct_step_{step}"] = 0.0
                                                else:
                                                    src_worker[
                                                        f"earnings_delta_closest_switch_pct_step_{step}"] = np.nan
                                            else:
                                                # Full wage loss
                                                src_worker[f"earnings_delta_closest_switch_step_{step}"] = -src_worker[
                                                    "annual_earnings"]
                                                src_worker[f"earnings_delta_closest_switch_sum_step_{step}"] = (
                                                        src_worker[f"earnings_delta_closest_switch_step_{step}"] *
                                                        src_worker[coeffy_weight]
                                                )
                                                src_worker[f"earnings_delta_closest_switch_sum_mio_step_{step}"] = (
                                                        src_worker[
                                                            f"earnings_delta_closest_switch_sum_step_{step}"] / 1e6
                                                )
                                                if pd.notna(src_worker["annual_earnings"]) and src_worker[
                                                    "annual_earnings"] != 0:
                                                    src_worker[f"earnings_delta_closest_switch_pct_step_{step}"] = (
                                                            -src_worker["annual_earnings"] / src_worker[
                                                        "annual_earnings"]
                                                    )  # = -1.0
                                                else:
                                                    src_worker[
                                                        f"earnings_delta_closest_switch_pct_step_{step}"] = np.nan

                                            #transition_number_data.append(src_worker) #OLD
                                            transition_number_data.append(src_worker.to_dict()) #NEW

                                else:
                                    # -----------------------------------------------------
                                    # Worker loop (#5b): no regional constraint (bulk)
                                    # -----------------------------------------------------
                                    target = target_occs_filtered.iloc[0]
                                    for _, src_worker in src_workers.iterrows():

                                        # NEW: Always stamp the skill just added
                                        if reskilling == "optimal" and added_skill is not None:
                                            src_worker[f"added_skill_idx_step_{step}"] = added_skill
                                            df_opt = self.df_optimal_upskilling_per_occ
                                            sel = (
                                                    (df_opt["ISCO08_3D_label"].str.strip().str.lower()
                                                     == search_label.strip().lower())
                                                    & (df_opt["idx_skill"] == added_skill)
                                            )
                                            src_worker[f"added_skill_label_step_{step}"] = df_opt.loc[sel, "skill_label"].iat[0]

                                        src_worker[f"transition_viable_step_{step}"] = True
                                        src_worker[f"transition_target_step_{step}"] = target["preferredLabel"]
                                        src_worker[f"transition_target_code_step_{step}"] = target["code"]
                                        src_worker[f"transition_target_rank_step_{step}"] = np.nan
                                        src_worker[f"target_category_step_{step}"] = target[
                                            self.category_versions[scenario]
                                        ]

                                        src_worker[f"n_viable_transitions_step_{step}"] = target_occs_filtered.shape[0]
                                        src_worker[f"n_viable_transitions_sum_step_{step}"] = (
                                                src_worker[f"n_viable_transitions_step_{step}"] * src_worker[
                                            coeffy_weight]
                                        )
                                        src_worker[f"earnings_delta_closest_switch_step_{step}"] = (
                                                target["annual_earnings"] - src_worker["annual_earnings"]
                                        )
                                        src_worker[f"earnings_delta_closest_switch_sum_step_{step}"] = (
                                                src_worker[f"earnings_delta_closest_switch_step_{step}"] *
                                                src_worker[coeffy_weight]
                                        )
                                        src_worker[f"earnings_delta_closest_switch_sum_mio_step_{step}"] = (
                                                src_worker[f"earnings_delta_closest_switch_sum_step_{step}"] / 1e6
                                        )

                                        # ADD: per-worker % change (mean-aggregated later)
                                        if pd.notna(src_worker["annual_earnings"]) and src_worker[
                                            "annual_earnings"] != 0:
                                            src_worker[f"earnings_delta_closest_switch_pct_step_{step}"] = (
                                                    src_worker[f"earnings_delta_closest_switch_step_{step}"] /
                                                    src_worker["annual_earnings"]
                                            )
                                        else:
                                            src_worker[f"earnings_delta_closest_switch_pct_step_{step}"] = np.nan

                                        #transition_number_data.append(src_worker) #OLD
                                        transition_number_data.append(src_worker.to_dict()) #NEW

                            else:
                                # -----------------------------------------------------
                                # Worker loop (#5c): no viable targets → all “fail”
                                # -----------------------------------------------------
                                for _, src_worker in src_workers.iterrows():

                                    # NEW: Always stamp the skill just added
                                    if reskilling == "optimal" and added_skill is not None:
                                        src_worker[f"added_skill_idx_step_{step}"] = added_skill
                                        df_opt = self.df_optimal_upskilling_per_occ
                                        sel = (
                                                (df_opt["ISCO08_3D_label"].str.strip().str.lower()
                                                 == search_label.strip().lower())
                                                & (df_opt["idx_skill"] == added_skill)
                                        )
                                        src_worker[f"added_skill_label_step_{step}"] = \
                                        df_opt.loc[sel, "skill_label"].iat[0]

                                    # Task E (reporting-only): symmetric_employment applies the
                                    # keep-job rule to BOTH flows (outward non-reachers keep their
                                    # current job instead of becoming unemployed). Default off.
                                    if scenario == "shortage" or symmetric_employment:
                                        # keep wage (Δ = 0)
                                        src_worker[f"earnings_delta_closest_switch_step_{step}"] = 0.0
                                        src_worker[f"earnings_delta_closest_switch_sum_step_{step}"] = 0.0
                                        src_worker[f"earnings_delta_closest_switch_sum_mio_step_{step}"] = 0.0
                                        if pd.notna(src_worker["annual_earnings"]) and src_worker[
                                            "annual_earnings"] != 0:
                                            src_worker[f"earnings_delta_closest_switch_pct_step_{step}"] = 0.0
                                        else:
                                            src_worker[f"earnings_delta_closest_switch_pct_step_{step}"] = np.nan
                                    else:
                                        # full wage loss
                                        src_worker[f"earnings_delta_closest_switch_step_{step}"] = -src_worker[
                                            "annual_earnings"]
                                        src_worker[f"earnings_delta_closest_switch_sum_step_{step}"] = (
                                                src_worker[f"earnings_delta_closest_switch_step_{step}"] *
                                                src_worker[coeffy_weight]
                                        )
                                        src_worker[f"earnings_delta_closest_switch_sum_mio_step_{step}"] = (
                                                src_worker[f"earnings_delta_closest_switch_sum_step_{step}"] / 1e6
                                        )
                                        # % change = -1.0 unless earnings is 0/NaN
                                        if pd.notna(src_worker["annual_earnings"]) and src_worker[
                                            "annual_earnings"] != 0:
                                            src_worker[f"earnings_delta_closest_switch_pct_step_{step}"] = -1.0
                                        else:
                                            src_worker[f"earnings_delta_closest_switch_pct_step_{step}"] = np.nan

                                    src_worker[f"n_viable_transitions_step_{step}"] = 0
                                    src_worker[f"n_viable_transitions_sum_step_{step}"] = 0
                                    src_worker[f"transition_viable_step_{step}"] = False
                                    src_worker[f"transition_target_step_{step}"] = None
                                    src_worker[f"transition_target_code_step_{step}"] = None
                                    src_worker[f"target_category_step_{step}"] = None
                                    #transition_number_data.append(src_worker) #OLD
                                    transition_number_data.append(src_worker.to_dict()) #NEW

                    if verbose:
                        print(f"[DEBUG3] Appending {len(transition_number_data)} worker records for NUTS2 {nuts_code}")

                    # end of loop over reskilling journey steps
                    results_by_region[nuts_code] = transition_number_data

                # combine transition results to df
                nested_list = list(results_by_region.values())
                flat_list = [item for sublist in nested_list for item in sublist]
                #df_transition_numbers = pd.concat(flat_list, axis=1).T #OLD
                df_transition_numbers = pd.DataFrame(flat_list) #NEW

                print("DEBUG columns before aggregation:", df_transition_numbers.columns.tolist()[:20],
                      "... +", len([c for c in df_transition_numbers.columns if c.startswith("added_skill_idx_step_")]),
                      "added_skill columns found")

                # obj to numeric
                # cast obj to float
                df_transition_numbers = df_transition_numbers.infer_objects()
                if "AGE" in df_transition_numbers.columns:
                    df_transition_numbers["AGE"] = pd.to_numeric(df_transition_numbers["AGE"], errors="coerce")

                # defensive coercion: ensure coeffy_weight is numeric & non-negative
                coeffy = coeffy_weight
                df_transition_numbers[coeffy] = pd.to_numeric(df_transition_numbers.get(coeffy, 0),
                                                              errors="coerce").fillna(0)
                if (df_transition_numbers[coeffy] < 0).any():
                    raise ValueError(
                        f"Negative {coeffy} in df_transition_numbers for country {country} in simulate_regional()")

                # NEW: carry original occupation through
                df_transition_numbers = df_transition_numbers.rename(
                    columns={
                        "ISCO08_3D": "orig_ISCO08_3D",
                        "ISCO08_3D_label": "orig_ISCO08_3D_label",
                    }
                )

                # subset results
                mean_group = df_transition_numbers.columns.str.startswith((
                    "n_viable_transitions_step",
                    "transition_viable",
                    "AGE",
                    "earnings_delta_closest_switch_step_",
                    "earnings_delta_closest_switch_hv_step_",
                    "earnings_delta_closest_switch_pct_step_",  # ADD
                    "earnings_delta_closest_switch_hv_pct_step_"  # ADD
                ))
                sum_group = df_transition_numbers.columns.str.startswith((
                    "earnings_delta_closest_switch_sum_step_",
                    "earnings_delta_closest_switch_sum_hv_step_",
                    "n_viable_transitions_sum_step_",
                    "COEFFY", "NOBS"
                ))

                # define aggregation pools
                cols_mean_group = df_transition_numbers.columns[mean_group]
                cols_sum_group = df_transition_numbers.columns[sum_group]

                # create agg func
                agg_funcs = {}
                for col in df_transition_numbers.columns:
                    if col in cols_mean_group:
                        agg_funcs[col] = "mean"
                    elif col in cols_sum_group:
                        agg_funcs[col] = "sum"
                    elif col == "annual_earnings":  # NEW
                        agg_funcs[col] = "mean"  # NEW
                    else:
                        continue

                # propagate original occupation by taking the first value
                agg_funcs["orig_ISCO08_3D"] = lambda s: s.iat[0]
                agg_funcs["orig_ISCO08_3D_label"] = lambda s: s.iat[0]

                # stamp through added_skill_idx/label columns (only exist in optimal+regC)
                for col in df_transition_numbers.columns:
                    if col.startswith("added_skill_idx_step_") or col.startswith("added_skill_label_step_"):
                        def agg_added_skill(s):
                            non_na = s.dropna()
                            if len(non_na) > 0:
                                return non_na.iat[0]
                            else:
                                return None
                        agg_funcs[col] = agg_added_skill

                # carry forward the chosen transition_target & its code
                for col in df_transition_numbers.columns:
                    if col.startswith("transition_target_step_") or col.startswith("transition_target_code_step_"):
                        agg_funcs[col] = lambda s: s.dropna().iat[0] if s.dropna().any() else None

                # aggregate
                df_transition_numbers_by_nuts_nace = (
                    df_transition_numbers.groupby(["COUNTRYW", "NUTS_ID", "NACE2_1D"])
                        .aggregate(agg_funcs)
                )

                # restore unsuffixed columns from baseline step (0)
                _base = 0
                copy_map = {
                    f"n_viable_transitions_step_{_base}": "n_viable_transitions",
                    f"n_viable_transitions_sum_step_{_base}": "n_viable_transitions_sum",
                    f"earnings_delta_closest_switch_step_{_base}": "earnings_delta_closest_switch",
                    f"earnings_delta_closest_switch_sum_step_{_base}": "earnings_delta_closest_switch_sum",
                    f"earnings_delta_closest_switch_pct_step_{_base}": "earnings_delta_closest_switch_pct",
                    f"earnings_delta_closest_switch_hv_step_{_base}": "earnings_delta_closest_switch_hv",
                    f"earnings_delta_closest_switch_sum_hv_step_{_base}": "earnings_delta_closest_switch_sum_hv",
                }
                for src, dst in copy_map.items():
                    if src in df_transition_numbers_by_nuts_nace.columns:
                        df_transition_numbers_by_nuts_nace[dst] = df_transition_numbers_by_nuts_nace[src]

                # Visualiser expects the Mio€ total
                if "earnings_delta_closest_switch_sum" in df_transition_numbers_by_nuts_nace.columns:
                    df_transition_numbers_by_nuts_nace["earnings_delta_closest_switch_sum_mio"] = (
                            df_transition_numbers_by_nuts_nace["earnings_delta_closest_switch_sum"] / 1e6
                    )

                # append
                scenario_results[country] = df_transition_numbers_by_nuts_nace

            simulation_results[scenario] = scenario_results

        print(">> Reached end of simulate_regional, about to save:", target_dir, fname)
        print("   simulation_results keys:", list(simulation_results.keys()))

        # save as pickle
        with open(os.path.join(target_dir, fname), "wb") as handle:
            pickle.dump(simulation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return simulation_results

    def visualise_simulation_results(
        self,
        simulation_results=None,
        base_dir=os.path.join(useful_paths.figure_dir, "reskilling_simulation"),
        year=2023,
        transition_optimisation="wage",
        version="baseline",
        show_plots=False,
    ):

        # read file if no results are passed
        if simulation_results is None:
            # path of in-file
            dirname = self.dirname_out.format(version, transition_optimisation, year)
            fname = "{}.pkl".format(dirname)

            # Load data (deserialize)
            fpath = os.path.join(base_dir, dirname, fname)
            with open(fpath, "rb") as handle:
                simulation_results = pickle.load(handle)

        # iterate over scenarios and countries
        for scenario, country_results_dict in simulation_results.items():
            for country, df_transition_numbers in country_results_dict.items():

                # select scenario-specific weighting coefficient
                coeffy_weight = self.transition_pool_weights[scenario]

                # calc stats
                n_obs = df_transition_numbers.shape[0]
                n_workers = df_transition_numbers[coeffy_weight].sum().astype(int)
                # ---------------------------------------------------------------------
                # REGIONAL AGGREGATION
                # ---------------------------------------------------------------------

                # Calculate avg number of transitions per threatened job
                df_transition_numbers_by_nuts = df_transition_numbers.groupby(
                    "NUTS_ID"
                ).sum()

                df_transition_numbers_by_nuts["n_viable_transitions_rel"] = (
                    df_transition_numbers_by_nuts.n_viable_transitions_sum
                    / df_transition_numbers_by_nuts[coeffy_weight]
                )

                # to gdf
                gdf_transition_numbers_by_nuts = pd.merge(
                    self.gdf[self.gdf["CNTR_CODE"] == country],
                    df_transition_numbers_by_nuts.reset_index(),
                    on="NUTS_ID",
                    how="left",
                )

                # ---------------------------------------------------------------------
                # REGIONAL PLOTS
                # ---------------------------------------------------------------------
                dirname = "{}_simulations_{}_optimisation_{}".format(
                    version, transition_optimisation, self.year
                )

                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))

                cmap_earnings = plt.get_cmap("coolwarm_r", 8)
                cmap_earnings.set_over("darkblue")
                cmap_earnings.set_under("darkred")

                # upper limits for colorbars
                vmax_transitions = 6
                vmax_wages = 40  # million euro

                # discrete bins with a hard cutoff at 1
                # [0,1) red, [1,2), [2,3), [3,4), [4,5), [5,6), [6,vmax]
                bounds_transitions = [0, 1, 2, 3, 4, 5, 6, np.nextafter(vmax_transitions, np.inf)]

                # one color per bin interval (len(bounds)-1)
                colors_transitions = [
                    "lightcoral",  # [0,1)
                    "#deebf7",  # [1,2)
                    "#c6dbef",  # [2,3)
                    "#9ecae1",  # [3,4)
                    "#6baed6",  # [4,5)
                    "#3182bd",  # [5,6)
                    "#08519c",  # [6,vmax]
                ]

                cmap_transitions, norm_transitions, ticks_transitions = plotting_utils.thresholded_discrete_cmap_and_norm(
                    base_colors=colors_transitions,
                    bounds=bounds_transitions,
                    over_color="black",
                )

                # transition numbers
                gdf_transition_numbers_by_nuts.plot(
                    column="n_viable_transitions_rel",
                    legend=True,
                    cmap=cmap_transitions,
                    norm=norm_transitions,  # <<< NEW (this is the important part)
                    legend_kwds={
                        "label": "Viable transitions per worker [-]",
                        "fraction": 0.03,
                        "extend": "max",
                        "ticks": ticks_transitions,
                    },
                    missing_kwds={
                        "facecolor": "lightgrey",
                        "hatch": "/",
                        "edgecolor": "grey",
                    },
                    edgecolor="grey",
                    linewidth=0.5,
                    ax=ax1,
                )
                gdf_transition_numbers_by_nuts.apply(
                    lambda x: ax1.annotate(
                        text=x.NUTS_NAME,
                        xy=x.geometry.centroid.coords[0],
                        ha="center",
                        alpha=0.5,
                        rotation=0,
                        fontsize=8,
                    ),
                    axis=1,
                )

                ax1.set_title(
                    "$Average = {:.2f}$".format(
                        gdf_transition_numbers_by_nuts.n_viable_transitions_rel.mean()
                    )
                )
                ax1.axis("off")

                # earnings losses
                v = vmax_wages
                gdf_transition_numbers_by_nuts.plot(
                    column="earnings_delta_closest_switch_sum_mio",
                    legend=True,
                    cmap=cmap_earnings,
                    vmin=-v,
                    vmax=v,
                    legend_kwds={
                        "label": "$\Delta$ Annual earnings [M€ (2023)]",
                        "fraction": 0.03,
                        "extend": "both",
                    },
                    missing_kwds={
                        "facecolor": "lightgrey",
                        "hatch": "/",
                        "edgecolor": "grey",
                    },
                    edgecolor="grey",
                    linewidth=0.5,
                    ax=ax2,
                )

                gdf_transition_numbers_by_nuts.apply(
                    lambda x: ax2.annotate(
                        text=x.NUTS_NAME,
                        xy=x.geometry.centroid.coords[0],
                        ha="center",
                        alpha=0.5,
                        rotation=0,
                        fontsize=8,
                    ),
                    axis=1,
                )

                ax2.set_title(
                    "$Total = {:.2f}~M€~(2023)$".format(
                        gdf_transition_numbers_by_nuts.earnings_delta_closest_switch_sum_mio.sum()
                    )
                )
                ax2.axis("off")

                # layout
                if show_plots:
                    fig.suptitle(
                        "Country: {country}\n Year: {year}\n Scenario: {scenario}\n Workers: {n_workers}\n N: {n_obs}\n Optimise: {optimise}\n Version: {version}".format(
                            version=version,
                            scenario=scenario.capitalize(),
                            country=country,
                            year=year,
                            optimise=transition_optimisation,
                            n_workers=n_workers,
                            n_obs=n_obs,
                        )
                    )
                fig.tight_layout()
                fig.subplots_adjust(top=1.4)

                fname = "results_{}_{}_{}_{}.png".format(
                    country, year, "regional", scenario
                )
                plt.savefig(
                    os.path.join(
                        base_dir,
                        dirname,
                        fname,
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )

                if not show_plots:
                    plt.cla()
                    fig.clf()
                    plt.close(fig)

                # ---------------------------------------------------------------------
                # INDUSTRY PLOTS (earnings losses and transition numbers)
                # ---------------------------------------------------------------------
                # --- NEW: align sector plot with map mask ---
                cntr_missing = ['IT', 'NL', 'DE', 'HU', 'AT', 'RO', 'PL', 'ES', 'LT', 'SI', 'CY', 'BE', 'CZ', 'HR',
                                'IS', 'LV']

                # work on a local copy so we don't touch the original
                df_sector = df_transition_numbers.copy()

                # earnings-per-worker variable used by the sector plot
                var_earn = "earnings_delta_rel_w" if "earnings_delta_rel_w" in df_sector.columns else "earnings_delta_rel"

                # if the current country is in the missing list, blank out earnings (consistent with map)
                if country in cntr_missing and var_earn in df_sector.columns:
                    df_sector[var_earn] = np.nan

                vars = [var_earn, "n_viable_transitions"] #NEW
                var_labels = [
                    "$\\Delta$ Annual earnings per worker (€)",
                    "Viable transitions per worker [-]",
                ]
                var_fname = ["earnings_per_worker", "transitions"]

                for i, var in enumerate(vars):
                    fig, (ax1, ax2) = plt.subplots(
                        ncols=2,
                        figsize=(10, 5),
                        sharey=True,
                        sharex=False,
                        gridspec_kw={"width_ratios": [0.7, 0.3]},
                    )

                    y_order = (
                        df_sector.groupby("NACE2_1D_label")
                        .median()[var]
                        .sort_values(ascending=False)
                        .index.values
                    )

                    # left
                    if var_fname[i] == "earnings_per_worker":
                        sns.boxplot(
                            data=df_sector,
                            x=var,
                            y="NACE2_1D_label",
                            orient="h",
                            fliersize=1,
                            showmeans=True,
                            meanprops={
                                "marker": "^",
                                "markerfacecolor": "white",
                                "markeredgecolor": "black",
                                "markersize": "5",
                            },
                            order=y_order,
                            palette="RdYlGn_r",
                            ax=ax1,
                        )

                        # right
                        # CHANGE: worker-weighted mean per sector (using scenario weight)
                        earn_col = var  # this is your per-worker column, e.g. 'earnings_delta_rel_w'
                        wt_col = self.transition_pool_weights[scenario]

                        wmean_df = (
                            df_sector
                            .dropna(subset=[earn_col, wt_col])
                            .groupby("NACE2_1D_label", as_index=False)
                            .apply(lambda g: pd.Series({
                                "wmean": np.average(g[earn_col].astype(float), weights=g[wt_col].astype(float))
                            }))
                        )

                        # keep the same order as left panel
                        wmean_df["NACE2_1D_label"] = pd.Categorical(wmean_df["NACE2_1D_label"],
                                                                    categories=y_order, ordered=True)
                        wmean_df = wmean_df.sort_values("NACE2_1D_label")

                        ax2.barh(
                            wmean_df["NACE2_1D_label"],
                            wmean_df["wmean"],
                            color=sns.color_palette("RdYlGn_r", len(wmean_df))
                        )

                        if ax2.containers:
                            ax2.bar_label(ax2.containers[-1], fmt="%.0f", label_type="center", fontsize=8)

                        ax2.set_xlabel("$\\Delta$ Annual earnings per worker (€)")
                        ax2.axvline(0, linestyle="-", color="grey", zorder=0)

                    elif var_fname[i] == "transitions":
                        sns.barplot(
                            data=df_sector,
                            x=var,
                            y="NACE2_1D_label",
                            orient="h",
                            estimator=np.mean,
                            ci="sd",
                            order=y_order,
                            palette="RdYlGn_r",
                            ax=ax1,
                        )
                        ax1.axvline(1, linestyle="-", color="lightcoral", zorder=0)

                    for ax in [ax1, ax2]:
                        ax.axvline(0, linestyle="-", color="grey", zorder=0)
                        ax.grid(linestyle=":")
                        ax.set_xlabel(None)
                        ax.set_ylabel(None)

                    # labelling
                    fig.text(0.7, 0.0, var_labels[i], ha="center")

                    if show_plots:
                        fig.suptitle(
                            "Country: {country} | Year: {year} | Scenario: {scenario} | Workers: {n_workers} | N: {n_obs} | Optimise: {optimise} | Version: {version}".format(
                                version=version,
                                scenario=scenario.capitalize(),
                                country=country,
                                year=year,
                                optimise=transition_optimisation,
                                n_workers=int(n_workers),
                                n_obs=n_obs,
                            ),
                            fontsize="small",
                        )
                    fig.tight_layout()
                    fig.subplots_adjust(top=0.9)

                    # layout
                    sns.despine()

                    # save
                    fname = "results_{}_{}_{}_{}_{}.png".format(
                        country, year, "sectoral", var_fname[i], scenario
                    )
                    plt.savefig(
                        os.path.join(
                            useful_paths.figure_dir,
                            "reskilling_simulation",
                            dirname,
                            fname,
                        ),
                        dpi=300,
                        bbox_inches="tight",
                    )

                    if not show_plots:
                        plt.cla()
                        fig.clf()

    def visualise_simulation_results_eu(
        self,
        simulation_results=None,
        base_dir=os.path.join(useful_paths.figure_dir, "reskilling_simulation"),
        year=2023,
        transition_optimisation="wage",
        reskilling_version="baseline",
        step=1,
        show_title=True,
        show_annotations=False,
        show_map_boxplots=False,
        show_plots=False,
        cbar_fraction=0.025,
        vmax_transitions=8,
        vmax_wages=100,
        dynamic_wages_cmap=False,
        title_fontsize="small",
        save_tables=True,
        industry_subset_paper=True,
        regional_constraint=True,
        combine_vars_in_sector_plot=True,
    ):

        print(">>> ENTERING visualise_simulation_results_eu")

        # params
        xmin, xmax, ymin, ymax = bbox_eu_epsg_3035
        img_ext = "pdf" #NEW
        mpl.rcParams["pdf.fonttype"] = 42  #NEW, embed TrueType
        mpl.rcParams["ps.fonttype"] = 42 #NEW

        # construct path of in-file
        reg_constraint_str = "regC" if regional_constraint else "no-regC"

        # format: "{sim_version}_{opt_target}-opt_{reg_constraint}_{year}"
        dirname = self.dirname_out_reg.format(
            sim_version=reskilling_version,
            opt_target=transition_optimisation,
            reg_constraint=reg_constraint_str,
            year=self.year,
        )

        utils.ccdir(os.path.join(base_dir, dirname))
        fname = "{}.pkl".format(dirname)

        # read file if no results are passed
        if simulation_results is None:
            # Load data (deserialize)
            fpath = os.path.join(base_dir, dirname, fname)
            with open(fpath, "rb") as handle:
                simulation_results = pickle.load(handle)

        # iterate over scenarios and countries
        for scenario, country_results_dict in simulation_results.items():

            # NEW: guard against empty regional‐constraint runs
            if not country_results_dict:
                print(f"[INFO] no data to visualize for regional_constraints run, skipping.")
                return

            # concatenate data across countries
            df_transition_numbers = pd.concat(list(country_results_dict.values()))

            # optionally reset multi-index
            if isinstance(df_transition_numbers.index, pd.MultiIndex):
                df_transition_numbers = df_transition_numbers.reset_index()

            print(df_transition_numbers.columns)
            
            # cast obj to float
            df_transition_numbers = df_transition_numbers.infer_objects()
            df_transition_numbers["AGE"] = pd.to_numeric(df_transition_numbers["AGE"])

            # Vectorized creation of per-step M€ totals
            # 1) find totals in € (…_sum_step_<s>) and coerce to numeric
            sum_eur_cols = df_transition_numbers.filter(
                regex=r"^earnings_delta_closest_switch_sum_step_\d+$"
            ).columns
            if len(sum_eur_cols):
                df_transition_numbers[sum_eur_cols] = df_transition_numbers[sum_eur_cols].apply(
                    pd.to_numeric, errors="coerce"
                )
                # 2) build the entire M€ DataFrame at once and concat once
                mio_df = (df_transition_numbers[sum_eur_cols] / 1e6).rename(
                    columns=lambda c: c.replace("_sum_step_", "_sum_mio_step_")
                )
                df_transition_numbers = pd.concat([df_transition_numbers, mio_df], axis=1, copy=False)

            # 3) sanity check: do we have the requested step in M€?
            _needed = f"earnings_delta_closest_switch_sum_mio_step_{step}"
            if _needed not in df_transition_numbers.columns:
                raise ValueError(
                    f"Column '{_needed}' not found. "
                    f"Available: {sorted(c for c in df_transition_numbers.columns if c.startswith('earnings_delta_closest_switch_sum_mio_step_'))}"
                )

            # select scenario-specific weighting coefficient
            coeffy_weight = self.transition_pool_weights[scenario]

            # calc stats
            n_obs = df_transition_numbers.shape[0]
            n_workers = df_transition_numbers[coeffy_weight].sum()  # .astype(int)

            ind_label = "NACE2_1D_label"
            if not ind_label in df_transition_numbers.columns.values:
                df_transition_numbers = pd.merge(
                    df_transition_numbers,
                    self.nace_labels,
                    on="NACE2_1D",
                    how="left",
                    suffixes=("", "_drop"),
                )

            # ---------------------------------------------------------------------
            # REGIONAL AGGREGATION
            # ---------------------------------------------------------------------

            # Calculate avg number of transitions per threatened job, sum of earnings
            #  changes, etc.
            # print(df_transition_numbers.columns)
            # - per-worker step columns: average
            # - totals: sum
            # - everything else: first
            mean_group = df_transition_numbers.columns.str.startswith(
                ("n_viable_transitions_step_", "transition_viable", "AGE", "earnings_delta_closest_switch_step_")
            )
            sum_group = df_transition_numbers.columns.str.startswith(
                ("earnings_delta_closest_switch_sum_", "earnings_delta_closest_switch_sum_mio_step_", "COEFFY", "NOBS")
            )
            rest_group = ~(mean_group | sum_group)

            # define aggregation pools
            cols_mean_group = df_transition_numbers.columns[mean_group]
            cols_sum_group = df_transition_numbers.columns[sum_group]
            cols_rest_group = df_transition_numbers.columns[rest_group]

            # create agg func
            agg_funcs = {}
            for col in df_transition_numbers.columns:
                if col in cols_mean_group:
                    agg_funcs[col] = "mean"
                elif col in cols_sum_group:
                    agg_funcs[col] = "sum"
                elif col in cols_rest_group:
                    agg_funcs[col] = "first"

            # aggregate
            df_transition_numbers_by_nuts = (
                df_transition_numbers.groupby("NUTS_ID")
                .aggregate(agg_funcs)
                .drop(columns=["NUTS_ID"])
            )

            # to gdf
            gdf_transition_numbers_by_nuts = pd.merge(
                self.gdf[
                    self.gdf["CNTR_CODE"].isin(df_transition_numbers["COUNTRYW"].values)
                ],
                df_transition_numbers_by_nuts.reset_index(),
                on="NUTS_ID",
                how="left",
            )

            # map fix for countries like NL and MT (weird NUTS2 reporting)
            if not regional_constraint:
                step_col = f"n_viable_transitions_step_{step}"
                # country-level mean of the plotted metric
                mean_by_cty = (
                    df_transition_numbers
                    .groupby("COUNTRYW")[step_col]
                    .mean()
                )
                # detect countries whose aggregated DF has no real NUTS2 keys in the geometry
                bad_ctrs = []
                for c in df_transition_numbers["COUNTRYW"].unique():
                    nuts2_in_geom = set(
                        self.gdf.loc[
                            (self.gdf["CNTR_CODE"] == c) & (self.gdf["LEVL_CODE"] == 2),
                            "NUTS_ID"
                        ]
                    )
                    nuts2_in_df = set(df_transition_numbers_by_nuts.index) & nuts2_in_geom
                    if not nuts2_in_df:  # only has e.g. "NL00" (no NUTS2 overlap)
                        bad_ctrs.append(c)
                # paint the mean across all polygons for those countries
                for c in bad_ctrs:
                    gdf_transition_numbers_by_nuts.loc[
                        gdf_transition_numbers_by_nuts["CNTR_CODE"] == c, step_col
                    ] = mean_by_cty.get(c, np.nan)

            # ---------------------------------------------------------------------
            # REGIONAL PLOTS
            # ---------------------------------------------------------------------
            # SPLIT: two separate figures per step — one transitions map, one income map —
            # so each can be uploaded independently. ax1/ax3 belong to the transitions figure,
            # ax2/ax4 to the income figure. The plotting body below is unchanged (it just
            # draws onto ax1..ax4, which now live on two figures rather than one).
            if not show_map_boxplots:
                fig_trans, ax1 = plt.subplots(figsize=(10, 10))
                fig_inc, ax2 = plt.subplots(figsize=(10, 10))
                ax3 = ax4 = None
            else:
                fig_trans, (ax1, ax3) = plt.subplots(
                    nrows=2,
                    figsize=(10, 11),
                    gridspec_kw={"height_ratios": [0.9, 0.1], "hspace": 0},
                )
                fig_inc, (ax2, ax4) = plt.subplots(
                    nrows=2,
                    figsize=(10, 11),
                    gridspec_kw={"height_ratios": [0.9, 0.1], "hspace": 0},
                )
            vmax_transitions = 6
            bounds_transitions = [0, 1, 2, 3, 4, 5, 6, np.nextafter(vmax_transitions, np.inf)]

            print(">>> Using custom palette in EU visualiser")

            colors_transitions = [
                "lightcoral",  # [0,1)
                "#deebf7",  # [1,2)
                "#c6dbef",  # [2,3)
                "#9ecae1",  # [3,4)
                "#6baed6",  # [4,5)
                "#3182bd",  # [5,6)
                "#08519c",  # [6,vmax]
            ]

            cmap_transitions, norm_transitions, ticks_transitions = plotting_utils.thresholded_discrete_cmap_and_norm(
                base_colors=colors_transitions,
                bounds=bounds_transitions,
                over_color="black",
            )

            # transition numbers
            gdf_transition_numbers_by_nuts.plot(
                column="n_viable_transitions_step_{}".format(step),
                legend=True,
                cmap=cmap_transitions,
                norm=norm_transitions,
                legend_kwds={
                    "label": "Job transitions per worker [-]",
                    "fraction": cbar_fraction,
                    "extend": "max",
                    "ticks": ticks_transitions,
                },
                missing_kwds={
                    "facecolor": "lightgrey",
                    "hatch": "/",
                    "edgecolor": "grey",
                },
                edgecolor="grey",
                linewidth=0.5,
                ax=ax1,
            )

            if show_annotations:
                gdf_transition_numbers_by_nuts.apply(
                    lambda x: ax1.annotate(
                        text=x.NUTS_NAME,
                        xy=x.geometry.centroid.coords[0],
                        ha="center",
                        alpha=0.5,
                        rotation=0,
                        fontsize=8,
                    ),
                    axis=1,
                )

            ax1.set_title(
                "$Average = {:.2f}$".format(
                    gdf_transition_numbers_by_nuts[
                        "n_viable_transitions_step_{}".format(step)
                    ].mean()
                )
            )

            # PER-WORKER MAP FROM TOTALS ÷ WEIGHT

            # 1) mask countries with missing earnings data on the *totals* column
            cntr_missing = ['IT', 'NL', 'DE', 'HU', 'AT', 'RO', 'PL', 'ES', 'LT', 'SI', 'CY', 'BE', 'CZ', 'HR', 'IS',
                            'LV']
            tot_col = f"earnings_delta_closest_switch_sum_step_{step}"  # totals in €
            norm_factor = self.transition_pool_weights[scenario]  # e.g. COEFFY_share_unviable_to_decarbonize
            # scenario-appropriate worker noun (the income map is drawn for both flows; the
            # inward/shortage flow is NOT "at-risk", so relabel accordingly)
            _worker_noun = {"at_risk": "at-risk worker",
                            "shortage": "worker (inward flow)",
                            "high_carbon": "high-carbon worker"}.get(scenario, "worker")

            gdf_transition_numbers_by_nuts.loc[
                gdf_transition_numbers_by_nuts["CNTR_CODE"].isin(cntr_missing),
                tot_col
            ] = np.nan

            # 2) compute weighted per-worker (= totals / weight_sum)
            per_worker_col = f"earnings_delta_per_worker_step_{step}"  # new column name
            gdf_transition_numbers_by_nuts[per_worker_col] = (
                    gdf_transition_numbers_by_nuts[tot_col]
                    / gdf_transition_numbers_by_nuts[norm_factor]
            )

            # 2b) guard against zero/neg weights → NaN, and drop infs
            zero_w = gdf_transition_numbers_by_nuts[norm_factor] <= 0
            gdf_transition_numbers_by_nuts.loc[zero_w, per_worker_col] = np.nan
            gdf_transition_numbers_by_nuts[per_worker_col] = (
                gdf_transition_numbers_by_nuts[per_worker_col]
                .replace([np.inf, -np.inf], np.nan)
            )

            # 3) color scale (€, not M€)
            if dynamic_wages_cmap:
                vmax_wages = (
                    gdf_transition_numbers_by_nuts[per_worker_col]
                    .abs().quantile(0.98)
                )

            # keep your colormap construction
            cmap_earnings = plt.get_cmap("coolwarm_r", max(4, int((vmax_wages / 1000) * 4)))
            cmap_earnings.set_over("darkblue")
            cmap_earnings.set_under("darkred")

            # 4) ax2: earnings per worker map (weighted)
            gdf_transition_numbers_by_nuts.plot(
                column=per_worker_col,
                legend=True,
                cmap=cmap_earnings,
                vmin=-vmax_wages,
                vmax=vmax_wages,
                legend_kwds={
                    "label": f"Avg. annual income change per {_worker_noun} (€), regional population mean",
                    "fraction": cbar_fraction,
                    "extend": "both",
                },
                missing_kwds={
                    "facecolor": "lightgrey",
                    "hatch": "/",
                    "edgecolor": "grey",
                },
                edgecolor="grey",
                linewidth=0.5,
                ax=ax2,
            )

            if show_annotations:
                gdf_transition_numbers_by_nuts.apply(
                    lambda x: ax2.annotate(
                        text=x.NUTS_NAME,
                        xy=x.geometry.centroid.coords[0],
                        ha="center",
                        alpha=0.5,
                        rotation=0,
                        fontsize=8,
                    ),
                    axis=1,
                )

            # EU total: the per-cell € sums in the result frame are FAN-OUT-INFLATED
            # (each at-risk worker-group is replicated across its feasible targets), so their
            # naive sum overstates the true fiscal total by ~1-2 orders of magnitude — and the
            # inflation is NOT spatially uniform, so it cannot be undone with one global factor.
            # The per-worker map value itself is correct (the fan-out cancels in tot/weight).
            # Rebuild the total the correct way: per-worker € × TRUE raw headcount per region
            # (Σ pool-weight over the raw LFS, non-fanned), summed over the mapped regions.
            _true_hc = None
            if (self.lfs_data is not None) and (norm_factor in getattr(self.lfs_data, "columns", [])):
                _true_hc = self.lfs_data.groupby("NUTS_ID")[norm_factor].sum()
            if _true_hc is not None:
                _pw_by_nuts = gdf_transition_numbers_by_nuts.set_index("NUTS_ID")[per_worker_col]
                eu_total_mio = float(
                    (_pw_by_nuts * _true_hc.reindex(_pw_by_nuts.index)).sum(skipna=True)
                ) / 1e6
                ax2.set_title(
                    "Regional population mean per {} (total, mapped regions = {:.0f} M€, 2023)".format(_worker_noun, eu_total_mio)
                )
            else:
                # true (non-fanned) pool headcount unavailable here -> omit the total rather
                # than print the fan-out-inflated one. (Applies to plot-only regen of scenarios
                # whose pool weight is model-internal, e.g. shortage's COEFFY_share_shortage.)
                ax2.set_title("Regional population mean per {}".format(_worker_noun))

            # EU BBOX
            for ax in [ax1, ax2]:
                ax.axis("off")
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

                # add country borders (NUTS 0)
                self.gdf_all_levels[
                    self.gdf_all_levels["LEVL_CODE"] == 0
                ].geometry.boundary.plot(
                    ax=ax, color=None, edgecolor="black", linewidth=0.5
                )

            if show_map_boxplots:
                # ax3: distribution across regions (transitions)
                sns.boxplot(
                    x=gdf_transition_numbers_by_nuts[f"n_viable_transitions_step_{step}"],
                    ax=ax3,
                )

                # ax4: distribution across regions (earnings) # NEW: (earnings per worker)
                sns.boxplot(
                    x=gdf_transition_numbers_by_nuts[per_worker_col],
                    ax=ax4,
                )

            # --- legibility for two-column print: enlarge the "Average=" title and the
            #     colorbar label + tick labels (defaults are unreadable at final size) ---
            MAP_TITLE_FS, MAP_CBAR_LABEL_FS, MAP_CBAR_TICK_FS = 22, 20, 17
            for _ax in (ax1, ax2):
                if _ax is not None and _ax.get_title():
                    _ax.title.set_fontsize(MAP_TITLE_FS)
            for _f, _mains in [(fig_trans, {ax1, ax3}), (fig_inc, {ax2, ax4})]:
                for _cax in _f.axes:               # colorbar axes = figure axes that are not the map/box axes
                    if _cax not in _mains:
                        _cax.yaxis.label.set_size(MAP_CBAR_LABEL_FS)
                        _cax.xaxis.label.set_size(MAP_CBAR_LABEL_FS)
                        _cax.tick_params(labelsize=MAP_CBAR_TICK_FS)

            # layout + save: ONE figure per panel (transitions / income) so they can be
            # uploaded separately. (Previously a single side-by-side double-map saved once.)
            # Unified naming scheme: {flow}_{program}_{metric}_{scope}_stepNN.{ext}
            _PROGRAM = {"reskill-optimal": "tailored", "reskill-coreRanked": "transferable",
                        "reskill-green": "green", "reskill-digital": "digital"}
            program = _PROGRAM.get(reskilling_version, reskilling_version)
            scope = "regC" if regional_constraint else "noRegC"
            _panels = [("transitions_map", fig_trans), ("income_map_popmean_eur", fig_inc)]
            if show_title:
                _suptitle = (
                    "Country: {country}\n Year: {year}\n Scenario: {scenario}\n Workers: {n_workers}\n N: {n_obs}\n Optimise: {optimise}\n Simulation: {version}\n Regional: {regional_constraint}\n Journey step: {journey_step}".format(
                        version=reskilling_version,
                        scenario=scenario.capitalize(),
                        country="EU",
                        year=year,
                        optimise=transition_optimisation,
                        n_workers=int(n_workers),
                        n_obs=n_obs,
                        regional_constraint=regional_constraint,
                        journey_step=step,
                    )
                )
                for _lbl, _fig in _panels:
                    _fig.suptitle(_suptitle, fontsize=title_fontsize)

            for _lbl, _fig in _panels:
                _fig.tight_layout()
                _fig.savefig(
                    os.path.join(base_dir, dirname,
                                 "{}_{}_{}_{}_step{:02d}.{}".format(scenario, program, _lbl, scope, step, img_ext)),
                    bbox_inches="tight",  # dpi not needed for vector PDFs
                )
                if not show_plots:
                    _fig.clf()
                    plt.close(_fig)

            # save numbers
            if save_tables:
                vars = [per_worker_col, f"n_viable_transitions_step_{step}"]
                res = gdf_transition_numbers_by_nuts[vars].describe()
                res.to_csv(
                    os.path.join(
                        base_dir,
                        dirname,
                        "{}_{}_{}_{}.{}".format(
                            "EU", year, "regional", scenario, "csv"
                        ),
                    )
                )
            # ---------------------------------------------------------------------
            # INDUSTRY PLOTS (earnings losses and transition numbers)
            # ---------------------------------------------------------------------
            # --- NEW: mask countries with missing earnings for sector plots, to match the map ---
            cntr_missing = ['IT', 'NL', 'DE', 'HU', 'AT', 'RO', 'PL', 'ES', 'LT', 'SI', 'CY', 'BE', 'CZ', 'HR', 'IS',
                            'LV']

            # work on a filtered copy so maps/earlier steps remain unchanged
            df_sector = df_transition_numbers[~df_transition_numbers["COUNTRYW"].isin(cntr_missing)].copy()

            vars = [
                "earnings_delta_closest_switch_step_{}".format(step),
                "n_viable_transitions_step_{}".format(step),
            ]
            var_labels = [
                "$\\Delta$ Annual earnings per worker (€)",
                "Job transitions per worker",
            ]
            var_fname = ["earnings_per_worker", "transitions"]
            df_sector = df_sector.replace(
                to_replace={"NACE2_1D_label": self.nace_mapping}
            )

            # subset relevant industries for paper
            if industry_subset_paper:
                industry_subset = self.nace_labels.loc[
                    self.nace_labels["paper_selection"] == True, "NACE2_1D_label_short"
                ]
                df_sector = df_sector.loc[
                    df_sector["NACE2_1D_label"].isin(industry_subset)
                ]

            if not combine_vars_in_sector_plot:
                for i, var in enumerate(vars):
                    fname_snippet = var_fname[i]
                    fig, (ax1, ax2) = plt.subplots(
                        ncols=2,
                        figsize=(10, 5),
                        sharey=True,
                        sharex=False,
                        gridspec_kw={"width_ratios": [0.7, 0.3]},
                    )

                    # NEW: add weighted per-worker € series that reconciles to totals
                    norm_col = self.transition_pool_weights[scenario]  # e.g., COEFFY_...
                    df_sector[f"earnings_pw_w_step_{step}"] = (  # ADD
                            df_sector[f"earnings_delta_closest_switch_sum_step_{step}"]
                            / df_sector[norm_col]
                    )
                    df_sector.loc[df_sector[norm_col] <= 0, f"earnings_pw_w_step_{step}"] = np.nan

                    # NEW: order by what is plotted on the left
                    if var_fname[i] == "earnings_per_worker":
                        metric_left = f"earnings_pw_w_step_{step}"
                    else:
                        metric_left = var

                    y_order = (
                        df_sector.groupby("NACE2_1D_label")
                        .aggregate({metric_left: "median"})
                        .sort_values(by=metric_left, ascending=False)
                        .index.values
                    )

                    from matplotlib.container import BarContainer
                    def _main_bar_container(ax):
                        bar_containers = [c for c in ax.containers if isinstance(c, BarContainer)]
                        return max(bar_containers, key=lambda bc: len(bc.patches)) if bar_containers else None

                    # left
                    if var_fname[i] == "earnings_per_worker":
                        sns.boxplot(
                            data=df_sector,
                            x=f"earnings_pw_w_step_{step}",
                            y="NACE2_1D_label",
                            orient="h",
                            showfliers=False,
                            showmeans=True,
                            meanprops={
                                "marker": "^",
                                "markerfacecolor": "white",
                                "markeredgecolor": "black",
                                "markersize": "5",
                            },
                            order=y_order,
                            palette="RdYlGn_r",
                            ax=ax1,
                        )

                        # right (worker-weighted mean per sector)
                        earn_col = f"earnings_pw_w_step_{step}"
                        wt = self.transition_pool_weights[scenario]  # e.g. "COEFFY_share_unviable_to_decarbonize"

                        # guard for missing columns
                        if earn_col in df_sector.columns and wt in df_sector.columns:
                            wmean_df = (
                                df_sector
                                .dropna(subset=[earn_col, wt])
                                .groupby("NACE2_1D_label", as_index=False)
                                .apply(lambda g: pd.Series({
                                    "wmean": np.average(g[earn_col].astype(float), weights=g[wt].astype(float))
                                }))
                                .sort_values("wmean", ascending=False)
                            )
                            # keep the same order you computed for the left panel
                            wmean_df["NACE2_1D_label"] = pd.Categorical(wmean_df["NACE2_1D_label"], categories=y_order,
                                                                        ordered=True)
                            wmean_df = wmean_df.sort_values("NACE2_1D_label")

                            ax2.barh(wmean_df["NACE2_1D_label"], wmean_df["wmean"],
                                     color=sns.color_palette("RdYlGn_r", len(wmean_df)))

                            if ax2.containers:
                                ax2.bar_label(ax2.containers[-1], fmt="%.0f", label_type="center", fontsize=8)

                            ax2.set_xlabel("$\\Delta$ Annual earnings per worker (€)")
                            ax2.axvline(0, linestyle="-", color="grey", zorder=0)
                        else:
                            ax2.text(0.5, 0.5, "Missing columns for weighted mean", transform=ax2.transAxes,
                                     ha="center")

                    elif var_fname[i] == "transitions":
                        sns.barplot(
                            data=df_sector,
                            x=var,
                            y="NACE2_1D_label",
                            orient="h",
                            estimator=np.mean,
                            ci="sd",
                            order=y_order,
                            palette="RdYlGn_r",
                            ax=ax1,
                        )
                        ax1.axvline(1, linestyle="-", color="lightcoral", zorder=0)
                        ax1.set_xlim(0)
                        ax2.axvline(0, linestyle="-", color="grey", zorder=0) #NEW

                    for ax in [ax1, ax2]:
                        ax.axvline(0, linestyle="-", color="grey", zorder=0)
                        ax.grid(linestyle=":")
                        ax.set_xlabel(None)
                        ax.set_ylabel(None)

                    # labelling
                    fig.text(0.7, 0.0, var_labels[i], ha="center")

                    if show_title:
                        fig.suptitle(
                            "Country: {country} | Year: {year} | Scenario: {scenario} | Workers: {n_workers} | N: {n_obs} | Optimise: {optimise} | Version: {version} | Regional: {regional_constraint} | Journey step: {journey_step}".format(
                                version=reskilling_version,
                                scenario=scenario.capitalize(),
                                country="EU",
                                year=year,
                                optimise=transition_optimisation,
                                n_workers=int(n_workers),
                                n_obs=n_obs,
                                regional_constraint=regional_constraint,
                                journey_step=step,
                            ),
                            fontsize=title_fontsize,
                        )
                    fig.tight_layout()
                    fig.subplots_adjust(top=0.9)

                    # layout
                    sns.despine()

                    # save
                    fname = "{}_{}_{}_{}_{}_step_{}.{}".format(
                        "EU", year, "sectoral", fname_snippet, scenario, step, img_ext
                    )
                    plt.savefig(
                        os.path.join(base_dir, dirname, fname),
                        bbox_inches="tight",
                    )
            else:
                fname_snippet = "combined"
                var = "n_viable_transitions_step_{}".format(step)
                fig, (ax1, ax2) = plt.subplots(
                    ncols=2,
                    figsize=(10, 5),
                    sharey=True,
                    sharex=False,
                    gridspec_kw={"width_ratios": [0.7, 0.3]},
                )

                # weighted earnings per worker, consistent with the map (totals ÷ headcount)
                norm_col = self.transition_pool_weights[scenario]
                df_sector[f"earnings_pw_w_step_{step}"] = (
                        df_sector[f"earnings_delta_closest_switch_sum_step_{step}"] / df_sector[norm_col]
                )
                df_sector.loc[df_sector[norm_col] <= 0, f"earnings_pw_w_step_{step}"] = np.nan

                y_order = (
                    df_sector.groupby("NACE2_1D_label")
                    .aggregate({var: "mean"})
                    .sort_values(by=var, ascending=False)
                    .index.values
                )

                # fix bar labels
                from matplotlib.container import BarContainer
                def _main_bar_container(ax):
                    bar_containers = [c for c in ax.containers if isinstance(c, BarContainer)]
                    return max(bar_containers, key=lambda bc: len(bc.patches)) if bar_containers else None

                # left
                sns.barplot(
                    data=df_sector,
                    x="n_viable_transitions_step_{}".format(step),
                    y="NACE2_1D_label",
                    orient="h",
                    estimator=np.mean,
                    ci="sd",
                    order=y_order,
                    palette="RdYlGn_r",
                    ax=ax1,
                )
                bars_left = _main_bar_container(ax1)  #NEW
                if bars_left is not None:  #NEW
                    ax1.bar_label(bars_left, fmt="%.1f", label_type="center", color="white")  #NEW

                ax1.axvline(1, linestyle="-", color="lightcoral", zorder=0)
                ax1.set_xlim(0)
                ax1.set_xlabel("Job transitions per worker [-]")

                # right
                # CHANGE: worker-weighted mean per sector (consistent with map logic)
                earn_col = f"earnings_pw_w_step_{step}"
                wt_col = self.transition_pool_weights[scenario]

                wmean_df = (
                    df_sector
                    .dropna(subset=[earn_col, wt_col])
                    .groupby("NACE2_1D_label", as_index=False)
                    .apply(lambda g: pd.Series({
                        "wmean": np.average(g[earn_col].astype(float), weights=g[wt_col].astype(float))
                    }))
                )

                # keep same left-panel order
                wmean_df["NACE2_1D_label"] = pd.Categorical(wmean_df["NACE2_1D_label"],
                                                            categories=y_order, ordered=True)
                wmean_df = wmean_df.sort_values("NACE2_1D_label")

                ax2.barh(
                    wmean_df["NACE2_1D_label"],
                    wmean_df["wmean"],
                    color=sns.color_palette("RdYlGn_r", len(wmean_df))
                )

                if ax2.containers:
                    ax2.bar_label(ax2.containers[-1], fmt="%.0f", label_type="center", fontsize=8)

                ax2.set_xlabel("$\\Delta$ Annual earnings per worker (€)")
                ax2.axvline(0, linestyle="-", color="grey", zorder=0)

                for ax in [ax1, ax2]:
                    ax.grid(linestyle=":", axis="x")
                    ax.set_ylabel(None)

                # labelling
                if show_title:
                    fig.suptitle(
                        "Country: {country} | Year: {year} | Scenario: {scenario} | Workers: {n_workers} | N: {n_obs} | Optimise: {optimise} | Version: {version} | Regional: {regional_constraint} | Journey step: {journey_step}".format(
                            version=reskilling_version,
                            scenario=scenario.capitalize(),
                            country="EU",
                            year=year,
                            optimise=transition_optimisation,
                            n_workers=int(n_workers),
                            n_obs=n_obs,
                            regional_constraint=regional_constraint,
                            journey_step=step,
                        ),
                        fontsize=title_fontsize,
                    )
                fig.tight_layout()
                fig.subplots_adjust(top=0.9)

                # layout
                sns.despine()

                # save
                fname = "{}_{}_{}_{}_{}_step_{}.{}".format(
                    "EU", year, "sectoral", fname_snippet, scenario, step, img_ext
                )
                plt.savefig(
                    os.path.join(base_dir, dirname, fname),
                    bbox_inches="tight",
                )

                if not show_plots:
                    plt.cla()
                    fig.clf()

                # save numbers
                if save_tables:
                    vars = [per_worker_col, f"n_viable_transitions_step_{step}"]  # CHANGED (use weighted €/worker)
                    res = gdf_transition_numbers_by_nuts[vars].describe()
                    res.to_csv(
                        os.path.join(
                            base_dir,
                            dirname,
                            "{}_{}_{}_{}.{}".format(
                                "EU", year, "regional", scenario, "csv"
                            ),
                        )
                    )

if __name__ == "__main__":
    import time
    import timeit
    from data.lfs import EuLfs

    # re-run simulations & plot, or plot only (load existing pickles)?
    # MUST be True for the full run — False only loads existing pkls and runs nothing.
    # NOTE: combos whose output pkl already exists are SKIPPED (line ~3757), so this also
    # acts as resume-on-restart. Therefore results/figures/reskilling_simulation/ must be
    # EMPTY (or stale variant dirs removed) before a fresh full run, or stale outputs are
    # kept. See the launch procedure.
    rerun_simulations = True

    # ---------------------------------------------------------------------
    # Input data
    # ---------------------------------------------------------------------

    if rerun_simulations:
        # EU-LFS data
        config = utils.load_config(
            os.path.join(useful_paths.config_dir, "eu_lfs_config.yml")
        )
        lfs = EuLfs(config=config)

        lfs_data = lfs.read_preprocessed_file(
            year=2023,
            input_fname_lfs="clean_eu_lfs_merged_{year}_with_final_unweighted_shares_and_earnings_incdecil_imputed",
        )

        # patch column names for new classification
        to_rename = [
            col
            for col in lfs_data.columns
            if "-" in col and (col.startswith("share") or col.startswith("COEFFY_share"))
        ]
        rename_map = {col: col.replace("-", "_") for col in to_rename}
        lfs_data.rename(columns=rename_map, inplace=True)
        # ── DEBUG
        print("share_… columns now:", [c for c in lfs_data.columns if c.startswith("share_")])
        print("COEFFY_share_… columns now:", [c for c in lfs_data.columns if c.startswith("COEFFY_share_")])

        # add required categories
        cats_regionw = lfs_data["REGION_2DW"].cat.categories.values
        cats_region = lfs_data["REGION_2D"].cat.categories.values

        placeholder_cat = "99"

        new_cats_regionw = list(set(cats_region) - set(cats_regionw))
        new_cats_region = list(set(cats_regionw) - set(cats_region))

        new_cats_regionw.append(placeholder_cat)
        new_cats_region.append(placeholder_cat)

        lfs_data["REGION_2DW"] = lfs_data["REGION_2DW"].cat.add_categories(new_cats_regionw)
        lfs_data["REGION_2D"] = lfs_data["REGION_2D"].cat.add_categories(new_cats_region)

        # fill nans in regionw with region
        lfs_data["REGION_2DW"] = lfs_data["REGION_2DW"].fillna(lfs_data["REGION_2D"])

        # fill remaining nans with placeholder
        lfs_data["REGION_2DW"] = lfs_data["REGION_2DW"].fillna("99")
        lfs_data["REGION_2D"] = lfs_data["REGION_2D"].fillna("99")

        # update nuts id
        lfs_data["NUTS_ID"] = lfs_data["COUNTRYW"].astype("string") + lfs_data[
            "REGION_2DW"
        ].astype("string")
    else:
        lfs_data = None

    # initialise class
    rp = ReskillingPathways(
        osm_version="weighted", sim_metric="cooc", lfs_data=lfs_data, year=2023
    )

    # ---------------------------------------------------------------------
    # Simulation parameters
    # ---------------------------------------------------------------------

    # countries to analyse
    countries = [
        "AT",
        "BE",
        #"BG", #only has ISCO 2 digit level
        "CH",
        "CY",
        "CZ",
        "DE",
        "DK",
        "EL",
        "EE",
        "ES",
        "FI",
        "FR",
        "HR",
        "HU",
        "IE",
        "IS",
        "IT",
        "LT",
        "LU",
        "LV",
        #"MT", #only has ISCO 1 digit level
        "NL",
        "NO",
        "PL", #why initially dropped?
        "PT",
        "RO",
        "SE",
        #"SI", #only has ISCO 2 digit level
        "SK",
    ]

    # transition pools to analyse
    # Both published flows: at_risk (outward, the headline figures) AND shortage (inward).
    # Running shortage alone would silently drop the entire outward flow. high_carbon is an
    # alternative outward definition — add it here if the SI needs it.
    scenarios = ["at_risk", "shortage"]  # was ["shortage"] — outward must not be skipped

    # reskilling options to consider
    reskilling_modes = [
        "optimal",
        #"coreness_weighted",
        "coreness_ranked",
        "digital",
        "green",
    ]

    # optimisation target for job transitions
    optimise = "wage"

    # skill overlap requirement thresholds (viable, highly viable)
    transition_thresholds = [
        # (1.00, 3.68),  # "thresh-viable-low"
        # (1.616368047779022, 6.500853535353546),  # "thresh-low"
        # None,  # (2.25, 7.25)  # "thresh-perc"
        (3.68, 10.80),  # "thresh-emp"
    ]
    shortcuts = ["thresh-viable-isco4d-v2"]  # ["thresh-low", "thresh-perc", "thresh-emp"]

    # consideration of regional mobility constraints
    regional_constraints = [True, False] # [True, False]

    # length of reskilling journey, PER FLOW — matching how the published reference was
    # generated: at_risk (outward) at 20, shortage (inward) at 30. CONFIRMED from
    # review-copy: every at_risk pkl has step columns 0..20, every shortage pkl 0..30.
    # This matters because the result's population/share aggregates (COEFFY, NOBS,
    # COEFFY_share_*) are journey-length-dependent and the EU figures weight by them, so
    # running at_risk at 30 would NOT reconcile with the reference outward figures.
    # (Set inside the scenario loop below.)
    JOURNEY_BY_FLOW = {"at_risk": 20, "high_carbon": 20, "shortage": 30}

    # name mapping
    processing_dict = dict(zip(transition_thresholds, shortcuts))

    # ---------------------------------------------------------------------
    # Processing loop
    # ---------------------------------------------------------------------
    start = timeit.default_timer()
    for transition_threshold, shortcut in processing_dict.items():

        # map scenario names to a clean folder suffix (no spaces, no weird chars)
        SCENARIO_SUFFIX = {
            "shortage": "shortage",
            "at_risk": "at_risk",
            "high_carbon": "highcarbon",  # no underscore to keep it short/portable
        }

        for scenario in scenarios:
            # per-flow journey length + steps (outward at_risk=20, inward shortage=30),
            # matching the published reference.
            reskilling_journey_length = JOURNEY_BY_FLOW.get(scenario, 30)
            steps = np.arange(0, reskilling_journey_length + 1)
            print(f"[CONFIG] scenario={scenario}  journey={reskilling_journey_length}")

            # 1) Build a scenario-specific base results dir
            scenario_tag = SCENARIO_SUFFIX.get(scenario, str(scenario))
            results_dir = os.path.join(
                useful_paths.figure_dir,
                "reskilling_simulation",
                scenario_tag,
            )
            os.makedirs(results_dir, exist_ok=True)
            print(f"[OUT] scenario '{scenario}' → {results_dir}")

            for regional_constraint in regional_constraints:
                # run simulations for each reskilling mode
                for reskilling_mode in reskilling_modes:

                    # 2) Keep your dirname logic (regC vs no-regC already included)
                    dirname = rp.dirname_out_reg.format(
                        sim_version=rp.simulation_name[reskilling_mode],
                        opt_target=optimise,
                        reg_constraint="regC" if regional_constraint else "no-regC",
                        year=rp.year,
                    )
                    pkl_path = os.path.join(results_dir, dirname, f"{dirname}.pkl")

                    print(f"=== {dirname} (scenario={scenario}) ===")
                    print(reskilling_mode)
                    t0 = time.time()

                    # 3) Run per-scenario, pass [scenario] (not the whole list)
                    if rerun_simulations and not os.path.exists(pkl_path):
                        print("→ Running simulation…")
                        simulation_results = rp.simulate_regional(
                            level="isco_3_digit",
                            countries=countries,
                            scenarios=[scenario],  # <-- only this scenario
                            transition_optimisation=optimise,
                            reskilling=reskilling_mode,
                            reskilling_journey_length=reskilling_journey_length,
                            region_constraints=regional_constraint,
                            target_job_availability_coeffy="COEFFY_mean+sd",
                            mask_diagonal=True,
                            transition_thresholds=transition_threshold,
                            out_dir=results_dir,  # <-- scenario-specific base
                        )
                    else:
                        if rerun_simulations:
                            print("→ Pickle exists; skipping re-run")
                        else:
                            print("→ rerun_simulations=False; will load existing pickle")
                        simulation_results = None

                    print(f"→ setup + simulation took {(time.time() - t0) / 60:.1f} min")

                    # 4) Visualise into the same scenario folder
                    for step in steps:
                        rp.visualise_simulation_results_eu(
                            simulation_results=simulation_results,
                            transition_optimisation=optimise,
                            reskilling_version=rp.simulation_name[reskilling_mode],
                            step=step,
                            regional_constraint=regional_constraint,
                            base_dir=results_dir,  # <-- scenario-specific base
                            vmax_wages=3000,
                            vmax_transitions=8,
                            show_title=False,
                            title_fontsize="small",
                            cbar_fraction=0.025,
                            combine_vars_in_sector_plot=True,
                        )

    # Your statements here
    stop = timeit.default_timer()
    print("Time (min): ", (stop - start) / 60)
