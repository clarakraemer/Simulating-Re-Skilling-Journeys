import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
import pickle

import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src import utils, plotting_utils
from src.data.framework import Esco
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
        self, osm_version="weighted", sim_metric="cooc", lfs_data=None, year=2019
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

        # phaseout scenario implementations
        self.phaseout_scenarios = ["coal", "brown_techchange", "brown"]

        # coefficient weights for each scenario
        self.transition_pool_weights = {
            "coal": "COEFF",
            "brown_techchange": "COEFF_share_brown_slt",
            "brown": "COEFF_share_brown_sl",
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
        self.target_cats = ["neutral", "green"]

        # phase-out scenario specific list version for assigning GBN categories
        self.category_versions = {
            # TODO: discuss categories for coal case
            "coal": "category_sl",
            "brown_techchange": "category_slt",
            "brown": "category_sl",
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
            "random": "reskill-random",
            "computer_literacy": "reskill-digital",
            "coreness_weighted": "reskill-coreWeight",
            "coreness_percentile": "reskill-corePctl",
            "optimal": "reskill-optimal",
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
                "NUTS_RG_03M_2021_3035.shp",
                "NUTS_RG_03M_2021_3035.shp",
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
                self.nace_labels["NACE1D_label"], self.nace_labels["NACE1D_label_short"]
            )
        )

        # occupation-specific skill that unlocks most new transitions
        self.df_optimal_upskilling_per_occ = pd.read_csv(
            os.path.join(
                useful_paths.data_interim,
                "upskilling_analysis",
                "upskilling_best_100_skills_per_isco3d_occupation_merged.csv",
            ),
            index_col=0,
        )

        # coreness/centrality measures of all ESCO skills
        self.df_coreness = pd.read_pickle(
            os.path.join(
                useful_paths.data_processed, "esco", "skills_network_metrics.pkl"
            )
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
                "share_green": np.nanmean,
                "share_brown_sl": np.nanmean,
                "share_brown_slt": np.nanmean,
                "share_neutral_sl": np.nanmean,
                "share_neutral_slt": np.nanmean,
                "INCDECIL_imputed": np.nanmedian,
                "annual_earnings": np.nanmean,
            }

            isco_avg = lfs_country_subset.groupby(["ISCO", "ISCO3D_label"]).aggregate(
                agg_dict
            )

            # define the GBN category of an ISCO 3D group as the one with the
            # highest fraction (of ESCO occupations)
            isco_avg["category_sl"] = isco_avg[
                ["share_green", "share_brown_sl", "share_neutral_sl"]
            ].idxmax(axis=1)
            isco_avg["category_slt"] = isco_avg[
                ["share_green", "share_brown_slt", "share_neutral_slt"]
            ].idxmax(axis=1)

            isco_avg = isco_avg.replace(
                to_replace={
                    "category_sl": {
                        "share_green": "green",
                        "share_brown_sl": "brown",
                        "share_neutral_sl": "neutral",
                    },
                    "category_slt": {
                        "share_green": "green",
                        "share_brown_slt": "brown",
                        "share_neutral_slt": "neutral",
                    },
                }
            )

            # join
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

        Todo: not sure if the above holds true for the shortage/excess metrics.

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
            # Note: this is too slow. changed to directly work on the 3 digit matrix.
            df_occ_sim = occupation_distance.occ_sim_matrix_by_levels(
                occ_skills_mat=occ_skills_mat,
                osm_version=self.osm_version,
                sim_metric=self.sim_metric,
                upskilling_ids=upskilling_ids,
            )

        df_sim_matrix_agg = (
            df_occ_sim.groupby(level=lvl_code, axis=0)
            .aggregate(agg_func)
            .groupby(level=lvl_code, axis=1)
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

        # select all observations that are both within coal-specific occupations and
        #  industries
        if scenario == "coal":
            occ_subset = lfs_data_country["ISCO3D_label"].isin(
                list(self.coal_occupations.keys())
            )
            ind_subset = lfs_data_country["NACE1D_label"].isin(
                list(self.coal_industries.keys())
            )

            transition_pool = lfs_data_country[occ_subset & ind_subset]

        # select all observations which occupations are browner than the
        # across-occupation average of the share_brown_slt variable
        elif scenario == "brown_techchange":
            transition_pool = lfs_data_country.loc[
                lfs_data_country["share_brown_slt"]
                > isco_grp_avg["share_brown_slt"].mean()
            ]

        # select all observations which occupations are browner than the
        # across-occupation average of the share_brown_sl variable
        elif scenario == "brown":
            transition_pool = lfs_data_country.loc[
                lfs_data_country["share_brown_sl"]
                > isco_grp_avg["share_brown_sl"].mean()
            ]
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
            ["NUTS_ID", "ISCO3D"]
        ).aggregate(
            {
                "COEFF": np.sum,
                "COEFF_share_brown_sl": np.sum,
                "COEFF_share_brown_slt": np.sum,
                "COEFF_share_green": np.sum,
                "COEFF_share_neutral_sl": np.sum,
                "COEFF_share_neutral_slt": np.sum,
            }
        )

        # beautify
        jobs_by_regions_countries = occ_number_by_nuts2_and_isco3.reset_index()
        jobs_by_regions_countries["COUNTRYW"] = jobs_by_regions_countries[
            "NUTS_ID"
        ].str.slice(0, 2)

        # create categories of aggregate target jobs for SL and SLT versions
        jobs_by_regions_countries["COEFF_share_target_sl"] = (
            jobs_by_regions_countries["COEFF_share_green"]
            + jobs_by_regions_countries["COEFF_share_neutral_sl"]
        )
        jobs_by_regions_countries["COEFF_share_target_slt"] = (
            jobs_by_regions_countries["COEFF_share_green"]
            + jobs_by_regions_countries["COEFF_share_neutral_slt"]
        )

        # ceil all numeric results
        numeric_cols = jobs_by_regions_countries.select_dtypes(
            include=[np.number]
        ).columns.values
        jobs_by_regions_countries[numeric_cols] = jobs_by_regions_countries[
            numeric_cols
        ].apply(np.ceil, axis=1)

        return jobs_by_regions_countries

    def simulate(
        self,
        level="isco_3_digit",
        countries=None,
        scenarios=None,
        reskilling=None,
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
            None, "random", "computer_literacy", "coreness_weighted",
            "coreness_percentile", "optimal"
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

        if reskilling is None:
            # read sim matrix
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
            coeff_weight = self.transition_pool_weights[scenario]

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

                # impute missing region of work with region of home & update NUTS codes
                # TODO: move to preprocessing of LFS data
                df_transition_pool.loc[
                    df_transition_pool["REGIONW"].isna(), "NUTS_ID"
                ] = df_transition_pool.loc[
                    df_transition_pool["REGIONW"].isna(), "COUNTRYW"
                ].astype(
                    str
                ) + df_transition_pool.loc[
                    df_transition_pool["REGIONW"].isna(), "REGION"
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

                # populate transition pool dict
                # todo: do i need this snippet?
                transition_pool_dict = {}
                for i, s in df_transition_pool.iterrows():
                    transition_pool_dict[s.ISCO3D_label] = s.ISCO3D

                # save numbers by source industry
                # todo: does not work for every country, overwrites results in each
                #  iteration as of now
                # df_transition_pool.groupby("NACE1D_label").sum().iloc[:, :4].to_csv(
                #     os.path.join(target_dir, "source_transition_pool.csv")
                # )

                # ---------------------------------------------------------------------
                # Worker-level transition simulation
                # ---------------------------------------------------------------------
                transition_number_data = []
                if verbose:
                    print(
                        "    transition pool: {} workers (n={})".format(
                            df_transition_pool[coeff_weight].sum(),
                            df_transition_pool.shape[0],
                        )
                    )

                for i, search_obs in df_transition_pool.iterrows():
                    search_label = search_obs.ISCO3D_label
                    idx_occ = df_occs.loc[
                        df_occs["preferredLabel"] == search_label
                    ].index.values[0]

                    # -----------------------------------------------------------------
                    # Upskilling step (optional)
                    # -----------------------------------------------------------------
                    if reskilling is not None:
                        similarity_matrix = self.reskill(
                            idx_occ,
                            search_label,
                            reskilling_mode=reskilling,
                            mask_diagonal=mask_diagonal,
                        )

                    # find closest target occupations
                    target_occs = occupation_distance.find_closest(
                        i=idx_occ, similarity_matrix=similarity_matrix, df=df_occs
                    )

                    # VIABILITY THRESHOLD
                    target_occs_filtered = target_occs.loc[
                        target_occs["similarity"] > q_viable
                    ]
                    target_occs_filtered_hv = target_occs.loc[
                        target_occs["similarity"] > q_highly_viable
                    ]

                    # target occ green or neutral
                    # todo: coal case needs adaptation (should job transitions only avoid coal occs? or broader brown occs?)
                    target_occs_filtered = target_occs_filtered.loc[
                        target_occs_filtered[category_version].isin(self.target_cats)
                    ]
                    target_occs_filtered_hv = target_occs_filtered_hv.loc[
                        target_occs_filtered_hv[category_version].isin(self.target_cats)
                    ]

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
                        if search_obs["annual_earnings"] != np.nan:
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
                    if not target.empty:
                        # Every worker transitions to target job based on switching
                        # logic. We evaluate the wage difference.
                        search_obs["earnings_delta_closest_switch"] = (
                            target["annual_earnings"].values[0]
                            - search_obs["annual_earnings"]
                        )
                        search_obs["earnings_delta_closest_switch_sum"] = (
                            target["annual_earnings"].values[0]
                            - search_obs["annual_earnings"]
                        ) * search_obs[coeff_weight]
                        search_obs["n_viable_transitions"] = target_occs_filtered.shape[
                            0
                        ]
                        search_obs["n_viable_transitions_sum"] = (
                            target_occs_filtered.shape[0] * search_obs[coeff_weight]
                        )
                        search_obs["transition_viable"] = True
                        search_obs["transition_target"] = target[
                            "preferredLabel"
                        ].values[0]
                        search_obs["target_category"] = target[category_version].values[
                            0
                        ]
                    else:
                        # if no viable transition exists, I assume (FOR NOW), that the
                        # salary is lost entirely and has to be provided by the state
                        # (a bit unrealistic)
                        search_obs["earnings_delta_closest_switch"] = search_obs[
                            "annual_earnings"
                        ] * (-1)
                        search_obs["earnings_delta_closest_switch_sum"] = (
                            search_obs["annual_earnings"]
                            * (-1)
                            * search_obs[coeff_weight]
                        )
                        search_obs["n_viable_transitions"] = 0
                        search_obs["n_viable_transitions_sum"] = 0
                        search_obs["transition_viable"] = False
                        search_obs["transition_target"] = None
                        search_obs["target_category"] = None

                    if not target_hv.empty:
                        # Every worker transitions to target job based on switching
                        # logic. We evaluate the wage difference.
                        search_obs["earnings_delta_closest_switch_hv"] = (
                            target_hv["annual_earnings"].values[0]
                            - search_obs["annual_earnings"]
                        )
                        search_obs["earnings_delta_closest_switch_sum_hv"] = (
                            target_hv["annual_earnings"].values[0]
                            - search_obs["annual_earnings"]
                        ) * search_obs[coeff_weight]
                        search_obs["n_hv_transitions"] = target_occs_filtered_hv.shape[
                            0
                        ]
                        search_obs["n_hv_transitions_sum"] = (
                            target_occs_filtered_hv.shape[0] * search_obs[coeff_weight]
                        )
                        search_obs["transition_hv"] = True
                        search_obs["transition_target_hv"] = target_hv[
                            "preferredLabel"
                        ].values[0]
                        search_obs["target_category_hv"] = target_hv[
                            category_version
                        ].values[0]
                    else:
                        # if no viable transition exists, I assume (FOR NOW), that the
                        # salary is lost entirely and has to be provided by the state
                        # (a bit unrealistic)
                        search_obs["earnings_delta_closest_switch_hv"] = search_obs[
                            "annual_earnings"
                        ] * (-1)
                        search_obs["earnings_delta_closest_switch_sum_hv"] = (
                            search_obs["annual_earnings"]
                            * (-1)
                            * search_obs[coeff_weight]
                        )
                        search_obs["n_hv_transitions"] = 0
                        search_obs["n_hv_transitions_sum"] = 0
                        search_obs["transition_hv"] = False
                        search_obs["transition_target_hv"] = None
                        search_obs["target_category_hv"] = None

                    transition_number_data.append(search_obs)

                # to df
                if len(transition_number_data) > 0:
                    df_transition_numbers = pd.concat(transition_number_data, axis=1).T
                    df_transition_numbers = df_transition_numbers.infer_objects()

                    # scale to mio
                    df_transition_numbers["earnings_delta_closest_switch_sum_mio"] = (
                        df_transition_numbers["earnings_delta_closest_switch_sum"]
                        / 10**6
                    )
                    df_transition_numbers[
                        "earnings_delta_closest_switch_sum_hv_mio"
                    ] = (
                        df_transition_numbers["earnings_delta_closest_switch_sum_hv"]
                        / 10**6
                    )

                    # store country results
                    scenario_results[country] = df_transition_numbers
                else:
                    continue

            # store scenario-country results
            simulation_results[scenario] = scenario_results

        # save as pickle
        with open(os.path.join(target_dir, fname), "wb") as handle:
            pickle.dump(simulation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return simulation_results

    def reskill(
        self,
        idx_occ,
        occ_skills_mat,
        search_label=None,
        reskilling_mode="optimal",
        skill_rank=1,
        q_coreness=99.9,
        mask_diagonal=True,
    ):
        """

        Parameters
        ----------
        idx_occ : int
            Index of search occupation.
        search_label : str
            Label of search occupation.
            TODO: seems a bit redundant given we already pass idx_occ. See if it can be
             skipped.
        reskilling_mode : str
            Mode of reskilling. One of:
                random: workers acquire skills at random, without respecting the
                    complementarity with their existing skillset. Serves as a null-model.
                computer_literacy: workers acquire the skill "have computer literacy"
                    this knowledge item has a relatively high coreness of 0.45.
                    Description: "Utilise computers, IT equipment and modern day
                    technology in an efficient way."
                coreness_weighted: workers randomly acquire a skill, although with
                    probabilities weighted by a skill's coreness
                coreness_percentile: workers randomly acquire a skill out of those skills
                    within the percentile defined by q_coreness.
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

        #   upskill optimally
        if reskilling_mode == "optimal":
            upskilling_data = self.df_optimal_upskilling_per_occ.loc[
                self.df_optimal_upskilling_per_occ["ISCO3D_label"] == search_label
            ]
            # subtract 1 from skill rank because python is 0-indexed
            col_loc = upskilling_data.columns.get_loc("idx_skill")
            idx_skill = upskilling_data.iloc[skill_rank - 1, col_loc]

        # example: reskill everybody with computer skills.
        # note: can't simulate a reskilling journey with length > 1
        elif reskilling_mode == "computer_literacy":
            skill_label = "have computer literacy"
            idx_skill = self.df_coreness.loc[
                self.df_coreness["preferredLabel"] == skill_label
            ].index.values[0]

        # draw random skill out of sample of skills higher than a given percentile
        # across the coreness distribution
        elif reskilling_mode == "coreness_percentile":
            thresh = np.percentile(self.df_coreness["coreness"], q=q_coreness)
            reskilling_options = self.df_coreness.loc[
                self.df_coreness["coreness"] > thresh
            ]
            idx_skill = reskilling_options.sample(n=1).index.values[0]

        # draw random skill with probabilities weighted by coreness values
        elif reskilling_mode == "coreness_weighted":
            idx_skill = self.df_coreness.sample(n=1, weights="coreness").index.values[0]

        # Null model: randomly draw skill for retraining
        elif reskilling_mode == "random":
            idx_skill = np.random.randint(0, self.n_skills)
        else:
            raise NotImplementedError

        # Steps dependent of reskilling mode:
        # 3) update occ-skills matrix with worker's newly acquired skill
        #    Note: this adds the skill as an essential skill (value of 1).
        occ_skills_mat_3d_updated = occ_skills_mat.copy()
        occ_skills_mat_3d_updated.iloc[idx_occ, idx_skill] = 1

        # 4) re-calculate the occupation similarity matrix (via dot product)
        occ_sim_mat_3d_updated = np.dot(
            occ_skills_mat_3d_updated.values,
            occ_skills_mat_3d_updated.values.transpose(),
        )

        # 5) fill the diagonal with zeros to avoid self-transitions
        if mask_diagonal:
            np.fill_diagonal(occ_sim_mat_3d_updated, 0)

        return occ_sim_mat_3d_updated, occ_skills_mat_3d_updated

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
        target_job_availability_coeff="COEFF_mean+sd",
        out_dir=os.path.join(useful_paths.figure_dir, "reskilling_simulation"),
    ):
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
            None, "random", "computer_literacy", "coreness_weighted",
            "coreness_percentile", "optimal"
            ]. See documentation of self.reskill.
        target_job_availability_coeff: str
            Availability of jobs in country-region-occupation. Based on long-term
            (1998-2019) positive employment fluctuations.
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

        # create output dir and fnames
        # todo: rename?
        reg_constraint_str = "regC" if region_constraints else "no-regC"

        # format: "{sim_version}_{opt_target}-opt_{reg_constraint}_{year}"
        dirname = self.dirname_out_reg.format(
            sim_version=self.simulation_name[reskilling],
            opt_target=transition_optimisation,
            reg_constraint=reg_constraint_str,
            year=self.year,
        )
        fname = "{}.pkl".format(dirname)
        target_dir = os.path.join(out_dir, dirname)
        utils.ccdir(target_dir)

        # read job availability per country, region and isco group
        # todo: read into class variable
        jobs_by_regions_countries = pd.read_pickle(
            os.path.join(
                "C:",
                os.sep,
                "eurostat_data",
                "processed",
                "lfs_employment_fluctuations_abs_1998_2019.pkl",
            )
        )

        # ceil job availability
        numeric_cols = jobs_by_regions_countries.select_dtypes(
            include=[np.number]
        ).columns.values

        jobs_by_regions_countries[numeric_cols] = np.ceil(
            jobs_by_regions_countries[numeric_cols]
        )

        # -----------------------------------------------------------------------------
        # Scenario loop (#1)
        # -----------------------------------------------------------------------------
        simulation_results = {}
        for scenario in scenarios:
            print("scenario: {}".format(scenario))
            # select scenario-specific weighting coefficient
            coeff_weight = self.transition_pool_weights[scenario]

            # scenario-specific categories of occupations
            category_version = self.category_versions[scenario]

            # -------------------------------------------------------------------------
            # Country loop (#2)
            # -------------------------------------------------------------------------
            scenario_results = {}
            for country in countries:
                if verbose:
                    print("  country: {}".format(country))

                # select scenario- and country-specific transition pool
                df_transition_pool = self.define_transition_pool(
                    scenario=scenario, country=country
                )

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

                # jobs by region
                jobs_by_regions = jobs_by_regions_countries[
                    jobs_by_regions_countries["COUNTRYW"] == country
                ]

                # -------------------------------------------------------------------------
                # Regions loop (#3)
                # -------------------------------------------------------------------------
                results_by_region = {}
                jobs_by_region_updated = {}
                for nuts_code in tqdm(nuts_codes):
                    # print(nuts_code)

                    # these nuts codes don't exist, no idea why they show up in data
                    # todo: move elsewhere
                    if nuts_code in [
                        "DE10",
                        "DE20",
                        "DE70",
                        "DE90",
                        "DEB0",
                        "DED0",
                        "DEA0",
                    ]:
                        continue

                    # absolute n of jobs in region per occupation category
                    # bookkeeping of regional constraints in absorptive employment
                    # capacity has to happen here (!)
                    jobs_by_region = jobs_by_regions[
                        jobs_by_regions["NUTS_ID"] == nuts_code
                    ]

                    # regional subset of transition pool
                    regional_transition_pool = df_transition_pool.loc[
                        df_transition_pool["NUTS_ID"] == nuts_code
                    ]

                    # regional subset of source occupations
                    # shuffle order to randomise the next loop (randomisation #1)
                    src_occ_groups = list(regional_transition_pool.ISCO3D.unique())
                    np.random.shuffle(src_occ_groups)

                    # -----------------------------------------------------------------
                    # Occupation loop (#4): source occupation categories (ISCO 3-digit)
                    # -----------------------------------------------------------------
                    transition_number_data = []
                    for src_occ_group in src_occ_groups:

                        # pool of transitioning workers in region i and occupation j
                        src_workers = regional_transition_pool.loc[
                            regional_transition_pool.ISCO3D == src_occ_group
                        ]

                        # ceil number of workers searching new job
                        src_workers[coeff_weight] = np.ceil(src_workers[coeff_weight])

                        # absolute number of transitioning workers in
                        # region i and occupation j
                        n_workers_transitioning = src_workers[coeff_weight].sum()

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
                        #   - I can now simulate reskilling journeys (-:
                        # -------------------------------------------------------------

                        # init baseline occupation-skills matrix (needs to be outside
                        #  reskilling journey loop)
                        if reskilling is not None:
                            occ_skills_mat = self.occ_skills_mat_3d.copy()

                        # loop over reskilling journey (for baseline, it has length 1)
                        # note: step = 0 represents the baseline
                        for step in np.arange(
                            start=0, stop=reskilling_journey_length + 2, step=1
                        ):
                            # print("reskilling journey step: {}".format(step))

                            if reskilling is not None and step > 0:
                                # iteratively update the occupation-skills matrix
                                #  while storing all previous changes made within
                                #  a given reskilling scenario
                                similarity_matrix, occ_skills_mat = self.reskill(
                                    occ_skills_mat=occ_skills_mat,
                                    idx_occ=idx,
                                    search_label=search_label,
                                    reskilling_mode=reskilling,
                                    skill_rank=step-1,
                                    mask_diagonal=mask_diagonal,
                                )

                            # find closest target occupations
                            target_occs = occupation_distance.find_closest(
                                i=idx, similarity_matrix=similarity_matrix, df=df_occs
                            )

                            # filter criteria 1: sim > viability threshold
                            target_occs_filtered = target_occs.loc[
                                target_occs["similarity"] > q_viable
                            ]

                            # filter criteria 2: target occupation is neutral or green
                            # KEY choice: what criteria to use for coal case?
                            # i think it makes sense to only restrict coal occupations
                            # todo: would be good to move this part into the class init
                            filtering_criteria_by_scenario = {
                                # "coal": (
                                #     ~target_occs_filtered["code"].isin(
                                #         list(self.coal_occupations.values())
                                #     )
                                # ),
                                "coal": (
                                    target_occs_filtered["category_sl"].isin(
                                        self.target_cats
                                    )
                                ),
                                "brown": (
                                    target_occs_filtered["category_sl"].isin(
                                        self.target_cats
                                    )
                                ),
                                "brown_techchange": (
                                    target_occs_filtered["category_slt"].isin(
                                        self.target_cats
                                    )
                                ),
                            }

                            # filter out unvalid target occs
                            target_occs_filtered = target_occs_filtered.loc[
                                filtering_criteria_by_scenario[scenario]
                            ]

                            # combine with (updated) job availability at regional level
                            # note: regional job availability is evaluated separately
                            #  for each iteration, independent of reskilling journey
                            #  step and length.
                            target_occs_filtered = target_occs_filtered.merge(
                                jobs_by_region,
                                left_on="code",
                                right_on="ISCO3D",
                                how="left",
                            )

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
                            src_workers = src_workers.sample(frac=1)

                            # check if viable transitions exists
                            if not target_occs_filtered.empty:
                                if region_constraints:
                                    # -----------------------------------------------------
                                    # Worker loop (#5): Individual observations from survey
                                    # -----------------------------------------------------
                                    for i, src_worker in src_workers.iterrows():
                                        # go through potential transition options ranked
                                        # by wage
                                        rank = 1
                                        while rank <= n_targets:

                                            # filter criteria 3: workers choose the viable
                                            # target occupation that provides the best wage
                                            target = target_occs_filtered.iloc[
                                                rank - 1, :
                                            ]

                                            # determine target job employment based on coeff
                                            # print(country, nuts_code)
                                            # print(jobs_by_region)
                                            # print(target)
                                            # print(jobs_by_region.loc[
                                            #           jobs_by_region["ISCO3D"] == target[
                                            #               "code"]])

                                            try:
                                                target_job_availability = (
                                                    jobs_by_region.loc[
                                                        jobs_by_region["ISCO3D"]
                                                        == target["code"],
                                                        target_job_availability_coeff,
                                                    ].values[0]
                                                )
                                            except IndexError as e:
                                                print("No target jobs available.")
                                                continue

                                            # if the target occupation can regionally absorb all
                                            # transitioning workers related to the survey
                                            # observation, every worker chooses this transition
                                            if (
                                                target_job_availability
                                                > src_worker[coeff_weight]
                                            ):

                                                # target occupation
                                                src_worker[
                                                    "transition_viable_step_{}".format(
                                                        step
                                                    )
                                                ] = True
                                                src_worker[
                                                    "transition_target_step_{}".format(
                                                        step
                                                    )
                                                ] = target["preferredLabel"]
                                                src_worker[
                                                    "transition_target_code_step_{}".format(
                                                        step
                                                    )
                                                ] = target["code"]
                                                src_worker[
                                                    "transition_target_rank_step_{}".format(
                                                        step
                                                    )
                                                ] = rank
                                                src_worker[
                                                    "target_category_step_{}".format(
                                                        step
                                                    )
                                                ] = target[
                                                    self.category_versions[scenario]
                                                ]

                                                # stats on # of viable transitions
                                                src_worker[
                                                    "n_viable_transitions_step_{}".format(
                                                        step
                                                    )
                                                ] = target_occs_filtered.shape[0]
                                                src_worker[
                                                    "n_viable_transitions_sum_step_{}".format(
                                                        step
                                                    )
                                                ] = (
                                                    src_worker[
                                                        "n_viable_transitions_step_{}".format(
                                                            step
                                                        )
                                                    ]
                                                    * src_worker[coeff_weight]
                                                )

                                                # wage difference
                                                src_worker[
                                                    "earnings_delta_closest_switch_step_{}".format(
                                                        step
                                                    )
                                                ] = (
                                                    target["annual_earnings"]
                                                    - src_worker["annual_earnings"]
                                                )
                                                src_worker[
                                                    "earnings_delta_closest_switch_sum_step_{}".format(
                                                        step
                                                    )
                                                ] = (
                                                    src_worker[
                                                        "earnings_delta_closest_switch_step_{}".format(
                                                            step
                                                        )
                                                    ]
                                                    * src_worker[coeff_weight]
                                                )

                                                # scale aggregate wage changes to millions
                                                src_worker[
                                                    "earnings_delta_closest_switch_sum_mio_step_{}".format(
                                                        step
                                                    )
                                                ] = (
                                                    src_worker[
                                                        "earnings_delta_closest_switch_sum_step_{}".format(
                                                            step
                                                        )
                                                    ]
                                                    / 10**6
                                                )

                                                # update post-transition job availability
                                                mask = (
                                                    jobs_by_region["ISCO3D"]
                                                    == target["code"]
                                                )
                                                jobs_by_region.loc[
                                                    mask, target_job_availability_coeff
                                                ] -= src_worker[coeff_weight]

                                                # update results
                                                transition_number_data.append(
                                                    src_worker
                                                )

                                                # update job availability after successful
                                                # transition
                                                jobs_by_region_updated[
                                                    nuts_code
                                                ] = jobs_by_region

                                                # jump out of loop if transition successful
                                                break
                                            else:
                                                # jump to target occupation with next-best
                                                # rank
                                                rank += 1

                                                # if no transition option is available due
                                                # to unsufficient job availability
                                                if rank > n_targets:
                                                    # save results
                                                    src_worker[
                                                        "transition_viable_step_{}".format(
                                                            step
                                                        )
                                                    ] = False
                                                    src_worker[
                                                        "transition_target_rank_step_{}".format(
                                                            step
                                                        )
                                                    ] = rank

                                                    # lost wage
                                                    src_worker[
                                                        "earnings_delta_closest_switch_step_{}".format(
                                                            step
                                                        )
                                                    ] = src_worker[
                                                        "annual_earnings"
                                                    ] * (
                                                        -1
                                                    )
                                                    src_worker[
                                                        "earnings_delta_closest_switch_sum_step_{}".format(
                                                            step
                                                        )
                                                    ] = (
                                                        src_worker[
                                                            "earnings_delta_closest_switch_step_{}".format(
                                                                step
                                                            )
                                                        ]
                                                        * src_worker[coeff_weight]
                                                    )

                                                    # scale aggregate wage changes to millions
                                                    src_worker[
                                                        "earnings_delta_closest_switch_sum_mio_step_{}".format(
                                                            step
                                                        )
                                                    ] = (
                                                        src_worker[
                                                            "earnings_delta_closest_switch_sum_step_{}".format(
                                                                step
                                                            )
                                                        ]
                                                        / 10**6
                                                    )

                                                    transition_number_data.append(
                                                        src_worker
                                                    )
                                                continue
                                else:
                                    # without regional constraints all workers with the same
                                    # isco code can be treated together
                                    target = target_occs_filtered.head(1)

                                    src_workers[
                                        "transition_viable_step_{}".format(step)
                                    ] = True
                                    src_workers[
                                        "transition_target_step_{}".format(step)
                                    ] = target["preferredLabel"]
                                    src_workers[
                                        "transition_target_code_step_{}".format(step)
                                    ] = target["code"]
                                    src_workers[
                                        "transition_target_rank_step_{}".format(step)
                                    ] = np.nan
                                    src_workers[
                                        "target_category_step_{}".format(step)
                                    ] = target[self.category_versions[scenario]]

                                    # stats on # of viable transitions
                                    src_workers[
                                        "n_viable_transitions_step_{}".format(step)
                                    ] = target_occs_filtered.shape[0]

                                    src_workers[
                                        "n_viable_transitions_sum_step_{}".format(step)
                                    ] = (
                                        src_workers[
                                            "n_viable_transitions_step_{}".format(step)
                                        ]
                                        * src_workers[coeff_weight]
                                    )

                                    # wage difference
                                    src_workers[
                                        "earnings_delta_closest_switch_step_{}".format(
                                            step
                                        )
                                    ] = (
                                        target["annual_earnings"].values[0]
                                        - src_workers["annual_earnings"]
                                    )
                                    src_workers[
                                        "earnings_delta_closest_switch_sum_step_{}".format(
                                            step
                                        )
                                    ] = (
                                        src_workers[
                                            "earnings_delta_closest_switch_step_{}".format(
                                                step
                                            )
                                        ]
                                        * src_workers[coeff_weight]
                                    )

                                    # scale aggregate wage changes to millions
                                    src_workers[
                                        "earnings_delta_closest_switch_sum_mio_step_{}".format(
                                            step
                                        )
                                    ] = (
                                        src_workers[
                                            "earnings_delta_closest_switch_sum_step_{}".format(
                                                step
                                            )
                                        ]
                                        / 10**6
                                    )

                                    # add one by one (workaround)
                                    for i, src_worker in src_workers.iterrows():
                                        transition_number_data.append(src_worker)

                            else:
                                # no need to loop over individual workers if no viable
                                # transition exists, I can treat them in bulk.
                                # if no viable transition exists, I assume that the salary
                                # is lost entirely and has to be provided by the state
                                # (how realistic?)
                                src_workers[
                                    "earnings_delta_closest_switch_step_{}".format(step)
                                ] = src_workers["annual_earnings"] * (-1)
                                src_workers[
                                    "earnings_delta_closest_switch_sum_step_{}".format(
                                        step
                                    )
                                ] = (
                                    src_workers[
                                        "earnings_delta_closest_switch_step_{}".format(
                                            step
                                        )
                                    ]
                                    * src_workers[coeff_weight]
                                )
                                src_workers[
                                    "n_viable_transitions_step_{}".format(step)
                                ] = 0
                                src_workers[
                                    "n_viable_transitions_sum_step_{}".format(step)
                                ] = 0
                                src_workers[
                                    "transition_viable_step_{}".format(step)
                                ] = False
                                src_workers[
                                    "transition_target_step_{}".format(step)
                                ] = None
                                src_workers[
                                    "transition_target_code_step_{}".format(step)
                                ] = None
                                src_workers[
                                    "target_category_step_{}".format(step)
                                ] = None

                                # scale aggregate wage changes to millions
                                src_workers[
                                    "earnings_delta_closest_switch_sum_mio_step_{}".format(
                                        step
                                    )
                                ] = (
                                    src_workers[
                                        "earnings_delta_closest_switch_sum_step_{}".format(
                                            step
                                        )
                                    ]
                                    / 10**6
                                )

                                # add one by one (workaround)
                                for i, src_worker in src_workers.iterrows():
                                    transition_number_data.append(src_worker)

                    # note: end of loop over reskilling journey steps
                    results_by_region[nuts_code] = transition_number_data

                # combine post-transition job availability to df
                # note: temporarily commented out, need to fix bug raised by AT
                # print(jobs_by_region_updated)
                # df_jobs_by_region_post = pd.concat(
                #     list(jobs_by_region_updated.values()), axis=0
                # )

                # combine transition results to df
                nested_list = list(results_by_region.values())
                flat_list = [item for sublist in nested_list for item in sublist]
                df_transition_numbers = pd.concat(flat_list, axis=1).T

                # obj to numeric
                # cast obj to float
                df_transition_numbers = df_transition_numbers.infer_objects()
                df_transition_numbers["AGE"] = pd.to_numeric(
                    df_transition_numbers["AGE"])

                # subset results
                mean_group = df_transition_numbers.columns.str.startswith(
                    ("n_viable_transitions_step", "transition_viable", "AGE")
                )
                sum_group = df_transition_numbers.columns.str.startswith(
                    ("earnings_delta_closest_switch", "COEFF", "NOBS")
                )

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
                    else:
                        continue

                # aggregate
                df_transition_numbers_by_nuts_nace = (
                    df_transition_numbers.groupby(["COUNTRYW", "NUTS_ID", "NACE1D"])
                        .aggregate(agg_funcs)
                )

                # append
                scenario_results[country] = df_transition_numbers_by_nuts_nace

            simulation_results[scenario] = scenario_results

        # save as pickle
        with open(os.path.join(target_dir, fname), "wb") as handle:
            pickle.dump(simulation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return simulation_results

    def visualise_simulation_results(
        self,
        simulation_results=None,
        base_dir=os.path.join(useful_paths.figure_dir, "reskilling_simulation"),
        year=2019,
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
                coeff_weight = self.transition_pool_weights[scenario]

                # calc stats
                n_obs = df_transition_numbers.shape[0]
                n_workers = df_transition_numbers[coeff_weight].sum().astype(int)
                # ---------------------------------------------------------------------
                # REGIONAL AGGREGATION
                # ---------------------------------------------------------------------

                # Calculate avg number of transitions per threatened job
                df_transition_numbers_by_nuts = df_transition_numbers.groupby(
                    "NUTS_ID"
                ).sum()

                df_transition_numbers_by_nuts["n_viable_transitions_rel"] = (
                    df_transition_numbers_by_nuts.n_viable_transitions_sum
                    / df_transition_numbers_by_nuts[coeff_weight]
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

                cmap_transitions = plotting_utils.discrete_cmap_with_manual_colors(
                    cmap_type="Blues",
                    n_classes=10,
                    colour_replacements={0: "lightcoral"},
                )
                cmap_transitions.set_over("darkblue")

                # upper limits for colorbars
                vmax_transitions = 10  # n viable transitions
                vmax_wages = 40  # million euro

                # transition numbers
                gdf_transition_numbers_by_nuts.plot(
                    column="n_viable_transitions_rel",
                    legend=True,
                    cmap=cmap_transitions,
                    vmin=0,
                    vmax=vmax_transitions,
                    legend_kwds={
                        "label": "Viable transitions per at-risk worker [-]",
                        "fraction": 0.03,
                        "extend": "max",
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
                        "label": "$\Delta$ Annual earnings [M€ (2019)]",
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
                    "$Total = {:.2f}~M€~(2019)$".format(
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
                vars = ["earnings_delta_closest_switch_sum_mio", "n_viable_transitions"]
                var_labels = [
                    "$\Delta$ Annual earnings [M€ (2019)]",
                    "Viable transitions per at-risk worker [-]",
                ]
                var_fname = ["earnings", "transitions"]

                for i, var in enumerate(vars):
                    fig, (ax1, ax2) = plt.subplots(
                        ncols=2,
                        figsize=(10, 5),
                        sharey=True,
                        sharex=False,
                        gridspec_kw={"width_ratios": [0.7, 0.3]},
                    )

                    y_order = (
                        df_transition_numbers.groupby("NACE1D_label")
                        .median()[var]
                        .sort_values(ascending=False)
                        .index.values
                    )

                    # left
                    if var_fname[i] == "earnings":
                        sns.boxplot(
                            data=df_transition_numbers,
                            x=var,
                            y="NACE1D_label",
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
                        sns.barplot(
                            data=df_transition_numbers,
                            x=var,
                            y="NACE1D_label",
                            orient="h",
                            estimator=np.sum,
                            ci=None,
                            order=y_order,
                            palette="RdYlGn_r",
                            ax=ax2,
                        )
                        ax2.bar_label(
                            ax2.containers[-1], fmt="%.0f", label_type="center"
                        )
                    elif var_fname[i] == "transitions":
                        sns.barplot(
                            data=df_transition_numbers,
                            x=var,
                            y="NACE1D_label",
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
        year=2019,
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

        # params
        xmin, xmax, ymin, ymax = bbox_eu_epsg_3035

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

            # concatenate data across countries
            df_transition_numbers = pd.concat(list(country_results_dict.values()))

            # optionally reset multi-index
            if isinstance(df_transition_numbers.index, pd.MultiIndex):
                df_transition_numbers = df_transition_numbers.reset_index()

            print(df_transition_numbers.columns)
            
            # cast obj to float
            df_transition_numbers = df_transition_numbers.infer_objects()
            df_transition_numbers["AGE"] = pd.to_numeric(df_transition_numbers["AGE"])

            # select scenario-specific weighting coefficient
            coeff_weight = self.transition_pool_weights[scenario]

            # calc stats
            n_obs = df_transition_numbers.shape[0]
            n_workers = df_transition_numbers[coeff_weight].sum()  # .astype(int)

            ind_label = "NACE1D_label"
            if not ind_label in df_transition_numbers.columns.values:
                df_transition_numbers = pd.merge(
                    df_transition_numbers,
                    self.nace_labels,
                    on="NACE1D",
                    how="left",
                    suffixes=("", "_drop"),
                )

            # ---------------------------------------------------------------------
            # REGIONAL AGGREGATION
            # ---------------------------------------------------------------------

            # Calculate avg number of transitions per threatened job, sum of earnings
            #  changes, etc.
            # print(df_transition_numbers.columns)
            # todo: dissect transition category and ranking if we need the info
            mean_group = df_transition_numbers.columns.str.startswith(
                ("n_viable_transitions_step", "transition_viable", "AGE")
            )
            sum_group = df_transition_numbers.columns.str.startswith(
                ("earnings_delta_closest_switch", "COEFF", "NOBS")
            )
            all = np.repeat(True, df_transition_numbers.shape[1])

            # find rest group: agg = first
            rest_group = np.logical_and(all, mean_group)
            rest_group = np.logical_and(rest_group, sum_group)
            rest_group = np.invert(rest_group)

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

            # df_transition_numbers_by_nuts["n_viable_transitions_rel_step_{}".format(step)] = (
            #     df_transition_numbers_by_nuts["n_viable_transitions_sum_step_{}".format(step)]
            #     / df_transition_numbers_by_nuts[coeff_weight]
            # )

            # to gdf
            gdf_transition_numbers_by_nuts = pd.merge(
                self.gdf[
                    self.gdf["CNTR_CODE"].isin(df_transition_numbers["COUNTRYW"].values)
                ],
                df_transition_numbers_by_nuts.reset_index(),
                on="NUTS_ID",
                how="left",
            )

            # ---------------------------------------------------------------------
            # REGIONAL PLOTS
            # ---------------------------------------------------------------------
            if not show_map_boxplots:
                fig, (ax1, ax2) = plt.subplots(
                    nrows=1,
                    ncols=2,
                    figsize=(20, 20),
                    gridspec_kw={"width_ratios": [0.5, 0.5]},
                )
            else:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                    nrows=2,
                    ncols=2,
                    figsize=(20, 15),
                    gridspec_kw={
                        "width_ratios": [0.5, 0.5],
                        "height_ratios": [0.9, 0.1],
                        "hspace": 0,
                    },
                )

            cmap_transitions = plotting_utils.discrete_cmap_with_manual_colors(
                cmap_type="Blues",
                n_classes=vmax_transitions,
                colour_replacements={0: "lightcoral"},
            )
            cmap_transitions.set_over("black")

            # transition numbers
            gdf_transition_numbers_by_nuts.plot(
                column="n_viable_transitions_step_{}".format(step),
                legend=True,
                cmap=cmap_transitions,
                vmin=0,
                vmax=vmax_transitions,
                legend_kwds={
                    "label": "Viable transitions per at-risk worker [-]",
                    "fraction": cbar_fraction,
                    "extend": "max",
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

            # mask countries with missing earnings data
            cntr_missing = ["AT", "CZ", "ES", "GR", "IS", "NO", "SE", "UK"]
            gdf_transition_numbers_by_nuts.loc[
                gdf_transition_numbers_by_nuts["CNTR_CODE"].isin(cntr_missing),
                "earnings_delta_closest_switch_step_{}".format(step),
            ] = np.nan

            # TODO: emergency normalise the earnings data to per-worker levels
            # per worker
            earnings_cols_rel = gdf_transition_numbers_by_nuts.columns[gdf_transition_numbers_by_nuts.columns.str.startswith(
                "earnings_delta_closest_switch_step")].values

            # Re-mask income data for countries that don’t report data (!)
            gdf_transition_numbers_by_nuts.loc[gdf_transition_numbers_by_nuts["CNTR_CODE"].isin(
                cntr_missing), earnings_cols_rel] = np.nan

            # TODO: THIS IS ONLY VALID IF RUNNING THE CODE FOR THE FFPOTC JOB POOL ALONE (!!!!!!!)
            # NOTE: can back-calculate from region-sector pairs because I also sum up all COEFF's in the reskilling simulations
            norm_factor = "COEFF_share_brown_slt"
            for col in earnings_cols_rel:
                gdf_transition_numbers_by_nuts[col] = gdf_transition_numbers_by_nuts[col] / gdf_transition_numbers_by_nuts[norm_factor]

            # cmap
            if dynamic_wages_cmap:
                vmax_wages = (
                    gdf_transition_numbers_by_nuts[
                        "earnings_delta_closest_switch_step_{}".format(step)
                    ]
                    .abs()
                    .quantile(q=0.98)
                )
            cmap_earnings = plt.get_cmap("coolwarm_r", (vmax_wages / 1000) * 4)
            cmap_earnings.set_over("darkblue")
            cmap_earnings.set_under("darkred")

            # ax2: earnings losses
            gdf_transition_numbers_by_nuts.plot(
                column="earnings_delta_closest_switch_step_{}".format(step),
                legend=True,
                cmap=cmap_earnings,
                vmin=-vmax_wages,
                vmax=vmax_wages,
                legend_kwds={
                    "label": "$\Delta$ Annual earnings (€)",
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

            ax2.set_title(
                "$Total = {:.2f}~M€~(2019)$".format(
                    gdf_transition_numbers_by_nuts[
                        "earnings_delta_closest_switch_step_{}".format(step)
                    ].sum()
                )
            )

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
                    x=gdf_transition_numbers_by_nuts[
                        "n_viable_transitions_step_{}".format(step)
                    ],
                    ax=ax3,
                )

                # ax4: distribution across regions (earnings)
                sns.boxplot(
                    x=gdf_transition_numbers_by_nuts[
                        "earnings_delta_closest_switch_step_{}".format(step)
                    ],
                    ax=ax4,
                )

            # layout
            if show_title:
                fig.suptitle(
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
                    ),
                    fontsize=title_fontsize,
                )
            fig.tight_layout()
            fig.subplots_adjust(top=1.4)

            fname = "{}_{}_{}_{}_step_{}.{}".format(
                "EU", year, "regional", scenario, step, "png"
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

            # save numbers
            # todo: adjust to jointly save the numbers from all reskilling steps
            vars = [
                "earnings_delta_closest_switch_step_{}".format(step),
                "n_viable_transitions_step_{}".format(step),
            ]
            if save_tables:
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
            vars = [
                "earnings_delta_closest_switch_step_{}".format(step),
                "n_viable_transitions_step_{}".format(step),
            ]
            var_labels = [
                "$\Delta$ Annual earnings [M€ (2019)]",
                "Transitions per at-risk worker",
            ]
            var_fname = ["earnings", "transitions"]
            df_transition_numbers = df_transition_numbers.replace(
                to_replace={"NACE1D_label": self.nace_mapping}
            )

            # subset relevant industries for paper
            if industry_subset_paper:
                industry_subset = self.nace_labels.loc[
                    self.nace_labels["paper_selection"] == True, "NACE1D_label_short"
                ]
                df_transition_numbers = df_transition_numbers.loc[
                    df_transition_numbers["NACE1D_label"].isin(industry_subset)
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

                    y_order = (
                        df_transition_numbers.groupby("NACE1D_label")
                        .aggregate({var: "median"})
                        .sort_values(by=var, ascending=False)
                        .index.values
                    )

                    # left
                    if var_fname[i] == "earnings":
                        sns.boxplot(
                            data=df_transition_numbers,
                            x=var,
                            y="NACE1D_label",
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

                        # right
                        sns.barplot(
                            data=df_transition_numbers,
                            x=var,
                            y="NACE1D_label",
                            orient="h",
                            estimator=np.sum,
                            ci=None,
                            order=y_order,
                            palette="RdYlGn_r",
                            ax=ax2,
                        )
                        ax2.bar_label(
                            ax2.containers[-1], fmt="%.0f", label_type="center"
                        )
                    elif var_fname[i] == "transitions":
                        sns.barplot(
                            data=df_transition_numbers,
                            x=var,
                            y="NACE1D_label",
                            orient="h",
                            estimator=np.mean,
                            ci="sd",
                            order=y_order,
                            palette="RdYlGn_r",
                            ax=ax1,
                        )
                        ax1.axvline(1, linestyle="-", color="lightcoral", zorder=0)
                        ax1.set_xlim(0)

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
                    fname = "{}_{}_{}_{}_{}_step_{}.png".format(
                        "EU", year, "sectoral", fname_snippet, scenario, step
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

                y_order = (
                    df_transition_numbers.groupby("NACE1D_label")
                    .aggregate({var: "mean"})
                    .sort_values(by=var, ascending=False)
                    .index.values
                )

                # left
                sns.barplot(
                    data=df_transition_numbers,
                    x="n_viable_transitions_step_{}".format(step),
                    y="NACE1D_label",
                    orient="h",
                    estimator=np.mean,
                    ci="sd",
                    order=y_order,
                    palette="RdYlGn_r",
                    ax=ax1,
                )
                ax1.bar_label(
                    ax1.containers[-1], fmt="%.1f", label_type="center", color="white"
                )
                ax1.axvline(1, linestyle="-", color="lightcoral", zorder=0)
                ax1.set_xlim(0)
                ax1.set_xlabel("Transitions per at-risk worker [-]")

                # right
                sns.barplot(
                    data=df_transition_numbers,
                    x="earnings_delta_closest_switch_sum_mio_step_{}".format(step),
                    y="NACE1D_label",
                    orient="h",
                    estimator=np.sum,
                    ci=None,
                    order=y_order,
                    palette="RdYlGn_r",
                    ax=ax2,
                )
                ax2.bar_label(ax2.containers[-1], fmt="%.0f", label_type="center")
                ax2.set_xlabel("$\Delta$ Annual earnings [M€ (2019)]")
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
                fname = "{}_{}_{}_{}_{}_step_{}.png".format(
                    "EU", year, "sectoral", fname_snippet, scenario, step
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

                # save numbers
                if save_tables:
                    df_transition_numbers[vars] = df_transition_numbers[vars].astype(
                        float
                    )
                    res = df_transition_numbers.groupby("NACE1D_label")[vars].describe()
                    res.to_csv(
                        os.path.join(
                            base_dir,
                            dirname,
                            "{}_{}_{}_{}_{}_step_{}.{}".format(
                                "EU", year, "sectoral", fname_snippet, scenario, step, "csv"
                            ),
                        )
                    )


if __name__ == "__main__":
    import time
    import timeit
    from src.data.lfs import EuLfs

    # re-run simulations & plot or plot only?
    rerun_simulations = False

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
            year=2019,
            input_fname_lfs="eu_lfs_merged_{year}_with_final_unweighted_shares_and_earnings_incdecil_imputed",
        )

        # todo: move to lfs preprocessing chain
        # add required categories
        cats_regionw = lfs_data["REGIONW"].cat.categories.values
        cats_region = lfs_data["REGION"].cat.categories.values

        placeholder_cat = "99"

        new_cats_regionw = list(set(cats_region) - set(cats_regionw))
        new_cats_region = list(set(cats_regionw) - set(cats_region))

        new_cats_regionw.append(placeholder_cat)
        new_cats_region.append(placeholder_cat)

        lfs_data["REGIONW"] = lfs_data["REGIONW"].cat.add_categories(new_cats_regionw)
        lfs_data["REGION"] = lfs_data["REGION"].cat.add_categories(new_cats_region)

        # fill nans in regionw with region
        lfs_data["REGIONW"] = lfs_data["REGIONW"].fillna(lfs_data["REGION"])

        # fill remaining nans with placeholder
        lfs_data["REGIONW"] = lfs_data["REGIONW"].fillna("99")
        lfs_data["REGION"] = lfs_data["REGION"].fillna("99")

        # update nuts id
        lfs_data["NUTS_ID"] = lfs_data["COUNTRYW"].astype("string") + lfs_data[
            "REGIONW"
        ].astype("string")
    else:
        lfs_data = None

    # initialise class
    rp = ReskillingPathways(
        osm_version="weighted", sim_metric="cooc", lfs_data=lfs_data, year=2019
    )

    # ---------------------------------------------------------------------
    # Simulation parameters
    # ---------------------------------------------------------------------

    # countries to analyse
    countries = [
        "AT",
        "BE",
        "CH",
        # "CY",
        "CZ",
        "DE",
        "DK",
        # "EE",
        "ES",
        "FI",
        "FR",
        "GR",
        "HR",
        "HU",
        "IE",
        "IS",
        "IT",
        "LT",
        # "LU",
        # "LV",
        # "NL",
        "NO",
        "PT",
        "RO",
        "SE",
        "SK",
        "UK",
    ]

    # transition pools to analyse
    scenarios = ["brown_techchange"]  # ["coal", "brown_techchange", "brown"]

    # reskilling options to consider
    reskilling_modes = [
        "optimal",
        "coreness_weighted",
        "random",
        # "computer_literacy"
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
    regional_constraints = [True, False]

    # length of reskilling journey
    reskilling_journey_length = 20
    steps = np.arange(0, reskilling_journey_length + 1)

    # name mapping
    processing_dict = dict(zip(transition_thresholds, shortcuts))

    # ---------------------------------------------------------------------
    # Processing loop
    # ---------------------------------------------------------------------
    start = timeit.default_timer()
    for transition_threshold, shortcut in processing_dict.items():

        # results storage
        results_dir = os.path.join(
            useful_paths.figure_dir, "reskilling_final_{}".format(shortcut)
        )

        for regional_constraint in regional_constraints:
            # run simulations for each reskilling mode
            for reskilling_mode in reskilling_modes:
                print(reskilling_mode)
                t0 = time.time()

                if rerun_simulations:
                    simulation_results = rp.simulate_regional(
                        level="isco_3_digit",
                        countries=countries,
                        scenarios=scenarios,
                        transition_optimisation=optimise,
                        reskilling=reskilling_mode,
                        reskilling_journey_length=reskilling_journey_length,
                        region_constraints=regional_constraint,
                        target_job_availability_coeff="COEFF_mean+sd",
                        mask_diagonal=True,
                        transition_thresholds=transition_threshold,
                        out_dir=results_dir,
                    )
                else:
                    simulation_results = None

                # regional & sectoral distributions of transitions and earnings losses
                for step in steps:
                    rp.visualise_simulation_results_eu(
                        simulation_results=simulation_results,
                        transition_optimisation=optimise,
                        reskilling_version=rp.simulation_name[reskilling_mode],
                        step=step,
                        regional_constraint=regional_constraint,
                        base_dir=results_dir,
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
