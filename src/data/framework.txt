import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

import src
from src import UsefulPaths, utils, stats_utils

useful_paths = src.UsefulPaths()


class Crosswalks(UsefulPaths):
    """Stores crosswalks between various occupational frameworks."""

    def __init__(self):
        # inherit class storing useful paths
        UsefulPaths.__init__(self=self)

        # ONET-ESCO crosswalks by Kanders et al. (2020)
        self._onet_esco_mcc_full = None
        self._onet_esco_mcc_reduced = None

        self._onet_soc10 = None

        self._soc10_isco08_ibs = None
        self._soc10_isco08_bls = None
        self._onet_isco08_jrc = None

        self._esco_de_kldb2010 = None
        self._esco_it_cp2011 = None

    @property
    def onet_esco_mcc_full(self):
        """
        MCC ONET-ESCO crosswalk for the full set of ESCO v.1.0.3 occupations.
        2942 unique ESCO occupations are matched to 669 unique ONET occupations.

        This crosswalk misses out on 63 of the 138 O*NET "greening" occupations. The
        reason for this is (explanation by Karlis Kanders):

        "Although we hopefully have O*NET matches for every ESCO
        occupation - not every O*NET occupation has been matched to an ESCO occupation.
        The reason is because we wanted to use O*NET modelling in the context of
        transitioning between European ESCO occupations - so it was enough to find the
        closest O*NET occupation for each ESCO occupation."

        Returns
        -------

        """
        if self._onet_esco_mcc_full is None:
            self._onet_esco_mcc_full = pd.read_csv(
                os.path.join(
                    self.data_raw,
                    "mapping-career-causeways",
                    "codebase",
                    "data",
                    "processed",
                    "ESCO_ONET_xwalk_full.csv",
                )
            )
            # create SOC 6-digit col
            self._onet_esco_mcc_full["onet_code_6d"] = self._onet_esco_mcc_full[
                "onet_code"
            ].str[:7]

        return self._onet_esco_mcc_full

    @property
    def onet_esco_mcc_reduced(self):
        """
        MCC ONET-ESCO crosswalk for the 5-digit ESCO occupations. Lower-level (6, 7, 8)
        occupations inherit the same mapping as their parent.

        1680 unique level-5 ESCO occupations are mapped to 668 unique O*NET occupations.

        Same problem as with the full crosswalk, it misses a substantial amount of
        O*NET greening occupations due to the same reason as outlined in the description
        of the onet_esco_mcc_full function.

        Returns
        -------

        """
        if self._onet_esco_mcc_reduced is None:
            self._onet_esco_mcc_reduced = pd.read_csv(
                os.path.join(
                    useful_paths.data_raw,
                    "mapping-career-causeways",
                    "supplementary_online_data",
                    "ONET_ESCO_crosswalk",
                    "esco_onet_crosswalk_Nov2020.csv",
                )
            )
        return self._onet_esco_mcc_reduced

    @property
    def onet_soc10(self):
        """
        Lookup table between 1110 8-digit O*NET-SOC and 841 6-digit SOC occupations.
        From the supplementary online data of the MCC project.

        Returns
        -------

        """
        if self._onet_soc10 is None:
            self._onet_soc10 = pd.read_excel(
                os.path.join(
                    useful_paths.data_raw,
                    "mapping-career-causeways",
                    "supplementary_online_data",
                    "ONET_ESCO_crosswalk",
                    "lookups",
                    "ONET_to_US2010SOC.xlsx",
                )
            )
        return self._onet_soc10

    @property
    def soc10_isco08_ibs(self):
        """
        Crosswalk from 839 6-digit SOC-2010 to 436 4-digit ISCO-08 occupations from the
        Institute for Structural Research - IBS.

        Note: this is actually the same crosswalk as the offical BLS SOC10-ISCO-08 one!

        Returns
        -------

        """
        if self._soc10_isco08_ibs is None:
            self._soc10_isco08_ibs = pd.read_stata(
                os.path.join(
                    useful_paths.data_raw,
                    "crosswalks",
                    "onetsoc_to_isco_cws_ibs",
                    "soc10_isco08.dta",
                )
            )
        return self._soc10_isco08_ibs

    def soc10_isco08_bls(self, direction="soc_to_isco", select_cols=True, rename=True):
        """
        Official crosswalk from 878 6-digit SOC-2010 to 441 4-digit ISCO-08 occupations
        from the US BLS. August 2012 version with a June 2015 Update, downloaded from
        the Nesta project repository.

        Returns
        -------

        """
        header = 6

        direction_dict = {
            "soc_to_isco": "2010 SOC to ISCO-08",
            "isco_to_soc": "ISCO-08 to 2010 SOC",
        }
        sheet_name = direction_dict[direction]

        rename_dict = {
            "2010 SOC Code": "soc10_code",
            "2010 SOC Title": "soc10_title",
            "ISCO-08 Code": "isco08_code",
            "ISCO-08 Title EN": "isco08_title",
        }

        col_sel = list(rename_dict.keys())

        dtypes = {"ISCO-08 Code": str}

        if self._soc10_isco08_bls is None:
            self._soc10_isco08_bls = pd.read_excel(
                os.path.join(
                    useful_paths.data_raw,
                    "crosswalks",
                    "isco_soc_crosswalk.xls",
                ),
                sheet_name=sheet_name,
                header=header,
                dtype=dtypes,
            )

            if select_cols:
                self._soc10_isco08_bls = self._soc10_isco08_bls[col_sel]

            if rename:
                self._soc10_isco08_bls = self._soc10_isco08_bls.rename(
                    columns=rename_dict
                )

        return self._soc10_isco08_bls

    @property
    def onet_isco08_jrc(self):
        raise NotImplementedError

    @property
    def esco_de_kldb2010(self):
        """
        Crosswalk ESCO 1.0.3 - DE (KldB-2010) shared by ESCO team (NOT FINISHED).

        ESCO Version: ESCO V.1.0.3 (2942 unique occupations)
        Kldb Version: 2010 (2081 unique occupations)

        Note (After dropping rows without matches):
        - 2792 unique ESCO occupations
        - 2079 unique KldB occupations (5 digits)

        Returns
        -------

        """
        if self._esco_de_kldb2010 is None:
            self._esco_de_kldb2010 = pd.read_csv(
                os.path.join(
                    useful_paths.data_raw,
                    "crosswalks",
                    "ESCO_DE_Berufenet_mapping_draft_20220324.csv",
                ),
                header=14,
            )

            # Customise column names
            self._esco_de_kldb2010.columns = self._esco_de_kldb2010.columns.str.split(
                " "
            ).str.join("_")

        return self._esco_de_kldb2010

    @property
    def esco_it_cp2011(self):
        """
        Crosswalk ESCO 1.0.8 - IT CP-2011 (FINISHED). Shared by ESCO team.
        classificazione delle professioni: http://professioni.istat.it/cp2011/.

        Note:
        - 2980 unique ESCO occupations
        - 829 unique CP-2011 occupations (5-digit)
        - a small number of occupations are not labelled in both classifications

        Returns
        -------

        """
        if self._esco_it_cp2011 is None:
            self._esco_it_cp2011 = pd.read_csv(
                os.path.join(
                    useful_paths.data_raw, "crosswalks", "ESCO_IT_mapping_23042021.csv"
                ),
                header=14,
            )

            # Customise column names
            self._esco_it_cp2011.columns = self._esco_it_cp2011.columns.str.split(
                " "
            ).str.join("_")

        return self._esco_it_cp2011

    def it_cp2011_de_kldb2010(
        self,
        save=True,
        out_dir=os.path.join(useful_paths.data_interim, "crosswalks"),
        out_fname="crosswalk_it_cp2011_de_kldb2010_reduced",
    ):
        """
        Construct bridge between IT and DE classifications via ESCO.

        Returns
        -------

        """
        cw_de_esco = self.esco_de_kldb2010
        cw_it_esco = self.esco_it_cp2011

        # drop instances without match and convert ID to int
        cw_de_esco = cw_de_esco.loc[cw_de_esco["Mapping_relation"] != "no relation"]
        cw_de_esco.loc[:, "Classification_2_ID"] = cw_de_esco.loc[
            :, "Classification_2_ID"
        ].astype(int)

        # Join German mapping on Italian mapping
        cw_it_de = pd.merge(
            left=cw_it_esco,
            right=cw_de_esco,
            on="Classification_1_URI",
            how="left",
            suffixes=("_it", "_de"),
            validate="m:m",
        )

        # Drop rows where no match in German KldB was found
        cw_it_de_reduced = cw_it_de.dropna(subset=["Classification_2_ID_de"])
        cw_it_de_reduced.loc[:, "Classification_2_ID_de"] = cw_it_de_reduced.loc[
            :, "Classification_2_ID_de"
        ].astype(int)

        # save
        if save:
            utils.save_df_to_files(
                cw_it_de_reduced,
                output_dir=out_dir,
                fname_no_ext=out_fname,
                ftypes=["csv"],
            )

        return cw_it_de_reduced


class Onet(UsefulPaths):
    """Stores O*NET-specific data on green/brown occupations and tasks."""

    def __init__(self):
        # inherit class storing useful paths
        UsefulPaths.__init__(self=self)

        self._green_tasks_gtp = None
        self._green_tasks_narrow_jrc = None

        self._green_occupations_gtp = None
        self._green_occupations_vona2018 = None
        self._green_occupations_vona2019 = None
        self._green_occupations_narrow_jrc = None
        self._green_occupations_gilli2020 = None

        self._brown_occupations_vona2018 = None

    @property
    def green_tasks_gtp(self):
        """
        Green tasks from ONET Green Task Project.
        SOC 8-digit level, 138 occupations.

        Returns
        -------

        """
        if self._green_tasks_gtp is None:
            self._green_tasks_gtp = pd.read_excel(
                io=os.path.join(
                    self.data_raw,
                    "onet",
                    "Onet_GreenTask_AppA.xlsx",
                ),
                sheet_name="Tasks",
            )
        return self._green_tasks_gtp

    @property
    def green_occupations_gtp(self):
        """
        Greenness scores from ONET Green Task Project.
        SOC 8-digit level, 138 occupations.

        Returns
        -------

        """
        if self._green_occupations_gtp is None:
            self._green_occupations_gtp = pd.read_excel(
                io=os.path.join(
                    self.data_raw,
                    "onet",
                    "Onet_GreenTask_AppA.xlsx",
                ),
                sheet_name="Occupations",
            )
        return self._green_occupations_gtp

    @property
    def green_occupations_vona2018(self):
        """
        Greenness scores from Vona et al. (2018).
        SOC 8-digit level, 111 occupations.

        Returns
        -------

        """
        if self._green_occupations_vona2018 is None:
            self._green_occupations_vona2018 = pd.read_excel(
                io=os.path.join(self.data_raw, "onet", "Vona2018_table_a1.xlsx"),
                sheet_name="Greenness",
            )
        return self._green_occupations_vona2018

    def green_occupations_vona2019(self, agg_to_6d_soc=True):
        """
        Greenness scores from Vona et al. (2019) at SOC 6-digit level.

        For rule book (average, zero, value from single occupation
        group) for aggregation from 142 unique SOC 6-to 8-digit level occupations to
        40 unique 6 digit occupations see online appendix of paper.

        Returns
        -------

        """
        if self._green_occupations_vona2019 is None:
            self._green_occupations_vona2019 = pd.read_excel(
                io=os.path.join(
                    self.data_raw, "onet", "Vona2019_Greenness_6D_SOC.xlsx"
                ),
            )

            if agg_to_6d_soc:
                grouping_var = "soc10_6d_code"
                # note: all 8D occupations within a 6D group have the same score,
                # so either "first" or np.mean works well
                agg_dict = {"share_green_vona2019_6d": np.mean}
                self._green_occupations_vona2019 = (
                    self._green_occupations_vona2019.groupby(grouping_var)
                    .aggregate(agg_dict)
                    .reset_index()
                )

                # rename
                self._green_occupations_vona2019 = (
                    self._green_occupations_vona2019.rename(
                        columns={"share_green_vona2019_6d": "share_green_vona2019"}
                    )
                )

        return self._green_occupations_vona2019

    @property
    def green_occupations_gilli2020(self):
        """
        Greenness scores from Gilli et al. (2018).
        ISCO 3-digit level, 127 occupations.

        Returns
        -------

        """
        if self._green_occupations_gilli2020 is None:
            self._green_occupations_gilli2020 = pd.read_excel(
                os.path.join(
                    useful_paths.data_raw,
                    "onet",
                    "Gilli2020_AppC_Greenness_ISCO3D.xlsx",
                ),
                dtype={"ISCO_code": "str"},
            )
        return self._green_occupations_gilli2020

    @property
    def brown_occupations_vona2018(self):
        """
        Classification of brown occupations from Vona et al. (2018).
        SOC 6-digit level, 111 occupations.

        Returns
        -------

        """
        if self._brown_occupations_vona2018 is None:
            self._brown_occupations_vona2018 = pd.read_csv(
                os.path.join(self.data_raw, "onet", "Vona2018_brown_occupations.csv")
            )

            # pad to 8 digits
            # self._brown_occupations_vona2018["soc_code"] = (
            #     self._brown_occupations_vona2018["soc_code"] + ".00"
            # )

        return self._brown_occupations_vona2018

    @property
    def green_tasks_narrow_jrc(self):
        """
        Greenness scores from O*NET GTP mapped to Italian Classification of
        Occupations (CP2011) at 5-digit level.

        Notes:
        - 117 unique ISCO 5-digit occupations are covered (narrow classification scope)
        - Harmonized occ name of “Machinery operators for dry cleaning, bleaching and
            dyeing of industrial yarns and fabrics” (7.2.6.4.0) in excel file
        - corrected inconsistent greenness scores across tasks of same occupations
        Returns
        -------

        """
        if self._green_tasks_narrow_jrc is None:
            self._green_tasks_narrow_jrc = pd.read_excel(
                os.path.join(
                    self.data_raw, "onet", "Green_Occupations_jrc_v1_02052022.xlsx"
                )
            )
        return self._green_tasks_narrow_jrc

    @property
    def green_occupations_narrow_jrc(self):
        """
        Greenness scores from O*NET GTP mapped to Italian Classification of
        Occupations (CP2011) at 5-digit level.

        Task-level data collapsed into occupation-level data

        Returns
        -------

        """
        if self._green_occupations_narrow_jrc is None:
            # collapse task-level into occupation-level data
            group_cols = ["isco08_jrc", "occ_eng_jrc"]

            # define how cols should be aggregated
            agg_dict = {
                "n_green_tasks_jrc": np.sum,
                "n_tasks_jrc": np.mean,
                "greenness_jrc": np.mean,
            }

            # aggregate
            greenness_jrc = (
                self.green_tasks_narrow_jrc.groupby(group_cols)
                .aggregate(agg_dict)
                .reset_index()
            )

            # check if keys are unique
            assert greenness_jrc[group_cols[0]].duplicated().sum() == 0
            assert greenness_jrc[group_cols[1]].duplicated().sum() == 0

            self._green_occupations_narrow_jrc = greenness_jrc
        return self._green_occupations_narrow_jrc


class Esco(UsefulPaths):
    """
    Stores data from the European ESCO classification.
    """

    def __init__(self, language="en", version="v1.1.0", version_newest="v1.1.0"):
        # inherit class storing useful paths
        UsefulPaths.__init__(self=self)

        # extract language and versions
        self.esco_language = language  # "en"
        self.esco_version = version  # "v1.0.3" or "v1.1.0"
        self.esco_version_newest = version_newest  # "v1.1.0"
        self.esco_skills_hierarchy_version = (
            "v1.0.8" if self.esco_version == "v1.0.3" else "v1.1.0"
        )
        # -----------------------------------------------------------------------------
        # static parameters for GBN skill and occupation classifications
        # -----------------------------------------------------------------------------
        self.skill_types_gbn = ["green", "brown", "neutral"]
        self.skill_clsf_ds = ["esco", "eth"]
        self.skill_col_fmt = "skill_{type}_{ds}"

        # note: formatting string below needs to start with "share"
        self.gbn_share_fmt = "share_{type}_{ds}"

        # column names of unvalidated GBN skills in ESCO
        self.green_id_colname_esco = "skill_green_esco"
        self.brown_id_colname_esco = "skill_brown_esco"
        self.neutral_id_colname_esco = "skill_neutral_esco"

        # column names of ETH-validated GBN skills in ESCO
        self.green_id_colname_eth = "skill_green_eth"
        self.brown_id_colname_eth = "skill_brown_eth"
        self.neutral_id_colname_eth = "skill_neutral_eth"

        # column names of merged classification columns
        self.skills_clsf_colname_esco = "skill_classification_esco"
        self.skills_clsf_colname_eth = "skill_classification_eth"

        # column names of GBN occupations in O*NET
        self.green_id_onet = "is_green_onet"
        self.brown_id_onet = "is_brown_onet"
        self.neutral_id_onet = "is_neutral_onet"

        # define fmt strings for ISCO classification
        self.fmt_string_isco_lvl = "isco_level_{}"
        self.fmt_string_isco_label = "isco_label_{}"
        self.isco_join_col_of = "isco_code"

        # -----------------------------------------------------------------------------
        # functions for the aggregation of occupation data
        # -----------------------------------------------------------------------------
        # TODO: add agg func for discrete GBN classifications
        self.agg_func_dict = {
            "mean": np.nanmean,
            # "std": np.nanstd,
            # "median": np.nanmedian,
            # "iqr": stats_utils.naniqr,
        }
        # -----------------------------------------------------------------------------
        # initialise class variables, to be assigned later
        # -----------------------------------------------------------------------------
        self.osm = None
        self.osim = None

        # ESCO Core Data
        self._occupations = None
        self._occupations_to_skills = None
        self._skills = None
        self._skills_v103 = None
        self._skills_green = None
        self._skills_brown = None
        self._skills_hierarchy = None
        self._skills_groups = None
        self._isco_groups = None

        # Mapping Career Causeways
        self._skills_hierarchy_mcc = None
        self._skills_coreness_mcc = None
        self._job_zones_mcc = None
        self._covid_exposure_mcc = None

        # MCC crosswalks to onet
        self._crosswalk_onet_esco_mcc_full = None
        self._crosswalk_onet_esco_mcc_reduced = None

        # ONET-specific data
        self._green_occupations_onet = None
        self._brown_occupations_onet = None

        # merged data
        self.skills_metadata = None
        self.occupation_metadata = None

    @property
    def occupations(self):
        """
        ESCO occupations pillar.

        Returns
        -------

        """
        if self._occupations is None:
            self._occupations = pd.read_csv(
                os.path.join(
                    self.data_raw,
                    "esco",
                    self.esco_version,
                    "occupations_{}.csv".format(self.esco_language),
                ),
                dtype={"iscoGroup": "str"},
            )
        return self._occupations

    @property
    def skills(self):
        """
        ESCO skills, knowledge & competences pillar.

        Returns
        -------

        """
        if self._skills is None:
            self._skills = pd.read_csv(
                os.path.join(
                    self.data_raw,
                    "esco",
                    self.esco_version,
                    "skills_{}.csv".format(self.esco_language),
                )
            )
        return self._skills

    @property
    def skills_v103(self):
        """
        ESCO skills, knowledge & competences pillar from v.1.0.3. Needed for joining
        skills coreness data from MCC project.

        Returns
        -------

        """
        if self._skills_v103 is None:
            self._skills_v103 = pd.read_csv(
                os.path.join(
                    self.data_raw,
                    "esco",
                    "v1.0.3",
                    "skills_{}.csv".format(self.esco_language),
                )
            )
        return self._skills_v103

    @property
    def occupations_to_skills(self):
        """
        Mapping between ESCO occupations and skills.

        Returns
        -------

        """
        if self._occupations_to_skills is None:
            self._occupations_to_skills = pd.read_csv(
                os.path.join(
                    self.data_raw,
                    "esco",
                    self.esco_version,
                    "occupationSkillRelations.csv",
                )
            )
        return self._occupations_to_skills

    @property
    def skills_green(self):
        """
        ESCO green skill labels

        Returns
        -------

        """
        if self._skills_green is None:
            self._skills_green = pd.read_excel(
                os.path.join(
                    self.data_raw,
                    "esco",
                    self.esco_version_newest,
                    "GreenBrownSkillsValidationETH.xlsx",
                ),
                sheet_name="greenSkillsCollection_en",
            )
        return self._skills_green

    @property
    def skills_brown(self):
        """
        ESCO brown skill labels

        Returns
        -------

        """
        if self._skills_brown is None:
            self._skills_brown = pd.read_excel(
                os.path.join(
                    self.data_raw,
                    "esco",
                    self.esco_version_newest,
                    "GreenBrownSkillsValidationETH.xlsx",
                ),
                sheet_name="brownSkillsCollection_en",
            )
        return self._skills_brown

    @property
    def skills_hierarchy(self):
        """
        ESCO skills hierarchy

        Returns
        -------

        """
        if self._skills_hierarchy is None:
            self._skills_hierarchy = pd.read_csv(
                os.path.join(
                    self.data_raw,
                    "esco",
                    self.esco_skills_hierarchy_version,
                    "skillsHierarchy_{}.csv".format(self.esco_language),
                )
            )
        return self._skills_hierarchy

    @property
    def skills_groups(self):
        if self._skills_groups is None:
            self._skills_groups = pd.read_csv(
                os.path.join(
                    self.data_raw,
                    "esco",
                    self.esco_version,
                    "skillGroups_{}.csv".format(self.esco_language),
                )
            )
        return self._skills_groups

    @property
    def isco_groups(self):
        if self._isco_groups is None:
            self._isco_groups = pd.read_csv(
                os.path.join(
                    self.data_raw,
                    "esco",
                    self.esco_version,
                    "ISCOGroups_{}.csv".format(self.esco_language),
                ),
                dtype={"code": "str"},
            )
            # make sure to correct codes of military occupations
            self._isco_groups["code"] = (
                self._isco_groups["conceptUri"]
                .str.split("/")
                .apply(pd.Series)
                .loc[:, 5]
                .str.lstrip("C")
            )
        return self._isco_groups

    @property
    def skills_hierarchy_mcc(self):
        if self._skills_hierarchy_mcc is None:
            self._skills_hierarchy_mcc = pd.read_csv(
                os.path.join(
                    self.data_raw,
                    "mapping-career-causeways",
                    "codebase",
                    "data",
                    "processed",
                    "ESCO_skills_hierarchy",
                    "ESCO_skills_hierarchy.csv",
                )
            )
        return self._skills_hierarchy_mcc

    @property
    def skills_coreness_mcc(self):
        """
        Skills coreness for ESCO v.1.0.3. A composite measure for the centrality of a
        skill in the overall ESCO skill network.

        Returns
        -------

        """
        if self._skills_coreness_mcc is None:
            self._skills_coreness_mcc = pd.read_csv(
                os.path.join(
                    self.data_raw,
                    "mapping-career-causeways",
                    "codebase",
                    "data",
                    "interim",
                    "upskilling_analysis",
                    "skills_coreness_measure.csv",
                )
            )
        return self._skills_coreness_mcc

    @property
    def job_zones_mcc(self):
        """
        O*NET job zone mapped to ESCO v.1.0.3. Comprises a worker's education level,
        related work experience and on-the-job training.

        Returns
        -------

        """
        if self._job_zones_mcc is None:
            self._job_zones_mcc = pd.read_csv(
                os.path.join(
                    self.data_raw,
                    "mapping-career-causeways",
                    "codebase",
                    "data",
                    "processed",
                    "linked_data",
                    "ESCO_occupations_Job_Zones.csv",
                )
            )
        return self._job_zones_mcc

    @property
    def covid_exposure_mcc(self):
        if self._covid_exposure_mcc is None:
            self._covid_exposure_mcc = pd.read_csv(
                os.path.join(
                    self.data_raw,
                    "mapping-career-causeways",
                    "codebase",
                    "data",
                    "processed",
                    "linked_data",
                    "ESCO_occupations_COVID_Exposure.csv",
                )
            )
        return self._covid_exposure_mcc

    @property
    def crosswalk_onet_esco_mcc_full(self):
        """
        ONET-ESCO crosswalk for the full set of ESCO v.1.0.3 occupations (n=2941).

        Returns
        -------

        """
        if self._crosswalk_onet_esco_mcc_full is None:
            self._crosswalk_onet_esco_mcc_full = pd.read_csv(
                os.path.join(
                    self.data_raw,
                    "mapping-career-causeways",
                    "codebase",
                    "data",
                    "processed",
                    "ESCO_ONET_xwalk_full.csv",
                )
            )
        return self._crosswalk_onet_esco_mcc_full

    @property
    def crosswalk_onet_esco_mcc_reduced(self):
        pass

    def occupations_to_skills_md(self):
        """Enrich occupation-skills mapping with additional metadata."""
        cols_to_use = self.skills_metadata.columns.difference(
            self.occupations_to_skills.columns
        )

        # join select skills metadata to mapping
        keep_cols_skills_md = [
            "conceptUri",
            "reuseLevel",
            "preferredLabel",
            "description",
            "skill_classification_esco",
            "coreness",
        ]
        join_col_right = "conceptUri"
        occ_skills_mapping_smd_merged = pd.merge(
            left=self.occupations_to_skills,
            right=self.skills_metadata[keep_cols_skills_md],
            left_on="skillUri",
            right_on=join_col_right,
            how="left",
        ).drop(columns=[join_col_right])

        # join select occ metadata to mapping
        occupations = self.occupations.reset_index().rename(columns={"index": "id"})

        keep_cols_occ_md = [
            "id",
            "conceptUri",
            "iscoGroup",
            "preferredLabel",
            "description",
        ]

        occ_skills_mapping_all_merged = pd.merge(
            left=occ_skills_mapping_smd_merged,
            right=occupations[keep_cols_occ_md],
            left_on="occupationUri",
            right_on=join_col_right,
            how="left",
            suffixes=("_skills", "_occs"),
        ).drop(columns=[join_col_right])

        return occ_skills_mapping_all_merged

    def get_skills_for_occ(self, id=None, metadata=True, summary_only=True):
        if metadata:
            df_all = self.occupations_to_skills_md()
        else:
            raise NotImplementedError

        # subset based on occupation id or label
        if isinstance(id, str):
            search_col = "preferredLabel_occs"
        elif isinstance(id, int):
            search_col = "id"
        else:
            raise NotImplementedError

        df = df_all[df_all[search_col] == id]
        df = df.reset_index(drop=True)

        # summary
        if summary_only:
            keep_cols = [
                "preferredLabel_occs",
                "preferredLabel_skills",
                "relationType",
                "reuseLevel",
                "skillType",
                "skill_classification_esco",
                "coreness",
            ]
            df = df[keep_cols]

        return df

    def label_osm(self, osm):
        osm.index = self.occupations["preferredLabel"].values
        osm.columns = self.skills["preferredLabel"].values
        return osm

    # note: replaces function 'occupation_skills_matrix'
    def read_occ_skills_matrix(
        self,
        encoding_essential=1,
        encoding_optional=2,
        encoding_none=0,
        override=False,
        weight_optional=0.5,
        return_version="raw",
        assign_labels=False,
        target_path=os.path.join(
            useful_paths.data_processed, "esco", "occ_skills_matrix.pkl"
        ),
    ):
        """
        Calculate occupation-skills matrix (OSM) based on ESCO data. In the OSM rows
        denote occupations and columns skills. Essential and optional skills are
        differentiated based on the encodings. Weights for the optional skills can
        be assigned.

        Parameters
        ----------
        encoding_essential
        encoding_optional
        encoding_none
        override
        weight_optional
        return_version : str
            one of ["raw", "unweighted", "weighted"]
        assign_labels
        target_path

        Returns
        -------

        """

        if override or not os.path.exists(target_path):
            # build occupation skills matrix
            errors = 0
            skill_vectors = []

            for i in tqdm(range(len(self.occupations))):
                occ_uri = self.occupations.iloc[i, :][1]

                # lookup corresponding skills
                skill_list = self.occupations_to_skills[
                    self.occupations_to_skills["occupationUri"] == occ_uri
                ]

                # create vector
                skill_vector = []
                for j, skill in enumerate(self.skills.conceptUri.values):

                    if skill in skill_list.skillUri.values:
                        relation_type = skill_list.loc[
                            skill_list.skillUri == skill, "relationType"
                        ].values[0]

                        # skill needed for occupation and essential
                        if relation_type == "essential":
                            skill_vector.append(encoding_essential)
                        # skill needed for occupation and optional
                        elif relation_type == "optional":
                            skill_vector.append(encoding_optional)
                    else:
                        # skill not needed for occupation
                        skill_vector.append(encoding_none)

                # checkme: what is this line good for?
                indices = [i for i, j in enumerate(skill_vector) if j == 1]

                # sanity check
                if len(skill_list.skillUri) != np.sum(
                    np.invert(np.array(skill_vector) == 0)
                ):
                    errors += 1

                # append
                skill_vectors.append(skill_vector)

            # info
            print("n_errors: ", errors)

            # create df
            occ_skills_matrix_eo = pd.DataFrame(
                index=self.occupations.conceptUri,
                columns=self.skills.conceptUri,
                data=np.array(skill_vectors),
            )

            # save to disk
            occ_skills_matrix_eo.to_pickle(target_path)
        else:
            # read matrix from disk
            occ_skills_matrix_eo = pd.read_pickle(target_path)

            # differentiate between weighted and unweighted matrix
            if return_version == "unweighted":
                # assign encoding of essential skills (1) to optional skills
                occ_skills_matrix_eo.replace(
                    to_replace=encoding_optional, value=encoding_essential, inplace=True
                )
            elif return_version == "weighted":
                # assign specified weight to optional skills
                occ_skills_matrix_eo.replace(
                    to_replace=encoding_optional, value=weight_optional, inplace=True
                )
            else:
                # return raw, encoded version
                pass

            # optionally return labelled OSM
            if assign_labels:
                occ_skills_matrix_eo = self.label_osm(occ_skills_matrix_eo)

        return occ_skills_matrix_eo

    # todo: make functional
    def _calc_osm_variants(self, occ_skills_matrix_eo):
        """
        Calculate weighted and return_unweighted forms from raw OSM.

        Parameters
        ----------
        occ_skills_matrix_eo

        Returns
        -------

        """
        # get weights of weighted form
        replace_weighted = {
            1: self.config_data["ESCO"]["WEIGHT_ESSENTIAL_SKILL"],
            2: self.config_data["ESCO"]["WEIGHT_OPTIONAL_SKILL"],
        }

        # calc weighted form
        occ_skills_matrix_weighted = occ_skills_matrix_eo.replace(
            to_replace=replace_weighted
        )

        # calc return_unweighted form
        occ_skills_matrix_unweighted = occ_skills_matrix_eo.replace(
            to_replace=[1, 2], value=self.config_data["ESCO"]["WEIGHT_UNIFORM"]
        )

        return {
            "osm_weighted": occ_skills_matrix_weighted,
            "osm_unweighted": occ_skills_matrix_unweighted,
        }

    def occupation_skills_matrix(self):
        """
        Calculate occupation-skills matrix (OSM) based on ESCO data. In the OSM rows
        denote occupations and columns skills. Depending on the settings in the main
        configuration file, essential and optional skills are assigned different
        weights on an occupation-to-occupation basis. An return_unweighted variant is
        always also created as a baseline.

        Returns
        -------
        dict
            Dictionary storing two variants of the OSM as pd.DataFrame:
                (1) the weighted form (coded as np.float),
                (2) the return_unweighted/uniform form (coded as np.int8)
        """
        logger = logging.getLogger(__name__)
        logger.info("Creating occupation-skills matrix.")

        target_path = os.path.join(
            self.data_interim,
            "esco",
            "occ_skills_matrix.pkl",
        )

        if not os.path.exists(target_path):
            # build occupation skills matrix
            errors = 0
            skill_vectors = []

            for i in tqdm(range(len(self.occupations))):
                occ_uri = self.occupations.iloc[i, :][1]

                # lookup corresponding skills
                skill_list = self.occupations_to_skills[
                    self.occupations_to_skills["occupationUri"] == occ_uri
                ]

                # create vector
                skill_vector = []
                for j, skill in enumerate(self.skills.conceptUri.values):

                    if skill in skill_list.skillUri.values:
                        relation_type = skill_list.loc[
                            skill_list.skillUri == skill, "relationType"
                        ].values[0]

                        # skill needed for occupation and essential
                        if relation_type == "essential":
                            skill_vector.append(1)
                        # skill needed for occupation and optional
                        elif relation_type == "optional":
                            skill_vector.append(2)
                    else:
                        # skill not needed for occupation
                        skill_vector.append(0)

                # checkme: what is this line good for?
                indices = [i for i, j in enumerate(skill_vector) if j == 1]

                # sanity check
                if len(skill_list.skillUri) != np.sum(
                    np.invert(np.array(skill_vector) == 0)
                ):
                    errors += 1

                # append
                skill_vectors.append(skill_vector)

            # info
            print("n_errors: ", errors)

            # create df
            occ_skills_matrix_eo = pd.DataFrame(
                index=self.occupations.conceptUri,
                columns=self.skills.conceptUri,
                data=np.array(skill_vectors),
            )

            # save to disk
            occ_skills_matrix_eo.to_pickle(target_path)
        else:
            # read matrix from disk
            occ_skills_matrix_eo = pd.read_pickle(target_path)

        return self._calc_osm_variants(occ_skills_matrix_eo=occ_skills_matrix_eo)

    def occupation_similarity_matrix(self, version="weighted", weight_optional=0.5):
        """
        Calculate weighted and return_unweighted forms of the co-occurrence occupation
        similarity matrix (OSIM).

        Returns
        -------
        dict
            Dictionary containing X versions of the OSIM:
                (1) Skills COO based on weighted OSM
                (2) Skills COO based on return_unweighted OSM
        """
        logger = logging.getLogger(__name__)

        # read base OSM and calculate variants
        osm_dict = self.read_occ_skills_matrix(
            return_version=version, weight_optional=weight_optional
        )

        target_path_template = os.path.join(
            self.data_interim,
            "esco",
            self.esco_version,
            "occ_sim_matrix_{}_coo.pkl",
        )

        # calculate co-occurrence matrices for both variants
        # (via matrix multiplication with transpose form)
        osim_dict = {}
        for osm_version, osm_data in osm_dict.items():
            variant = osm_version.split("_")[1]

            # fixme: calculation of unweighted COOC matrix does not work
            if variant == "return_unweighted":
                continue

            logger.info("Calculating {} occupation similarity matrix.".format(variant))
            target_path = target_path_template.format(variant)

            if not os.path.exists(target_path):
                occ_sim_matrix_coo = np.dot(
                    osm_data.values, osm_data.values.transpose()
                )

                # to df
                df_occ_sim_matrix_weighted_coo = pd.DataFrame(
                    index=self.occupations.conceptUri,
                    columns=self.occupations.conceptUri,
                    data=occ_sim_matrix_coo,
                )

                # save
                df_occ_sim_matrix_weighted_coo.to_pickle(target_path)

            else:
                osim_dict["osim_{}".format(variant)] = pd.read_pickle(target_path)

        return osim_dict

    def combine_skills_metadata(
        self,
        override=True,
        export=True,
        output_dir=os.path.join(useful_paths.data_interim, "esco"),
        output_fname="skills_metadata_en",
        variable_selection=None,
    ):
        """
        Combine all skill-level metadata (SMD).

        Evaluation comments
            - ESCO v.1.1.0 comprises 13891 skills compared to v.1.0.3 with 13485 skills.
            - Hence, overall 406 new skills in v.1.1.0 compared to v.1.0.3.
            - 570 skills labelled as green in v.1.1.0, some of which are presumably new,
            while other's already existed.
            - After merging we don't obtain coreness values for 512 skills in v.1.1.0.
            - This means that the conceptUri of 512 - 406 = 106 skills that existed in
            the last version must have changed.

        Returns
        -------
        smd : pd.DataFrame
            DataFrame containing the raw ESCO skills pillar data enriched by additional
            metadata (green labels, coreness).
        """
        # target fpath
        target_path = os.path.join(output_dir, output_fname + ".pkl")

        if override or not os.path.exists(target_path):
            # -------------------------------------------------------------------------
            # Green Skills
            # -------------------------------------------------------------------------
            set_diff = utils.get_set_diff(self.skills_green, self.skills, "conceptUri")

            # copy skills df and join green skills information
            smd = self.skills.copy()
            smd = smd.merge(
                right=self.skills_green[set_diff],
                on="conceptUri",
                how="left",
                suffixes=["", "_green"],
                validate="one_to_one",
            )

            # TODO: add eth colnames once skills are fully classified
            smd = smd.fillna(value={self.green_id_colname_esco: False})

            # -------------------------------------------------------------------------
            # Brown Skills
            # -------------------------------------------------------------------------
            set_diff = utils.get_set_diff(self.skills_brown, self.skills, "conceptUri")

            smd = smd.merge(
                right=self.skills_brown[set_diff],
                on="conceptUri",
                how="left",
                validate="one_to_one",
                suffixes=["", "_brown"],
            )

            # TODO: add eth column names once skills are fully classified
            smd = smd.fillna(value={self.brown_id_colname_esco: False})

            # -------------------------------------------------------------------------
            # Derive Neutral Skills
            # -------------------------------------------------------------------------
            smd = classify_by_gbn(
                df=smd,
                col_name_green=self.green_id_colname_esco,
                col_name_brown=self.brown_id_colname_esco,
                col_name_neutral=self.neutral_id_colname_esco,
                col_name_clfc=self.skills_clsf_colname_esco,
            )

            # TODO: repeat for eth data once skills are fully classified
            # smd = classify_by_gbn(
            #     df=smd,
            #     col_name_green=self.green_id_colname_eth,
            #     col_name_brown=self.brown_id_colname_eth,
            #     col_name_neutral=self.neutral_id_colname_eth,
            #     col_name_clfc=self.skills_clsf_colname_eth
            # )

            # -------------------------------------------------------------------------
            # Coreness Metric
            # -------------------------------------------------------------------------
            # append conceptUri of ESCO v1.0.3 skills
            keep_cols = ["conceptUri", "preferredLabel"]
            skills_coreness_with_uri = self.skills_coreness_mcc.merge(
                right=self.skills_v103[keep_cols],
                left_on="preferred_label",
                right_on="preferredLabel",
                how="left",
                validate="one_to_one",
            )

            # Merge Coreness to Skills Pillar
            keep_cols = ["conceptUri", "coreness"]
            smd = smd.merge(
                right=skills_coreness_with_uri[keep_cols],
                on="conceptUri",
                how="left",
                validate="one_to_one",
            )

            # drop columns
            if variable_selection is not None:
                smd = smd[variable_selection]

            # assign to class var
            self.skills_metadata = smd

            # save for inspection
            # optional: save for inspection
            if export:
                utils.save_df_to_files(
                    smd, output_dir=output_dir, fname_no_ext=output_fname, sep=","
                )

        else:
            # read and assign to class var
            smd = pd.read_pickle(target_path)
            self.skills_metadata = smd

        return smd

    # todo: implement weighted form of esco-based greenness measure
    def calc_gbn_shares_skill_based(
        self, skills_metadata, drop_counts=True, essential_only=False
    ):
        # note: no weighting possible atm, just differentiation between essential and
        #  optional skills

        # read OSM version
        if not essential_only:
            osm = self.read_occ_skills_matrix(return_version="unweighted")
        else:
            osm = self.read_occ_skills_matrix(
                return_version="weighted", weight_optional=0
            )

        # number of occupation-specific skills
        n_total_specific_skills = osm.sum(axis=1).values

        data_out = {
            "n_total_specific_skills": n_total_specific_skills,
        }

        # calc occupational shares for each skill type
        colnames = []

        # TODO: remove [:1] once ETH skill classification is finished
        for ds in self.skill_clsf_ds[:1]:
            for skill_type in self.skill_types_gbn:
                col = self.skill_col_fmt.format(type=skill_type, ds=ds)
                colname_shares = self.gbn_share_fmt.format(type=skill_type, ds=ds)

                # number of occupation-specific green/brown/neutral skills
                specific_skills = (
                    osm.values * skills_metadata[col].astype(np.int8).values
                )

                n_gbn_specific_skills = specific_skills.sum(axis=1)

                # occupational share
                occ_share = n_gbn_specific_skills / n_total_specific_skills

                # append to dict
                data_out[
                    "n_{type}_specific_skills_{ds}".format(type=skill_type, ds=ds)
                ] = n_gbn_specific_skills

                data_out[colname_shares] = occ_share
                colnames.append(colname_shares)

        df_occ_shares_per_skill_type = pd.DataFrame(index=osm.index, data=data_out)

        # check if shares sum to 100%
        check_sum_of_shares = df_occ_shares_per_skill_type[colnames].sum(axis=1).values

        if not essential_only:
            assert np.allclose(check_sum_of_shares, 1)
        else:
            # occupation "specialist dentist" has no essential skills
            # 'http://data.europa.eu/esco/occupation/a580e79a-b752-49c1-b033-b5ab2b34bfba'
            # pass test for the moment
            pass

        # drop skill type counts
        if drop_counts:
            df_occ_shares_per_skill_type = df_occ_shares_per_skill_type.drop(
                columns=df_occ_shares_per_skill_type.columns[
                    df_occ_shares_per_skill_type.columns.str.startswith("n")
                ].values
            )

        # classify into discrete GBN categories
        # TODO: check if only one True per col.
        df_occ_shares_per_skill_type[
            "gbn_classification_esco"
        ] = df_occ_shares_per_skill_type[colnames].idxmax(axis=1)

        # rename to green/brown/neutral.
        df_occ_shares_per_skill_type["gbn_classification_esco"] = (
            df_occ_shares_per_skill_type["gbn_classification_esco"]
            .str.split(pat="_", expand=True)
            .iloc[:, 1]
        )

        return df_occ_shares_per_skill_type.reset_index()

    def read_greenness_task_based(self):
        greenness_onet_gtp = pd.read_excel(
            io=os.path.join(
                self.data_raw,
                "onet",
                "Onet_GreenTask_AppA.xlsx",
            ),
            sheet_name="Occupations",
        )

        greenness_onet_vona = pd.read_excel(
            io=os.path.join(self.data_raw, "onet", "Vona2018_table_a1.xlsx"),
            sheet_name="Greenness",
        )

        # merge GTP and Vona variants
        greenness_onet = greenness_onet_gtp.copy()
        greenness_onet = greenness_onet.merge(
            right=greenness_onet_vona,
            on="onet_code",
            how="left",
            validate="1:1",
            suffixes=["_gtp", "_vona2018"],
        )

        # merge to ESCO occupations pillar via crosswalk
        greenness_onet_esco = greenness_onet.copy()
        greenness_onet_esco = greenness_onet_esco.merge(
            right=self.crosswalk_onet_esco_mcc_full,
            on="onet_code",
            how="left",
            validate="1:m",
        )

        # downcast dtypes
        greenness_onet_esco = utils.downcast_df(greenness_onet_esco)

        # save
        greenness_onet.to_csv(
            os.path.join(self.data_interim, "onet", "task_based_greenness_onet.csv"),
            sep=";",
        )

        greenness_onet_esco.to_csv(
            os.path.join(
                self.data_interim,
                "onet",
                "task_based_greenness_onet_esco_{}.csv".format(
                    self.esco_version.replace(".", "")
                ),
            ),
            sep=";",
        )

        return greenness_onet_esco

    def read_brown_occupations_vona2018(self):
        """
        Read Vona et al. 2018 brown occupation classification and merge to ESCO occupations via Nesta ONET-ESCO crosswalk.

        Returns
        -------

        """
        brown_occs_vona2018 = pd.read_csv(
            os.path.join(self.data_raw, "onet", "Vona2018_brown_occupations.csv")
        )

        # pad to 8 digits
        brown_occs_vona2018["soc_code"] = brown_occs_vona2018["soc_code"] + ".00"

        # add binary brown occupation classification
        brown_occs_vona2018[self.brown_id_onet] = np.ones(
            brown_occs_vona2018.shape[0], dtype=bool
        )

        # merge to ESCO occupations pillar via crosswalk
        brown_occs_vona2018_esco = brown_occs_vona2018.merge(
            right=self.crosswalk_onet_esco_mcc_full,
            left_on="soc_code",
            right_on="onet_code",
            how="left",
            # validate="m:1",
        ).drop(columns=["soc_code", "occupation"], axis=1)

        # drop duplicates & missing vals
        dups = brown_occs_vona2018_esco.concept_uri.duplicated()
        brown_occs_vona2018_esco = brown_occs_vona2018_esco.loc[~dups]

        brown_occs_vona2018_esco.dropna(subset="concept_uri", inplace=True)

        # downcast dtype
        brown_occs_vona2018_esco = utils.downcast_df(brown_occs_vona2018_esco)

        # rearrange
        col_order = [
            "id",
            "concept_uri",
            "preferred_label",
            "isco_level_4",
            "is_brown_onet",
            "onet_code",
            "onet_occupation",
        ]
        brown_occs_vona2018_esco = brown_occs_vona2018_esco[col_order]

        # save
        brown_occs_vona2018_esco.to_csv(
            os.path.join(
                self.data_interim,
                "onet",
                "brown_occupations_vona2018_esco_{}.csv".format(
                    self.esco_version.replace(".", "")
                ),
            ),
            sep=";",
        )

        return brown_occs_vona2018_esco

    def isco_correspondence_table(self):
        df_isco = self.isco_groups.copy()

        for lvl in [1, 2, 3, 4]:
            df_sub = df_isco.loc[
                df_isco.code.str.len() == lvl, "preferredLabel"
            ].reindex(df_isco.index)
            df_isco["isco_label_{}".format(lvl)] = df_sub

        df_isco = df_isco.rename(
            columns={
                "code": "isco_code",
                "preferredLabel": "isco_label",
            }
        )

        return df_isco

    def combine_occupation_metadata(
        self,
        skills_metadata,
        override=True,
        export=True,
        variable_selection=None,
        output_dir=os.path.join(useful_paths.data_interim, "esco"),
        output_fname="occ_metadata_en",
    ):
        # target fpath
        target_fpath = os.path.join(output_dir, output_fname + ".pkl")

        if override or not os.path.exists(target_fpath) is None:
            # init container
            omd = self.occupations.copy()

            # attach isco labels per level
            omd = self.attach_isco_to_occupations(omd)

            # merge onet codes and names via crosswalk
            # omd = omd.merge(
            #     right=self.crosswalk_onet_esco_mcc_full,
            #     left_on="conceptUri",
            #     right_on="concept_uri",
            #     how="left",
            #     suffixes=["", "_y"],
            #     validate="1:1",
            # )

            # Read data sets
            df_gbn_shares_esco_all = self.calc_gbn_shares_skill_based(
                skills_metadata=skills_metadata
            )

            df_gbn_shares_esco_ess = self.calc_gbn_shares_skill_based(
                skills_metadata=skills_metadata, essential_only=True
            )

            # merge variables to occupation df
            # -------------------------------------------------------------------------
            # TODO: join --> ONET job zone, COVID Exposure

            # join ESCO-based GBN shares
            omd = omd.merge(
                right=df_gbn_shares_esco_all,
                on="conceptUri",
                how="left",
                suffixes=["", "_y"],
                validate="1:1",
            )

            omd = omd.merge(
                right=df_gbn_shares_esco_ess,
                on="conceptUri",
                how="left",
                suffixes=["", "_ess"],
                validate="1:1",
            )

            """
            df_greenness_onet_esco = self.read_greenness_task_based()
            df_brown_occs_vona2018_esco = self.read_brown_occupations_vona2018()
            
            # join O*NET-based G shares
            omd = omd.merge(
                right=df_greenness_onet_esco.dropna(subset="concept_uri"),
                left_on="conceptUri",
                right_on="concept_uri",
                how="left",
                suffixes=["", "_y"],
                validate="1:1",
            )

            # join O*Net based Brown occupations
            omd = omd.merge(
                right=df_brown_occs_vona2018_esco,
                left_on="conceptUri",
                right_on="concept_uri",
                how="left",
                suffixes=["", "_y"],
                validate="1:1",
            )

            # create ONET-based G/B/N classification

            # TODO: refactor to function
            # classify green  & neutral occupations (data from GTP covers more
            # occupations, therefore using those. correlation with Vona 2018
            # greenness scores ~ 1)
            green_threshold = 0

            omd[self.green_id_onet] = (
                omd[self.gbn_share_fmt.format(type="green", ds="gtp")] > green_threshold
            )

            omd[self.brown_id_onet] = omd[self.brown_id_onet].fillna(False)

            omd[self.neutral_id_onet] = (omd[self.green_id_onet] == False) & (
                omd[self.brown_id_onet] == False
            )

            # there are 30 occupations that have been matched to both brown and green
            query_ambiguous_cases = (omd[self.brown_id_onet] == True) & (
                omd[self.green_id_onet] == True
            )

            # define ambiguous cases as brown (see thesis)
            omd.loc[query_ambiguous_cases, self.green_id_onet] = False
            # occ_metadata = occ_metadata.reset_index(drop=True)

            # create single classification column
            # TODO: assert check if only one True per col.
            omd["gbn_classification_onet"] = omd[
                [self.brown_id_onet, self.green_id_onet, self.neutral_id_onet]
            ].idxmax(axis=1)

            # rename to green/brown/neutral.
            omd["gbn_classification_onet"] = (
                omd["gbn_classification_onet"]
                .str.split(pat="_", expand=True)
                .iloc[:, 1]
            )

            # filter out duplicate columns
            omd = omd.drop(omd.filter(regex="_y$").columns.tolist(), axis=1)

            # check if all occs are classified
            assert omd["gbn_classification_onet"].isna().sum() == 0

            # make sure there are no duplicates
            assert omd.conceptUri.duplicated().sum() == 0

            # assign to class variable
            self.occupation_metadata = omd

            # optional: save for inspection
            if export:
                utils.save_df_to_files(
                    omd, output_dir=output_dir, fname_no_ext=output_fname, sep=";"
                )
        else:
            # read and assign to class var
            omd = pd.read_pickle(target_fpath)
            self.occupation_metadata = omd
            """

            if variable_selection is not None:
                omd = omd[variable_selection]

        return omd

    def attach_isco_to_occupations(self, occupations, top_level=4):
        # pad at 4d level
        occupations[self.fmt_string_isco_lvl.format(top_level)] = occupations[
            "iscoGroup"
        ].str.pad(width=top_level, side="left", fillchar="0")
        # decompose isco 4-digit into lower levels
        for lvl in [1, 2, 3]:
            new_colname = self.fmt_string_isco_lvl.format(lvl)
            occupations[new_colname] = occupations["isco_level_4"].str[0:lvl]

        # attach labels at each level
        df_isco = self.isco_correspondence_table()
        for lvl in [1, 2, 3, 4]:
            occupations = occupations.merge(
                left_on=self.fmt_string_isco_lvl.format(lvl),
                right=df_isco.loc[
                    :, ("isco_code", self.fmt_string_isco_label.format(lvl))
                ],
                right_on="isco_code",
            ).drop(columns=["isco_code"])

        return occupations

    # todo (minor): add COVID and automation-related data to agg rerun_simulations
    def aggregate_occ_data_by_isco(
        self,
        isco08_digits=[1, 2, 3, 4],
        use_weights=False,
    ):
        if self.occupation_metadata is None:
            self.combine_occupation_metadata()
        self.occupation_metadata["n_occ_esco"] = np.ones(len(self.occupation_metadata))

        # one csv per isco level
        target_fpath_csv = os.path.join(
            self.data_interim,
            "esco",
            self.esco_version,
            "occ_metadata_{esco_lang}_by_isco08.csv".format(
                esco_lang=self.esco_language
            ),
        )

        # dict storing data across all agg levels
        target_fpath_pkl = os.path.join(
            self.data_interim,
            "esco",
            self.esco_version,
            "occ_metadata_{esco_lang}_by_isco08.pkl".format(
                esco_lang=self.esco_language
            ),
        )

        # TODO: add ONET-based GBN data
        agg_funcs = list(self.agg_func_dict.values())
        agg_dict = {
            "share_green_gtp": agg_funcs,
            "n_occ_esco": np.sum,
        }
        for skill_type in self.skill_types_gbn:
            agg_dict[self.gbn_share_fmt.format(type=skill_type, ds="esco")] = agg_funcs

        list_of_occ_data = []
        for n_digits in isco08_digits:
            group_var_1 = self.fmt_string_isco_lvl.format(n_digits)
            group_var_2 = self.fmt_string_isco_label.format(n_digits)

            # aggregate
            if not use_weights:
                occ_grouped = self.occupation_metadata.groupby(
                    [group_var_1, group_var_2]
                ).agg(agg_dict)
            else:
                raise NotImplementedError("TODO: implement weighted aggregation")

            # rename cols
            occ_grouped.columns = [
                "_".join(col).replace("nan", "") for col in occ_grouped.columns
            ]

            # reset index
            occ_grouped = occ_grouped.reset_index()

            # fill nans in GTP columns with zeros
            gtp_cols = occ_grouped.columns[
                occ_grouped.columns.str.startswith("share_green_gtp")
            ]
            occ_grouped[gtp_cols] = occ_grouped[gtp_cols].fillna(0)

            # rename
            occ_grouped[self.isco_join_col_of] = occ_grouped[group_var_1]
            # occ_grouped = occ_grouped.rename(columns={group_var: "ISCO"})

            # append to dict
            list_of_occ_data.append(occ_grouped)

        df_out = pd.concat(list_of_occ_data, axis=0).reset_index(drop=True)

        df_out.to_csv(target_fpath_csv)
        df_out.to_pickle(target_fpath_pkl)

        # if not os.path.exists(target_fpath_pkl):
        #     pd.to_pickle(obj=df_out, filepath_or_buffer=target_fpath_pkl)
        # else:
        #     df_out = pd.read_pickle(target_fpath_pkl)

        return df_out

    # todo: implement reading weighted 3D shares
    def read_gbn_classification(self, agg_to_isco_at_digit=None, version="unweighted"):

        # ESCO-level
        if agg_to_isco_at_digit is None and version == "unweighted":
            # read final lists
            df_sl = pd.read_csv(
                os.path.join(
                    self.data_processed,
                    "esco",
                    "esco_level_gbn_classification_short_lists.csv",
                ),
                index_col=0,
            )

            df_sl_tobi = pd.read_csv(
                os.path.join(
                    self.data_processed,
                    "esco",
                    "esco_level_gbn_classification_short_lists_tobi.csv",
                ),
                index_col=0,
            )

            # get esco-isco mapping
            esco_to_isco = self.attach_isco_to_occupations(self.occupations)

            # merge mapping to lists
            df_sl_merged = df_sl.merge(
                esco_to_isco, on="conceptUri", how="left"
            ).rename(
                columns={
                    "gbn_classification_short_list": "classification",
                    "preferredLabel_x": "preferredLabel",
                }
            )
            df_sl_tobi_merged = df_sl_tobi.merge(
                esco_to_isco, on="conceptUri", how="left"
            ).rename(columns={"gbn_classification_short_list": "classification"})

            # combine both list version
            gbn_classification_merged = pd.merge(
                df_sl_merged[
                    [
                        "conceptUri",
                        "preferredLabel",
                        "classification",
                        "isco_level_4",
                        "isco_level_3",
                        "isco_level_2",
                        "isco_level_1",
                    ]
                ],
                df_sl_tobi_merged[["conceptUri", "classification"]],
                on="conceptUri",
                how="left",
                suffixes=("_sl", "_slt"),
            )

            return gbn_classification_merged.drop_duplicates().reset_index(drop=True)

        # ISCO-level
        elif agg_to_isco_at_digit is not None and version == "unweighted":
            gbn_shares_no_wt = pd.read_pickle(
                os.path.join(
                    self.data_processed,
                    "esco",
                    "final_gbn_shares_by_isco_unweighted.pkl",
                )
            )

            # select specified ISCO level
            gbn_shares_no_wt_sl = gbn_shares_no_wt.query(
                "isco_level == {} & list_version == 'short_list'".format(
                    agg_to_isco_at_digit
                )
            )
            gbn_shares_no_wt_slt = gbn_shares_no_wt.query(
                "isco_level == {} & list_version == 'short_list_tobi'".format(
                    agg_to_isco_at_digit
                )
            )

            # combine both list version
            gbn_shares_no_wt_isco = pd.merge(
                gbn_shares_no_wt_sl[
                    [
                        "isco_code",
                        "preferredLabel_isco",
                        "share_green",
                        "share_brown",
                        "share_neutral",
                    ]
                ],
                gbn_shares_no_wt_slt[["isco_code", "share_brown", "share_neutral"]],
                on="isco_code",
                how="left",
                suffixes=("_sl", "_slt"),
            )
            return gbn_shares_no_wt_isco


class Classifications(UsefulPaths):
    def __init__(self):
        # inherit class storing useful paths
        UsefulPaths.__init__(self=self)

        self._nace = None
        self._nace_1d = None
        self._nuts_3035 = None
        self._nuts_4326 = None
        self._isco = None

    @property
    def nace(self):
        if self._nace is None:
            self._nace = pd.read_csv(
                os.path.join(
                    self.data_raw, "classifications", "NACE_REV2_20210618_112431.csv"
                )
            )
        return self._nace

    @property
    def nace_1d(self):
        if self._nace_1d is None:
            self._nace_1d = pd.read_csv(
                os.path.join(
                    self.data_raw, "classifications", "NACE_REV2_1d_section_codes.csv"
                ),
                delimiter=";",
            )
        return self._nace_1d

    def read_nace_at_level(self, level=1):
        return self.nace.loc[self.nace["Level"] == level]

    @property
    def nuts_3035(self):
        if self._nuts_3035 is None:
            self._nuts_3035 = gpd.read_file(
                os.path.join(self.data_raw, "geodata", "NUTS_RG_03M_2021_3035.shp")
            )
        return self._nuts_3035

    @property
    def nuts_4326(self):
        if self._nuts_4326 is None:
            self._nuts_4326 = gpd.read_file(
                os.path.join(self.data_raw, "geodata", "NUTS_RG_03M_2021_4326.shp")
            )
        return self._nuts_4326

    @property
    def isco(self):
        if self._isco is None:
            self._isco = pd.read_csv(
                os.path.join(
                    self.data_raw,
                    "esco",
                    "v1.1.0",
                    "ISCOGroups_en.csv",
                ),
                dtype={"code": "str"},
            )
        return self._isco


def classify_by_gbn(
    df,
    col_name_green,
    col_name_brown,
    col_name_neutral,
    col_name_clfc="skillClassification",
):

    df[col_name_neutral] = (df[col_name_green] == False) & (df[col_name_brown] == False)
    # sanity check
    ambiguous_cases = (df[col_name_brown] == True) & (df[col_name_green] == True)
    assert ambiguous_cases.sum() == 0

    if col_name_clfc is not None:
        # create skill classification column: green/brown/neutral
        gbn_classification = df[
            [col_name_green, col_name_brown, col_name_neutral]
        ].idxmax(axis=1)

        df[col_name_clfc] = gbn_classification.str.split(pat="_", expand=True).iloc[
            :, 1
        ]
    return df


if __name__ == "__main__":

    # ESCO
    esco = Esco()

    # combine skills metadata
    skill_var_sel = []
    smd = esco.combine_skills_metadata()
    # esco.combine_occupation_metadata(skills_metadata=smd)
