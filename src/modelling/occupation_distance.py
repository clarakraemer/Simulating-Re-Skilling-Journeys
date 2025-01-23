import os
import numpy as np
import pandas as pd
from scipy.spatial import distance

from src import utils
from src.data.framework import Esco

esco = Esco()
useful_paths = utils.UsefulPaths()


# two occupations oA, oB: transition direction oA --> oB
def skill_shortage(oA, oB):
    """
    Compute skill shortage based on Nedelkoska et al. (2015) as the sum of skills that
    are possessed in greater intensity by oB than by oA weighted by skill importance or
    intensity.

    Can be interpreted as the human capital that is missing for a worker in oA to
    transition into oB. One advantage over comparable metrics is that this is an asymmetric
    occupation similarity measure. A disadvantage is that its calculation is
    computationally intensive for large numbers of transitions.

    Parameters
    ----------
    oA : np.array
        (weighted) Skill vector of source occupation.
    oB : np.array
        (weighted) Skill vector of target occupation.

    Returns
    -------
    skill_shortage: float
        Skill shortage for the occupation transition oA --> oB.
    """
    return np.sum(oB[oB > oA] - oA[oB > oA])


def skill_excess(oA, oB):
    """
    Compute skill excess based on Nedelkoska et al. (2015) as the sum of skills that
    are possessed in greater intensity by oA than by oB weighted by skill importance or
    intensity.

    Can be interpreted as the human capital that is lost when a worker in oA transitions
    into oB. One advantage over comparable metrics is that this is an asymmetric
    occupation similarity measure. A disadvantage is that its calculation is
    computationally intensive for large numbers of transitions.

    Parameters
    ----------
    oA : np.array
        (weighted) Skill vector of source occupation.
    oB : np.array
        (weighted) Skill vector of target occupation.

    Returns
    -------
    skill_excess: float
        Skill excess for the occupation transition oA --> oB.
    """
    return np.sum(oA[oA > oB] - oB[oA > oB])


def find_closest(i, similarity_matrix, df, best="max"):
    """
    Method for reporting the closest neighbours to a node i given a similarity matrix;
    useful during exploratory data analysis.

    Parameters
    ----------
    i (int OR None):
        Determines for which node where are assessing the closest neighbours;
        if i==None, a random node is chosen.
    similarity_matrix (numpy.ndarray):
        Similarity matrix determining the closeness between each pair of nodes.
    df (pandas.DataFrame):
        Dataframe to be used for reporting the closest neighbours; must have then
        same number of rows as the similarity matrix

    Returns
    -------
    df (pandas.DataFrame):
        The same input dataframe with an added column for similarity values
        between node i and the rest of the nodes, ordered in a descending order
        of similarity.

    """
    if type(i) == type(None):
        i = np.random.randint(similarity_matrix.shape[0])

    # increasing similarity values for COOC, decreasing values for shortage and excess
    if best == "max":
        most_similar = np.flip(np.argsort(similarity_matrix[i, :]))
        similarity = np.flip(np.sort(similarity_matrix[i, :]))
    elif best == "min":
        most_similar = np.argsort(similarity_matrix[i, :])
        similarity = np.sort(similarity_matrix[i, :])

    df = df.copy().loc[most_similar]
    df["similarity"] = similarity

    # drop source occ
    df = df.drop(index=i, axis=0)

    return df


def create_multiindex_for_esco_occs(occ):
    occ["esco_5_digit"] = occ["code"].str.slice(0, 6)
    occ["isco_4_digit"] = occ["iscoGroup"]
    occ["isco_3_digit"] = occ["iscoGroup"].astype(str).str.slice(0, 3)

    # Create multiindex
    arrays = [
        occ.conceptUri.values,
        occ.esco_5_digit.values,
        occ.isco_4_digit.values,
        occ.isco_3_digit.values,
    ]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(
        tuples, names=["concept_uri", "esco_5_digit", "isco_4_digit", "isco_3_digit"]
    )
    return index


def occ_sim_matrix_by_levels(
    occ_skills_mat=None,
    osm_version="weighted",
    sim_metric="cooc",
    diagonal_zeros=False,
    upskilling_ids=None,
):
    """

    Parameters
    ----------
    sim_metric
    osm_version
    diagonal_zeros

    Returns
    -------

    """

    # create multiindex for all granularity levels
    occ = esco.occupations
    index = create_multiindex_for_esco_occs(occ)

    if osm_version == "weighted" and sim_metric == "cooc":
        if occ_skills_mat is None:
            # read bipartite adjacency matrix for occupations and skills
            occ_skills_mat = esco.read_occ_skills_matrix(return_version=osm_version)

        # if upskilling ids are provided
        # note: currently only working for ISCO 3D level
        if upskilling_ids is not None:
            id_occ, idx_skill = upskilling_ids
            occ_skills_mat.index = index

            # find occ and skill ids of ESCO-level matrix
            id_skill = occ_skills_mat.columns.values[idx_skill]
            occ_subset = occ_skills_mat.index.get_level_values(3) == id_occ

            # designate as essential skill
            occ_skills_mat.loc[occ_subset, id_skill] = 1

        # calculate esco-level similarity
        occ_sim = np.dot(occ_skills_mat.values, occ_skills_mat.values.transpose())

    elif osm_version == "weighted" and sim_metric == "shortage":
        occ_sim = pd.read_pickle(
            os.path.join(
                useful_paths.data_processed,
                "esco",
                "occ_sim_matrix_weighted_skill_shortage.pkl",
            )
        ).values
    elif osm_version == "weighted" and sim_metric == "excess":
        occ_sim = pd.read_pickle(
            os.path.join(
                useful_paths.data_processed,
                "esco",
                "occ_sim_matrix_weighted_skill_excess.pkl",
            )
        ).values
    elif osm_version == "weighted" and sim_metric == "shortage_excess_avg":
        occ_sim = pd.read_pickle(
            os.path.join(
                useful_paths.data_processed,
                "esco",
                "occ_sim_matrix_weighted_skill_shortage_excess_avg.pkl",
            )
        ).values

    if diagonal_zeros:
        np.fill_diagonal(occ_sim, 0)

    # occ_sim.index = index
    df_occ_sim = pd.DataFrame(index=index, columns=index, data=occ_sim)
    return df_occ_sim


if __name__ == "__main__":
    from time import time

    start = time()

    # lambda functions for scipy cdist implementation
    func_skills_shortage = lambda A, B: np.sum(B[B > A] - A[B > A])
    func_skills_excess = lambda A, B: np.sum(A[A > B] - B[A > B])

    # load esco class
    esco = Esco()
    occ = esco.occupations.conceptUri.values

    # choose OSM version (weighted/unweighted)
    osm_version = "weighted"

    # read bipartite adjacency matrix for occupations and skills
    M = esco.read_occ_skills_matrix(return_version=osm_version)

    # compute shortage
    M_shortage = distance.cdist(XA=M, XB=M, metric=func_skills_shortage)
    df_shortage = pd.DataFrame(index=occ, columns=occ, data=M_shortage)
    df_shortage.to_pickle(
        os.path.join(
            useful_paths.data_processed,
            "esco",
            "occ_sim_matrix_weighted_skill_shortage.pkl",
        )
    )

    # compute excess
    M_excess = distance.cdist(XA=M, XB=M, metric=func_skills_excess)
    df_excess = pd.DataFrame(index=occ, columns=occ, data=M_excess)
    df_excess.to_pickle(
        os.path.join(
            useful_paths.data_processed,
            "esco",
            "occ_sim_matrix_weighted_skill_excess.pkl",
        )
    )

    # compute average
    df_avg = (df_shortage + df_excess) / 2
    df_avg.to_pickle(
        os.path.join(
            useful_paths.data_processed,
            "esco",
            "occ_sim_matrix_weighted_skill_shortage_excess_avg.pkl",
        )
    )

    end = time()

    print("time elapsed: {} min".format((end - start) / 60))
