import logging
from src.data import preprocess
from src.visualization import visualize
from src.data.lfs import EulfsDs
from src.data.framework import Esco
from src.visualization.visualize import EulfsVis

# CONFIGS
config_paths = "paths_config.yml"
config_data = "data_config.yml"
config_model = "model_config.yml"
config_vis = "vis_config.yml"

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # ---------------------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------------------

    # ESCO
    # esco = Esco(
    #     fn_config_path=config_paths,
    #     fn_config_data=config_data,
    # )
    #
    # esco.occupation_skills_matrix()
    # esco.occupation_similarity_matrix()
    # esco.skills_metadata()
    # esco.merged_occupation_metadata()
    # esco.aggregate_occ_data_by_isco()
    #
    # # EU-LFS
    # eulfs = EulfsDs(
    #     fn_config_path=config_paths,
    #     fn_config_data=config_data,
    # )
    #
    # eulfs.preprocess()

    # ---------------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------------

    # EU-LFS
    eulfs_visualiser = EulfsVis(
        fn_config_data=config_data,
        fn_config_path=config_paths,
        fn_vis_config=config_vis,
        year=2019,
    )

    eulfs_visualiser.create_maps(base_dir_name=Target)
    # eulfs_visualiser.create_occupation_barplots(n_occ="all")
    # eulfs_visualiser.create_occupation_barplots(n_occ=10)
    # eulfs_visualiser.create_occupation_barplots(n_occ=20)
    # eulfs_visualiser.create_industry_boxplots()
