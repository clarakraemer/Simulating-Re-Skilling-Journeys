# -*- coding: utf-8 -*-
import logging
from src.data.lfs import EulfsDs
from src.data.framework import Esco


# note: need to uncomment click commands for CLI usage
# @click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
# @click.argument("config_filepath", type=click.Path(exists=True))
def main(config_paths, config_data):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    preprocessed data (saved in ../interim).

    Parameters
    ----------
    config_paths : str
        Name of yml file containing path configurations.
    config_data : str
        Name of yml file containing data configurations.
    Returns
    -------
    None
    """
    logger = logging.getLogger(__name__)
    logger.info("Making interim data sets from raw data.")
    pass


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main(
        config_paths="paths_config.yml",
        config_data="data_config.yml",
    )
