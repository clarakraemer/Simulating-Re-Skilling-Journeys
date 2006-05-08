# Reskilling for Europe's Decarbonization 

This study examines how the design of reskilling programs can facilitate job switches during Europe’s decarbonization. To do so, we develop a data-driven simulation of reskilling journeys that compares reskilling programs focused on green, digital, transferable, and tailored skills. 

The main file for running the simulation is "reskilling.py" (src, modelling).
Necessary configurations are set-up and placed in "configs" and "src". 
Functions based on Nesta et al. (2020) are downloaded as "mapping_career_causways" (src).

To replicate our study, researchers need to complete the following steps:
1. Register for EU-LFS data at Eurostat and download the publicly available ESCO version 1.10.
2. Obtain the occupational classification from Zaussinger et al. (2025)
3. Place the data in one folder under the project root, based on the file structure specified in "paths_config.yml" (configs).
4. Run notebooks 01-05 for data preparation.
5. Run the main simulation file "reskilling.py" (src, modelling)
6. Run notebook 06 for result visualization. 

We use additional and publicly available metadata from Eurostat, including names of NUTS-2 regions and EU-SILC as specified in the configs. 