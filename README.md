# tg_analysis

Prototype Python-scripts made for "Evolution of sea-level trends along the Norwegian coast from 1960 to 2100".
Author: Kristian Breili

This repository includes the scripts that have been used to compute the results in the paper "Evolution of sea-level trends along the Norwegian coast from 1960 to 2100". Tide gauge records, ERA5 reanalyses and the NAO-index can be accessed at the Permanent Service for Mean Sea Level (www.psmsl.org, last access: September 2021), the Climate Data Store of the Copernicus Climate Change Service (https://cds.climate.copernicus.eu/cdsapp#!/home, last access: September 2021), and the National Centers for Environmental Information of the National Oceanic and Atmospheric Administration (https://www.ncdc.noaa.gov/teleconnections/nao/, last access: October 2021), respectively. The GIA-model NKG2016LU can be downloaded from Lantmäteriet.se (https://www.lantmateriet.se/en/maps-and-geographic-information/gps-geodesi-och-swepos/Referenssystem/Landhojning/, last access: September 2021).

In order to run the script tg_ssa_mcap.py, the SSA-class available at www.kaggle.com must be stored in a separate file named SSA_class.py.

setup.txt
File with setup (paths to tide-gauge files) and auxiliary parameters.

tg_ssa_mcap.py
Script for singular spectrum analysis of tide gauge records

psmsl_rate_and_plot
Script that calculate the sea-level rate and make a time series plot of a tide gauge record. 

ssa_class.py
Python class for singular spectrum analysis of time series

interpolate_nkg2016lu.py
Script for computing vertical land motion from the NKG2016LU VLM-model.
