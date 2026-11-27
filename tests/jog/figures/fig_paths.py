"""Input/output locations of the figure scripts (environment variables; defaults relative to the working directory).

JOG_FIG_OUT      output folder for the PDF/PNG figures                          (default: figures_out)
JOG_FIG_INPUTS   inputs prepared from the evaluation (metrics h5, npz, json)    (default: inputs)
JOG_RESULTS      results/ of the evaluation package (tests/jog/eval)             (default: results)
JOG_RESULTS_V1   results_v1/ of the evaluation package (rerun products)          (default: results_v1)
JOG_SHP          folder with Brahmaputra_3_shapefiles/ and DRB_3_shapefiles/     (default: <PYGLDA_V1_DATA>/basin/shp)
JOG_ESA_SING     published ESA-SING folder (shapefiles/, <basin>/gravity_signal_cov/)
"""
import os

OUT = os.environ.get('JOG_FIG_OUT', 'figures_out')
INPUTS = os.environ.get('JOG_FIG_INPUTS', 'inputs')
RESULTS = os.environ.get('JOG_RESULTS', 'results')
RESULTS_V1 = os.environ.get('JOG_RESULTS_V1', 'results_v1')
SHP = os.environ.get('JOG_SHP', os.path.join(os.environ.get('PYGLDA_V1_DATA', '/work/data_for_w3'), 'basin', 'shp'))
ESA_SING = os.environ.get('JOG_ESA_SING', 'ESA_SING')
os.makedirs(OUT, exist_ok=True)
