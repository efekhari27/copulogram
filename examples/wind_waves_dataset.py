#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""

import pandas as pd 
import numpy as np
import openturns as ot
import copulogram as cp

# 
data = pd.read_csv("data/wind_waves_ANEMOC_1H.csv", index_col=0)
data = data.iloc[:1000]

# Draw static copulogram on data
data.columns = ["$\\theta_{wind}$", "$U$", "$\\theta_{wave}$", "$H_s$", "$T_p$"]
copulogram = cp.Copulogram(data, latex=True)
alpha = 0.3
marker = "."
copulogram.draw(color="C7", alpha=alpha, kde_on_marginals=False, save_file="figures/wind_waves.jpg", marker=marker)
copulogram.draw(color="C7", kde_on_marginals=False, quantile_contour_levels=[0.25, 0.5, 0.75], save_file="figures/wind_waves_contours.jpg")


# Fake output
output = data["$U$"] ** 3 * ((np.pi / 180) * data["$\\theta_{wind}$"]) + (data["$H_s$"] ** 2 * data["$T_p$"]) / ((np.pi / 180) * data["$\\theta_{wave}$"])  
data['output'] = np.log10(output)
data.columns = ["$\\theta_{wind}$", "$U$", "$\\theta_{wave}$", "$H_s$", "$T_p$", "output"]
# Draw static copulogram on data
copulogram = cp.Copulogram(data, latex=True)
copulogram.draw(color="C7", hue="output", alpha=alpha, kde_on_marginals=False, save_file="figures/wind_waves_woutput.jpg", marker="o")
# Draw interactive copulogram on data
copulogram.draw_interactive(color="C7", hue="output", alpha=0.8)

