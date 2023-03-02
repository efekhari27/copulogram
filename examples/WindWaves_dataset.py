#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""

import pandas as pd 
import numpy as np
import copulogram as cp


df_processed = pd.read_csv('data/df_ANEMOC_1H.csv')
all_indexes = df_processed.index.values
saved_colomns = ['θ_wind (deg)', 'U_hub (m/s)', 'θ_wave_new (deg)', 'Hs (m)', 'Tp (s)']
df = df_processed.loc[:, saved_colomns]
new_columns = ["$\\theta_{wind}$", "$U$", "$\\theta_{wave}$", "$H_s$", "$T_p$"]
df.columns = new_columns

# Monte Carlo sub-sample to make the plots and the fits faster
N = 1000
subsampled_indexes = np.random.choice(all_indexes, N, replace=False)
data = df.loc[subsampled_indexes, new_columns]
# Draw static copulogram
copulogram = cp.Copulogram(data, latex=True)
copulogram.draw(color="C1", alpha=0.1, save_file="figures/wind_waves.jpg")