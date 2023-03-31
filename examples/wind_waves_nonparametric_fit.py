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


df_processed = pd.read_csv('data/df_ANEMOC_1H.csv')
all_indexes = df_processed.index.values
saved_columns = ['θ_wind (deg)', 'U_hub (m/s)', 'θ_wave_new (deg)', 'Hs (m)', 'Tp (s)']
df = df_processed.loc[:, saved_columns]

# Monte Carlo sub-sample to make the plots and the fits faster
N = 10000
subsampled_indexes = np.random.choice(all_indexes, N, replace=False)
data = df.loc[subsampled_indexes, saved_columns]
new_columns = ["$\\theta_{wind}$", "$U$", "$\\theta_{wave}$", "$H_s$", "$T_p$"]
data.columns = new_columns
# Draw static copulogram on data
copulogram = cp.Copulogram(data, latex=True)
alpha = 0.1
marker = "."
copulogram.draw(color="C7", alpha=alpha, kde_on_marginals=False, save_file="figures/wind_waves.jpg", marker=marker)

# Fitting marginals with parametric and nonparametric methods
U_data = ot.Sample(data["$U$"].values.reshape(-1, 1))
U_dist = ot.WeibullMinFactory().build(U_data)

Hs_data = ot.Sample(data["$H_s$"].values.reshape(-1, 1))
Hs_dist = ot.WeibullMinFactory().build(Hs_data)

Tp_data = ot.Sample(data["$T_p$"].values.reshape(-1, 1))
kernel = ot.KernelSmoothing()
silverman_kw = kernel.computeSilvermanBandwidth(Tp_data[:2000])
Tp_kde = kernel.build(Tp_data[:2000], silverman_kw)

WaveDir_data = ot.Sample(data["$\\theta_{wave}$"].values.reshape(-1, 1))
kernel = ot.KernelSmoothing()
mixed_kw = kernel.computeMixedBandwidth(WaveDir_data[:1000])
WaveDir_kde = kernel.build(WaveDir_data[:2000], mixed_kw)
WaveDir_kde = ot.TruncatedDistribution(WaveDir_kde, 0., 365.)

WindDir_data = ot.Sample(data["$\\theta_{wind}$"].values.reshape(-1, 1))
kernel = ot.KernelSmoothing()
mixed_kw = kernel.computeMixedBandwidth(WindDir_data[:2000])
WindDir_kde = kernel.build(WindDir_data[:2000], mixed_kw)
WindDir_kde = ot.TruncatedDistribution(WindDir_kde, 0., 365.)

# Fit copula and generate Monte Carlo sample
m_ebc = 100
npm = cp.NonParametricModel(data)
fitted_distribution = npm.fit(marginals=[WindDir_kde, U_dist, WaveDir_kde, Hs_dist, Tp_kde],
                              m=m_ebc)
simulated_data = npm.generate_mc(N)

# Draw static copulogram on data
copulogram = cp.Copulogram(simulated_data, latex=True)
copulogram.draw(color="C3", alpha=alpha, kde_on_marginals=False, marker=marker, save_file=f"figures/wind_waves_simulated_{m_ebc}.jpg")
