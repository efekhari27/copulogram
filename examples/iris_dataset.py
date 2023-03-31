#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""

import seaborn as sns
import copulogram as cp

data = sns.load_dataset('iris')
copulogram = cp.Copulogram(data)
copulogram.draw(save_file="figures/iris1.jpg")
copulogram.draw(alpha=0.8, hue='species', kde_on_marginals=False, save_file="figures/iris2.jpg")
copulogram.draw(hue='species', quantile_contour_levels=[0.2, 0.4, 0.6, 0.8], save_file="figures/iris_contours.jpg")