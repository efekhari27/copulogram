#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""
import numpy as np
import pandas as pd
import copulogram as cp

# Modified Ishigami input-output sample
size = 1000
X = np.random.standard_normal((size, 3))
Y = np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])
XY = np.hstack((X, Y.reshape(-1, 1)))
data = pd.DataFrame(XY, columns=['$X_1$', '$X_2$', '$X_3$', 'Y'])

copulogram = cp.Copulogram(data)
copulogram.draw(color='C7', marker='.', alpha=0.5, save_file="figures/ishigami.jpg")

# Adding a binary class depending on the output
import seaborn as sns
from matplotlib.colors import to_rgba
data["Failed"] = "True"
data.loc[(data["Y"] > 7.), "Failed"] = "False"

color_palette = sns.color_palette([to_rgba('C2', 0.5), to_rgba('C3', 0.9)], as_cmap=True)
copulogram = cp.Copulogram(data)
copulogram.draw(hue="Failed", marker='.', hue_colorbar=color_palette, save_file="figures/ishigami_threshold.jpg")