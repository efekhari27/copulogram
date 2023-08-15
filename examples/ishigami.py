#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""
import numpy as np
import pandas as pd
import openturns as ot 
import copulogram as cp
from openturns.usecases import ishigami_function

im = ishigami_function.IshigamiModel()
size = 5000
X = im.distributionX.getSample(size)
Y = im.model(X)

plotting_size = 1000
data = pd.DataFrame(np.array(X[:plotting_size]), columns=list(X.getDescription()))
data['Y'] = np.array(Y[:plotting_size])
copulogram = cp.Copulogram(data)
copulogram.draw(color='C7', marker='.', alpha=0.5, save_file="figures/ishigami.jpg")