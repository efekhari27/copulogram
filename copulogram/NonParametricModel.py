#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""

import numpy as np
import pandas as pd
import openturns as ot

class NonParametricModel:
    """
    Non-parametric method to fit a multivariate distribution and generate i.i.d samples from it.
    
    Parameters
    ----------
    data : pd.Dataframe()
        Input dataset to be plotted. Must be a pandas DataFrame object. 
        A preprocessing removing every missing data is applied.

    Example
    --------
    >>> TBD
    """

    def __init__(self, data):
        self.data = data.dropna().astype(float)
        #check types
        self.N_data = data.shape[0]
        self.d = data.shape[1]
        self.distribution = None

    def fit(self, marginals=None, m=None, kw_list=None, truncation_array=None):

        if marginals is None:
            n_kde = 1000 # limits the data needed for the marginal fitting
            if kw_list is not None:
                if len(kw_list) != self.d:
                    raise ValueError(f"kw_list dimension:{len(kw_list)} different than the data dimension:{self.d}")
                marginals = [ot.KernelSmoothing().build(self.data.iloc[:n_kde, [i]], kw_list[i]) for i in range(self.d)]
            else: 
                marginals = [ot.KernelSmoothing().build(self.data.iloc[:n_kde, [i]], kw_list[i]) for i in range(self.d)]
        
        if truncation_array is not None: 
            for i in truncation_array[:, 0]:
                marginals[i] = ot.TruncatedDistribution(marginals[i], truncation_array[i, 1], truncation_array[i, 2])
        
        if m is None: 
            # Default value minimizing the MISE (see Lasserre thesis (2022))
            m = 1 + self.N_data ** (2 / (self.d + 4))
        fitted_copula =  ot.BernsteinCopulaFactory().build(ot.Sample(self.data.values), m)
        self.distribution = ot.ComposedDistribution(marginals, fitted_copula)
        return self.distribution
    
    def generate_mc(self, N):
        if self.distribution is None :
            raise ValueError("Please fit a nonparametric model before generating samples")
        simulated_data = np.array(self.distribution.getSample(N))
        return pd.DataFrame(simulated_data, columns=self.data.columns)
    
    def generate_qmc(self, N):
        if self.distribution is None :
            raise ValueError("Please fit a nonparametric model before generating samples")
        sobol_experiment = ot.LowDiscrepancyExperiment(ot.SobolSequence(self.d), self.distribution, N, False)
        simulated_data = np.array(sobol_experiment.generate(N))
        return pd.DataFrame(simulated_data, columns=self.data.columns)