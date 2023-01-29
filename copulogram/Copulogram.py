#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt

class Copulogram:
    """
    Interactive plot for multivariate distributions.
    The lower triangle is a matrixplot of the data (without transformation), 
    while the upper triangle is a matrixplot of the ranked data.

    Parameters
    ----------
    data : pd.Dataframe()
        Input dataset to be plotted. Must be a pandas DataFrame object. 
        A preprocessing removing every missing data is applied.

    Example
    --------
    >>> TBD
    """
    def __init__(
        self, 
        data, 
        latex=False
        ):
        self.data = data.dropna()
        self.N = data.shape[0]
        if latex:
            rc('font', **{'family': 'Times'})
            rc('text', usetex=True)
            rc('font', size=18)# Set the default text font size
            rc('axes', titlesize=20)# Set the axes title font size
            rc('axes', labelsize=18)# Set the axes labels font size
            rc('xtick', labelsize=18)# Set the font size for x tick labels
            rc('ytick', labelsize=18)# Set the font size for y tick labels
            rc('legend', fontsize=18)# Set the legend font size

    def draw(self, 
            color='C0',
            alpha=1.,
            hue=None,
            hue_palette="viridis",
            kde_on_marginals=True,
            quantile_contour_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            ):
        """
        Draws the copulogram plot with a static or interactive option. 

        Parameters
        ----------
        kde_on_marginals : Bool
            Defines the type of plot on the diagonal. Histogram when 
            the variable is set to False, kernel density estimation otherwise.
        quantile_contour_levels : 1-d list of floats
            When the variable takes a value, the contours of the quantiles 
            defined by the variable are plotted.

        Returns
        -------
        copulogram : TBD
        """
        df = self.data.copy(deep=True)
        df_numeric = df._get_numeric_data()
        plotted_cols = np.array(df_numeric.columns)
        plotted_cols = np.delete(plotted_cols, np.where(plotted_cols == hue)).tolist()
        if hue is not None:
            copulogram = sns.PairGrid(df[plotted_cols + [hue]], hue=hue)
            if kde_on_marginals:
                copulogram.map_diag(sns.kdeplot, hue=None, color=".3")
            else:
                copulogram.map_diag(sns.histplot, hue=None, color=".3", bins=20)
            copulogram.map_lower(sns.scatterplot, palette=hue_palette, alpha=alpha)
            temp = df_numeric[plotted_cols].rank() / self.N * df_numeric[plotted_cols].max().values
            temp[hue] = df[hue]
            copulogram.data = temp
            copulogram = copulogram.map_upper(sns.scatterplot, palette=hue_palette, alpha=alpha)
        else : 
            copulogram = sns.PairGrid(df_numeric)
            if kde_on_marginals:
                copulogram.map_diag(sns.kdeplot, hue=None, color=color)
            else:
                copulogram.map_diag(sns.histplot, hue=None, color=color, bins=20)
            copulogram.map_lower(plt.scatter, color=color, alpha=alpha)
            temp = df_numeric.rank() / self.N * df_numeric.max().values
            copulogram.data = temp
            copulogram = copulogram.map_upper(plt.scatter, color=color, alpha=alpha)
        return copulogram

    def draw_interactive(self, 
            kde_on_marginals=True,
            quantile_contour_levels=None):
        """
        Draws the copulogram plot with a static or interactive option. 

        Parameters
        ----------
        kde_on_marginals : Bool
            Defines the type of plot on the diagonal. Histogram when 
            the variable is set to False, kernel density estimation otherwise.
        quantile_contour_levels : 1-d list of floats
            When the variable takes a value, the contours of the quantiles 
            defined by the variable are plotted.  

        Returns
        -------
        copulogram : TBD
        """

        return None