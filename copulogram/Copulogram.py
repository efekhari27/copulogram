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
            hue_palette=None,
            kde_on_marginals=True,
            quantile_contour_levels=None,
            save_file=None,
            marker='o',
            subplot_size=2.5
            ):
        """
        Draws the copulogram plot with a static or interactive option. 

        Parameters
        ----------
        color : string
            The matplotlib color on every element of the graph as long as "hue" is None.
        alpha : float
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        hue : string
            Grouping variable that will produce points with different colors. 
            Can be either categorical or numeric, although color mapping will behave differently in latter case.
        hue_palette : string
            Method for choosing the colors to use when mapping the hue semantic. 
            By default "tab10" for categorical mapping, and "viridis" for continuous mapping.
        kde_on_marginals : boolean
            Defines the type of plot on the diagonal. Histogram when 
            the variable is set to False, kernel density estimation otherwise.
        quantile_contour_levels : 1-d list of floats
            When the variable takes a value, the contours of the quantiles 
            defined by the variable are plotted.
        save_file : string 
            When this variable is not None, it saves the plot in the current repertory.
        marker : string
            Defines the scatterplots markers according to Matplotlib formalism.
        subplot_size : float
            Defines the size of each subplot in inches.
        
        Returns
        -------
        copulogram : matplotlib.axes.Axes
            The matplotlib axes containing the plot.
        """
        df = self.data.copy(deep=True)
        df_numeric = df._get_numeric_data()
        plotted_cols = np.array(df_numeric.columns)
        plotted_cols = np.delete(plotted_cols, np.where(plotted_cols == hue)).tolist()
        if hue is None:
            copulogram = sns.PairGrid(df_numeric, height=subplot_size)
            if kde_on_marginals:
                copulogram.map_diag(sns.kdeplot, hue=None, color=color)
            else:
                copulogram.map_diag(sns.histplot, hue=None, color=color, bins=20)

            if quantile_contour_levels is None:
                copulogram.map_lower(plt.scatter, color=color, alpha=alpha, marker=marker)
                temp = df_numeric.rank() / self.N * df_numeric.max().values
                copulogram.data = temp
                copulogram = copulogram.map_upper(plt.scatter, color=color, alpha=alpha, marker=marker)
            else:
                copulogram.map_lower(sns.kdeplot, levels=quantile_contour_levels, color=color)
                temp = df_numeric.rank() / self.N * df_numeric.max().values
                copulogram.data = temp
                copulogram.map_upper(sns.kdeplot, levels=quantile_contour_levels, color=color)
        else : 
            if hue_palette is None:
                if df[hue].dtype =='O':
                    hue_palette = 'tab10'
                else:
                    hue_palette='viridis'
            copulogram = sns.PairGrid(df[plotted_cols + [hue]], hue=hue, palette=hue_palette, height=subplot_size)
            if kde_on_marginals:
                copulogram.map_diag(sns.kdeplot, hue=None, color=".3")
            else:
                copulogram.map_diag(sns.histplot, hue=None, color=".3", bins=20)
            
            if quantile_contour_levels is None:
                copulogram.map_lower(sns.scatterplot, alpha=alpha, marker=marker)
                temp = df_numeric[plotted_cols].rank() / self.N * df_numeric[plotted_cols].max().values
                temp[hue] = df[hue]
                copulogram.data = temp
                copulogram = copulogram.map_upper(sns.scatterplot, alpha=alpha, marker=marker)
            else:
                copulogram.map_lower(sns.kdeplot, levels=quantile_contour_levels)
                temp = df_numeric[plotted_cols].rank() / self.N * df_numeric[plotted_cols].max().values
                temp[hue] = df[hue]
                copulogram.data = temp
                copulogram = copulogram.map_upper(sns.kdeplot, levels=quantile_contour_levels)

            copulogram.add_legend(title=hue)

        if save_file is not None:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
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

## TODO
# Include contours
# Add interactive aspect
# 