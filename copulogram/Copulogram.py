#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Elias Fekhari
"""

#%%
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc, ticker, rcParams, colors, cm
import matplotlib.pyplot as plt
from itertools import product

class Copulogram:
    """
    Draws a plot for multivariate distributions.
    The lower triangle is a matrixplot of the data (without transformation), 
    while the upper triangle is a matrixplot of the ranked data.

    Parameters
    ----------
    data : pd.Dataframe()
        Input dataset to be plotted. Must be a pandas DataFrame object. 
        A preprocessing removing every missing data is applied.

    Example
    --------
    >>> import seaborn as sns
    >>> import copulogram as cp
    
    >>> data = sns.load_dataset('iris')
    >>> copulogram = cp.Copulogram(data)
    >>> copulogram.draw()
    """
    def __init__(
        self, 
        data, 
        latex=False
        ):
        self.data = data.dropna()
        self.N = data.shape[0]
        self._bins = 15
        self._bw_method = 'silverman' # If set to scalar, it becomes the bandwidth
        if latex:
            rc('font', **{'family': 'Times'})
            rc('text', usetex=True)
            rc('font', size=18)# Set the default text font size
            rc('axes', titlesize=20)# Set the axes title font size
            rc('axes', labelsize=18)# Set the axes labels font size
            rc('xtick', labelsize=18)# Set the font size for x tick labels
            rc('ytick', labelsize=18)# Set the font size for y tick labels
            rc('legend', fontsize=18)# Set the legend font size
        rcParams["axes.formatter.limits"] = [-2, 2]


    def draw(self, 
            color='C0',
            alpha=None,
            hue=None,
            hue_colorbar=None,
            kde_on_marginals=False,
            pdf_on_marginals=False,
            quantile_contour_levels=None,
            save_file=None,
            marker='o',
            subplot_size=2.5,
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
        hue_colorbar : string
            Method for choosing the colors to use when mapping the hue semantic. 
            By default "tab10" for categorical mapping, and "viridis" for continuous mapping.
        kde_on_marginals : boolean
            Defines the type of plot on the diagonal. Histogram when 
            the variable is set to False, kernel density estimation otherwise.
        cdf_on_marginals: boolean
            If set to True, plots on the diagonal are cumulative distribution functions (CDFs)
            If set to False, plots on the diagonal are probability density functions (PDFs).
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
        df_full = self.data.copy(deep=True)
        df = df_full._get_numeric_data()
        rdf = df.rank() / self.N            
        if hue is None: 
            dim = df.shape[1]
            hue_diag = None
            hue_colorbar = None
            hue_colorbar_diag = None
        else: 
            # Default colors
            if hue_colorbar is None: 
                if (df_full[hue].dtype =='O') or (df_full[hue].unique().size < 10):
                    hue_colorbar = 'tab10'
                else:
                    hue_colorbar ='viridis'
            if df_full[hue].dtype !='O': # Numeric hue
                df = df.drop(columns=[hue])
                rdf = rdf.drop(columns=[hue])
                hue_diag = None
                hue_colorbar_diag = None
                color ='C7'
            else: 
                hue_diag = hue
                hue_colorbar_diag = hue_colorbar
                color = None
            dim = df.shape[1]
            df[hue] = df_full[hue]
            rdf[hue] = df_full[hue]
        xmins = df.min()
        xmaxs = df.max()
        copulogram, self._axs = plt.subplots(dim, dim, figsize=(dim * subplot_size, dim * subplot_size))
        for i, j in product(range(dim), range(dim)):
            ax = self._axs[i, j]
            # Diagonal #############
            if i == j:
                if kde_on_marginals:                 
                    sns.kdeplot(data=df, x=df.columns[j], 
                                        hue=hue_diag,
                                        ax=ax,
                                        alpha=alpha,
                                        color=color, 
                                        palette=hue_colorbar_diag,
                                        bw_method = self._bw_method,
                                        fill=True,
                                        cumulative=pdf_on_marginals,
                                        multiple="stack",
                                        legend=False,
                                        common_norm=False,
                                )
                    ax.set_xlim(xmins[i], xmaxs[i])
                else: 
                    sns.histplot(data=df, x=df.columns[j],  
                                        hue=hue_diag,
                                        ax=ax,
                                        alpha=alpha,
                                        color=color, 
                                        palette=hue_colorbar_diag,
                                        bins=self._bins,
                                        fill=True,
                                        cumulative=pdf_on_marginals,
                                        stat='density',
                                        multiple="stack",
                                        legend=False,
                                        common_norm=False, 
                                )
                # Ticker tuning 
                xticks = np.linspace(xmins[j], xmaxs[j], 4)
                ax.set_xticks(xticks)
                ax.set_xticklabels(ax.get_xticks(), rotation=90)
                ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3}"))
                # Axis tuning
                if i == 0 : 
                    ax.set_xlabel(None)
                    ax.xaxis.set_ticklabels([])
                    ax.set_ylabel(df.columns[i])
                elif i == dim-1: 
                    ax.set_ylabel(None)
                    ax.yaxis.set_ticks_position("right")
                else : 
                    ax.set_xlabel(None)
                    ax.set_ylabel(None)
                    ax.xaxis.set_ticklabels([])
                    ax.yaxis.set_ticklabels([])

            # Bottom triangle #############                
            elif i > j:
                if quantile_contour_levels is not None:     
                    sns.kdeplot(data=df, x=df.columns[j], y=df.columns[i], 
                            hue=hue,
                            ax=ax,
                            alpha=alpha,
                            color=color, 
                            levels=quantile_contour_levels,
                            palette=hue_colorbar,
                    )
                else: 
                    sns.scatterplot(data=df, x=df.columns[j], y=df.columns[i], 
                            hue=hue,
                            ax=ax,
                            alpha=alpha,
                            color=color,
                            marker=marker,
                            palette=hue_colorbar,
                            )
                # Ticker tuning 
                xticks = np.linspace(xmins[j], xmaxs[j], 4)
                yticks = np.linspace(xmins[i], xmaxs[i], 4)              
                ax.set_xticks(xticks)
                ax.set_yticks(yticks)
                ax.set_xticklabels(ax.get_xticks(), rotation=90)
                ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3}"))
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3}"))
                # Axis tunning
                if (0 < j):
                    ax.yaxis.set_ticklabels([])
                    ax.set_ylabel(None)
                if (i < dim-1) : 
                    ax.xaxis.set_ticklabels([])
                    ax.set_xlabel(None)
                if j == 0:
                    ax.set_ylabel(df.columns[i])
                if i == dim-1:
                    ax.set_xlabel(df.columns[j])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if hue is not None: 
                    if (i == 1):
                        if (df_full[hue].dtype =='O') or (df_full[hue].unique().size < 10): # Discrete hue
                            discrete_hue = True
                            sns.move_legend(ax, "center right", bbox_to_anchor=(dim * 1.5, 0.), title=hue)
                        else: 
                            discrete_hue = False
                            #outpath_colorbar = ax.get_children()[2]
                            copulogram.set_figwidth(dim * subplot_size + 2)
                            ax.legend_.remove()
                    else:
                        ax.legend_.remove()

            # Top triangle #############
            else: 
                if quantile_contour_levels is not None:     
                    sns.kdeplot(data=rdf, x=df.columns[j], y=df.columns[i], 
                            hue=hue,
                            ax=ax,
                            alpha=alpha,
                            color=color, 
                            levels=quantile_contour_levels,
                            palette=hue_colorbar,
                            legend=False
                    )                 
                else: 
                    sns.scatterplot(data=rdf, x=df.columns[j], y=df.columns[i], 
                            hue=hue,
                            ax=ax,
                            alpha=alpha,
                            color=color,
                            marker=marker,
                            palette=hue_colorbar,
                            legend=False
                            )
                # Axis tuning
                ax.xaxis.set_ticks_position("top")
                ax.yaxis.set_ticks_position("right")
                ax.set_xticks([0., 0.5, 1.0])
                ax.set_yticks([0., 0.5, 1.0])
                ax.set_xticklabels(ax.get_xticks(), rotation=90)
                if (j < dim-1):
                    ax.yaxis.set_ticklabels([])
                if (i > 0) : 
                    ax.xaxis.set_ticklabels([])
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.set_xlabel(None)
                ax.set_ylabel(None)
            ax.xaxis.set_tick_params(width=1.2)   
            ax.yaxis.set_tick_params(width=1.2)
        if (hue is not None) and (not discrete_hue):
            self._norm = colors.Normalize(vmin=df[hue].min(), vmax=df[hue].max())
            scalar_map = cm.ScalarMappable(norm=self._norm ,cmap=hue_colorbar)
            copulogram.colorbar(scalar_map, ax=self._axs.ravel().tolist(), aspect=50, shrink=0.5, label=hue)
        if save_file is not None:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
        return copulogram
    