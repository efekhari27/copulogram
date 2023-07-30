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
from matplotlib import rc
import matplotlib.pyplot as plt

# Interactive imports
from itertools import product
from bokeh.layouts import gridplot
from bokeh.models import (BasicTicker, Circle, ColumnDataSource,
                          DataRange1d, Grid, LassoSelectTool, LinearAxis,
                          Plot, ResetTool)
from bokeh.transform import factor_cmap, linear_cmap


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
                copulogram.map_diag(sns.kdeplot, color=color)
            else:
                copulogram.map_diag(sns.histplot, color=color, bins=20)

            if quantile_contour_levels is None:
                copulogram.map_lower(plt.scatter, color=color, alpha=alpha, marker=marker)
                temp = (df_numeric.rank() / self.N)
                xmaxs = df_numeric.max().values
                xmins = df_numeric.min().values
                temp = temp * (xmaxs - xmins) + xmins
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
                if df[hue].dtype =='O':
                    copulogram.map_diag(sns.kdeplot, color=".3")
                else:
                    copulogram.map_diag(sns.kdeplot, hue=None, color=".3")
            else:
                if df[hue].dtype =='O':
                    copulogram.map_diag(sns.histplot, color=".3", bins=20)
                else:
                    copulogram.map_diag(sns.histplot, hue=None, color=".3", bins=20)
            
            if quantile_contour_levels is None:
                copulogram.map_lower(sns.scatterplot, alpha=alpha, marker=marker)
                #temp = df_numeric[plotted_cols].rank() / self.N * df_numeric[plotted_cols].max().values
                temp = (df_numeric[plotted_cols].rank() / self.N)
                xmaxs = df_numeric[plotted_cols].max().values
                xmins = df_numeric[plotted_cols].min().values
                temp = temp * (xmaxs - xmins) + xmins
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
            color='navy',
            alpha=1.,
            hue=None,
            hue_palette=None,
            marker='o',
            subplot_size=5
            ):
        """
        Draws the copulogram plot with a static or interactive option. 

        Parameters
        ----------
        TBD
        
        Returns
        -------
        copulogram : TBD
        """

        df = self.data.copy(deep=True)
        df_numeric = df._get_numeric_data()
        rdf = df_numeric.rank()

        plotted_cols = np.array(df_numeric.columns)
        plotted_cols = np.delete(plotted_cols, np.where(plotted_cols == hue)).tolist()
        dim = len(plotted_cols)

        if hue is not None: 
            df_numeric[hue] = df[hue]
            rdf[hue] = df[hue]
        source = ColumnDataSource(data=df_numeric)
        rsource = ColumnDataSource(data=rdf)

        plot_list = []
        for i, (y, x) in enumerate(product(plotted_cols, plotted_cols)):
            # Scatter plot
            if hue is None: 
                scatter_color = color
            else: 
                if df[hue].dtype =='O':
                    if hue_palette is None:
                        hue_palette = "Category10_3"
                    scatter_color = factor_cmap(hue, hue_palette, sorted(df[hue].unique()))
                else:
                    if hue_palette is None:
                        hue_palette="Spectral6"
                    scatter_color = linear_cmap(hue, hue_palette, low=df[hue].min(), high=df[hue].max())
                        
            
            circle = Circle(x=x, y=y, fill_alpha=alpha, size=5, line_color=None,
                            fill_color=scatter_color)
            # Lower triangle
            if (i%dim) <= (i//dim): # Column index smaller than row index (i.e., lower triangle)
                # Define one empty plot
                p = Plot(x_range=DataRange1d(bounds=(df[x].min(), df[x].max())), y_range=DataRange1d(bounds=(df[y].min(), df[y].max())),
                        background_fill_color="#fafafa",
                        border_fill_color="white", width=200, height=200, min_border=subplot_size)
                r = p.add_glyph(source, circle)
                # Delete diagonal plot
                if (i%dim) == (i//dim):
                    r.visible = False
                    p.grid.grid_line_color = None
            # Upper triangle
            elif (i%dim) > (i//dim): 
                # Define one empty plot
                p = Plot(x_range=DataRange1d(bounds=(rdf[x].min(), rdf[x].max())), y_range=DataRange1d(bounds=(rdf[y].min(), rdf[y].max())),
                        background_fill_color="#fafafa",
                        border_fill_color="white", width=200, height=200, min_border=5)
                r = p.add_glyph(rsource, circle)
            p.x_range.renderers.append(r)
            p.y_range.renderers.append(r)
            # First column ticks
            if i % dim == 0:
                p.min_border_left = p.min_border + 4
                p.width += 40
                yaxis = LinearAxis(axis_label=y)
                yaxis.major_label_orientation = "vertical"
                p.add_layout(yaxis, "left")
                yticker = yaxis.ticker
            else:
                yticker = BasicTicker()
            p.add_layout(Grid(dimension=1, ticker=yticker))

            # Last row ticks
            if i >= dim * (dim-1):
                p.min_border_bottom = p.min_border + 40
                p.height += 40
                xaxis = LinearAxis(axis_label=x)
                p.add_layout(xaxis, "below")
                xticker = xaxis.ticker
            else:
                xticker = BasicTicker()
            p.add_layout(Grid(dimension=0, ticker=xticker))
            p.add_tools(LassoSelectTool(), ResetTool())
            plot_list.append(p)

        grid_plot = gridplot(plot_list, ncols=dim)
        return grid_plot

##TODO:
# Add docstrings
# Remove the misleading yticks from the top left plot? Ideally we should add the index ticks on the top left of the plot
# Add color bar on the interactive method using : https://docs.bokeh.org/en/latest/docs/examples/basic/data/color_mappers.html


#%%
if __name__ == "__main__":
    #data = sns.load_dataset('iris')
    import pandas as pd
    data = pd.read_csv("../examples/data/wind_waves_ANEMOC_1H.csv", index_col=0)
    data = data.iloc[:1000]
    
    output = data["U_hub (m/s)"] ** 3 * ((np.pi / 180) * data["θ_wind (deg)"]) + (data["Hs (m)"] ** 2 * data["Tp (s)"]) / ((np.pi / 180) * data["θ_wave_new (deg)"])  
    data['output'] = np.log10(output)

    copulogram = Copulogram(data)
    copulogram.draw_interactive(hue="output")
# %%
