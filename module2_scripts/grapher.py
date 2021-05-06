
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import kurtosis, skew, linregress, pearsonr


class Grapher:
    
    """
    class for plotting different types of graphs with some standardized basic attributes
    """

    def __init__(self, xlabel, ylabel, title, df, ymin=None, ymax=None):        
        """
        Args:
            xlabel (str): Label for x-axis of plot
            xlabel (str): Lavel for y-axis of plot
            title (str): Title for graph
            df (pandas dataframe): dataframe to be plotted
            ymin (float): [Optional] parameter to set minimum y value of y axis
            ymax (float): [Optional] parameter to set maximum y value of y axis
        """
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.df = df
        self.ymin = ymin
        self.ymax = ymax
        return
    
    def static_attributes(self):        
        """
        set static attributes of all graphs that will be plotted using this class
        """
        plt.style.use('seaborn-darkgrid')
        plt.figure(figsize=(20,12))
        x = self.df.index
        y = {col: self.df[col].values.tolist() for col in self.df.columns.values.tolist()}
        plt.xlabel(self.xlabel, fontsize=20)
        plt.ylabel(self.ylabel, fontsize=18)
        plt.title(self.title, fontsize=25)
        plt.xticks(fontsize=13.5)
        plt.yticks(fontsize=13.5)
        plt.ticklabel_format(style='plain', useOffset=False)
        if self.ymin and self.ymax: plt.ylim(self.ymin, self.ymax)
        return plt, x, y
    
    @classmethod
    def plot_multiple_line_graph(
        cls, xlabel, ylabel, title, df, ymin=None, ymax=None
        ): 
        plt, x, y = cls(
            xlabel, ylabel, title, df, ymin, ymax
            ).static_attributes()
        for label, values in y.items():
            plt.plot(x, values, label=label, linewidth=1)
        plt.legend(bbox_to_anchor=(1.17, 0.85), fontsize=13)
        plt.axhline(y=0, color='black', linewidth=0.5)
        plt.show()
        
    @classmethod
    def plot_multiple_bar_graph(
        cls, xlabel, ylabel, title, df, ymin=None, ymax=None, barvalues=False, h=False
        ):
        plt, x, y = cls(
            xlabel, ylabel, title, df, ymin, ymax
            ).static_attributes()
        for label, values in y.items():
            if h:
                plt.barh(x, values, label=label, linewidth=1)
            else:
                plt.bar(x, values, label=label, linewidth=1)
            if barvalues:
                for i, v in enumerate(values):
                    if h:
                        plt.text(v * 1.005, i - 0.25, str(round(v,5)), {'size': 12})
                    else:
                        plt.text(i - 0.25, v * 1.005, str(round(v,5)), {'size': 12})
        plt.legend(bbox_to_anchor=(1.17, 0.85), fontsize=13)
        try:
            plt.xticks(np.arange(min(x), max(x)+1, 1.0))
        except:
            pass
        plt.show()

    @classmethod
    def plot_histogram(
        cls, xlabel, ylabel, title, df, bins=None, ymin=None, ymax=None
        ):
        plt, x, y = cls(
            xlabel, ylabel, title, df, ymin, ymax
            ).static_attributes()
        for values in y.values():
            plt.hist(values, bins)
        values = np.array(values).astype(np.float)
        min_ylim, max_ylim = plt.ylim()
        text_marker_dict = {
            'Mean': {
                'value': round(scipy.mean(values), 5), 
                'position': 0.95, 
                'linecolor': 'k'
                },
            'Median': {
                'value': round(scipy.median(values), 5), 
                'position': 0.92, 
                'linecolor': 'b'
                },
            'Skewness': {
                'value': round(skew(values), 5), 
                'position': 0.89
                },
            'Kurtosis': {
                'value': round(kurtosis(values)), 
                'position': 0.86
                },
            'StDev': {
                'value': round(np.std(values), 5),
                'position': 0.83
            }
        }
        for k, v in text_marker_dict.items():
            plt.text(
                max(values), 
                max_ylim * v['position'], 
                f"{k}: {v['value']}", 
                {'size': 14}
                )
            if k in ['Mean', 'Median']: 
                plt.axvline(
                    v['value'], 
                    color=v['linecolor'], 
                    linestyle='dashed', 
                    linewidth=1
                    )
        plt.xticks(np.arange(0, max(values)+1, 1.0))
        plt.show()


    @classmethod
    def plot_boxplot(
        cls, xlabel, ylabel, title, dct, df=pd.DataFrame(), bins=None, ymin=None, ymax=None
        ):
        plt, x, y = cls(
            xlabel, ylabel, title, df, ymin, ymax
            ).static_attributes()
        plt.boxplot(dct.values())
        plt.xticks([i+1 for i in range(len(dct))], dct.keys())
        plt.show()
