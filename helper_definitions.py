import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def plot_hist_of_arrays(list_of_arrays,
                        list_of_bins,
                        list_of_labels,
                        histtype='step',
                        xlabel='Enter X label!!',
                        ylabel='Counts',
                        xrange=None,
                        yrange=None,
                        title=None,
                        path_to_save_fig=False,
                        verbose=True
                        ):
    '''
    This function plots a histogram for a list of arrays
    :param list_of_arrays: input list of arrays to be plotted in a histogram
    :param list_of_bins: input list of bins to apply for the arrays for the histogram
    :param list_of_labels: input list of labels for the histograms
    :param histtype: step is the one we use here
    :param xlabel: string for label x axis of histogram
    :param ylabel: string for label y axis of histogram
    :param xrange: tuple for x range of histogram
    :param yrange: tuple for y range of histogram
    :param title: string for title of histogram
    :param path_to_save_fig: string for path to save histogram
    :param verbose:
    :return: bin content and bin edges for all input data
    '''
    counter = 0
    for array in list_of_arrays:
        if xrange:
            bin_content, bins_edges, _ = plt.hist(array, histtype=histtype, label=list_of_labels[counter],
                                                  bins=list_of_bins[counter], range=xrange)
        else:
            bin_content, bins_edges, _ = plt.hist(array, histtype=histtype, label=list_of_labels[counter],
                                                  bins=list_of_bins[counter])
        counter = counter + 1

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    if yrange:
        plt.ylim(yrange)
    if title:
        plt.title(title)
    if path_to_save_fig:
        plt.savefig(path_to_save_fig)
        if verbose:
            print(f'saving plot to {path_to_save_fig}')
    if verbose:
        plt.show()

    return bin_content, bins_edges

