import numpy as np
from matplotlib import pyplot as plt


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

    :param list_of_arrays:
    :param list_of_bins:
    :param list_of_labels:
    :param histtype: step is the one we use here
    :param xlabel:
    :param ylabel:
    :param xrange:
    :param yrange:
    :param title:
    :param path_to_save_fig:
    :param verbose:
    :return:
    '''
    counter = 0
    for array in list_of_arrays:
        if xrange:
            bin_content, bins_edges, _ = plt.hist(array, histtype=histtype, label=list_of_labels[counter], bins=list_of_bins[counter], range=xrange)
        else:
            bin_content, bins_edges, _ = plt.hist(array, histtype=histtype, label=list_of_labels[counter], bins=list_of_bins[counter])

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

