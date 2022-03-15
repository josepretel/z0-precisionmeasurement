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
    counter = 0
    for array in list_of_arrays:
        if xrange:
            plt.hist(array, histtype=histtype, label=list_of_labels[counter], bins=list_of_bins[counter], range=xrange)
        else:
            plt.hist(array, histtype=histtype, label=list_of_labels[counter], bins=list_of_bins[counter])

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

