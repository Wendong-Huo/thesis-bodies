# not succeed.

import warnings
import seaborn as sns
from seaborn import utils
import ray

ray.init(num_cpus=4)

@ray.remote(object_store_memory=200 * 1024)
def worker(self, row_i, col_j, hue_k, data_ijk, kw_color, kwargs, args, func_module, func):
    # If this subset is null, move on
    if not data_ijk.values.size:
        return

    # Get the current axis
    ax = self.facet_axis(row_i, col_j)

    # Decide what color to plot with
    kwargs["color"] = self._facet_color(hue_k, kw_color)

    # Insert the other hue aesthetics if appropriate
    for kw, val_list in self.hue_kws.items():
        kwargs[kw] = val_list[hue_k]

    # Insert a label in the keyword arguments for the legend
    if self._hue_var is not None:
        kwargs["label"] = utils.to_utf8(self.hue_names[hue_k])

    # Get the actual data we are going to plot with
    plot_data = data_ijk[list(args)]
    if self._dropna:
        plot_data = plot_data.dropna()
    plot_args = [v for k, v in plot_data.iteritems()]

    # Some matplotlib functions don't handle pandas objects correctly
    if func_module.startswith("matplotlib"):
        plot_args = [v.values for v in plot_args]

    # Draw the plot
    self._facet_plot(func, ax, plot_args, kwargs)

def map(self, func, *args, **kwargs):
    """Apply a plotting function to each facet's subset of the data.

    Parameters
    ----------
    func : callable
        A plotting function that takes data and keyword arguments. It
        must plot to the currently active matplotlib Axes and take a
        `color` keyword argument. If faceting on the `hue` dimension,
        it must also take a `label` keyword argument.
    args : strings
        Column names in self.data that identify variables with data to
        plot. The data for each variable is passed to `func` in the
        order the variables are specified in the call.
    kwargs : keyword arguments
        All keyword arguments are passed to the plotting function.

    Returns
    -------
    self : object
        Returns self.

    """
    # If color was a keyword argument, grab it here
    kw_color = kwargs.pop("color", None)

    if hasattr(func, "__module__"):
        func_module = str(func.__module__)
    else:
        func_module = ""

    # Check for categorical plots without order information
    if func_module == "seaborn.categorical":
        if "order" not in kwargs:
            warning = ("Using the {} function without specifying "
                        "`order` is likely to produce an incorrect "
                        "plot.".format(func.__name__))
            warnings.warn(warning)
        if len(args) == 3 and "hue_order" not in kwargs:
            warning = ("Using the {} function without specifying "
                        "`hue_order` is likely to produce an incorrect "
                        "plot.".format(func.__name__))
            warnings.warn(warning)

    # Iterate over the data subsets
    futures = []
    for (row_i, col_j, hue_k), data_ijk in self.facet_data():
         future = worker.remote(self, row_i, col_j, hue_k, data_ijk, kw_color, kwargs, args, func_module, func)
         futures.append(future)
    ray.get(futures)

    # Finalize the annotations and layout
    self._finalize_grid(args[:2])

    return self


