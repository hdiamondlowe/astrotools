from scipy.stats import binned_statistic
import numpy as np
import astropy.units as u


"""
Little tools that may come in handy.
"""

def bin_timeseries(t, y, yerr, bin_size=None, num_bin=None):
    """
    Bin time-series data and propagate per-point uncertainties.

    Parameters
    ----------
    t : array-like or astropy.units.Quantity
        Time values. If unitless, assumed to be in phase.
    y : array-like
        Flux (or other measurement) values.
    yerr : array-like
        1-sigma uncertainty per point (same shape as y).
    bin_size : float or astropy.units.Quantity, optional
        Bin width. Must have same unit as t (or both unitless).
    num_bin : int, optional
        Number of bins. Cannot be used with bin_size.

    Returns
    -------
    t_bin : ndarray or Quantity
        Bin centers.
    y_bin : ndarray
        Binned mean values.
    yerr_bin : ndarray
        Uncertainty on binned mean: sqrt(sum(err_i^2)) / N.
    n_bin : ndarray
        Number of points per bin.
    """
    if (bin_size is None and num_bin is None) or (bin_size is not None and num_bin is not None):
        raise ValueError("Must specify either bin_size or num_bin")

    # Extract unit information
    t_has_unit = hasattr(t, "unit")
    bin_size_has_unit = hasattr(bin_size, "unit") if bin_size is not None else False

    if t_has_unit and bin_size is not None and not bin_size_has_unit:
        raise ValueError("t has a unit but bin_size does not")
    if not t_has_unit and bin_size is not None and bin_size_has_unit:
        raise ValueError("t is unitless but bin_size has a unit")

    # Convert to unitless values
    if t_has_unit:
        t_vals = t.to_value(t.unit)
        unit_out = t.unit
    else:
        t_vals = np.asarray(t, dtype=float)
        unit_out = None

    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    tmin, tmax = np.nanmin(t_vals), np.nanmax(t_vals)

    # Build bin edges
    if bin_size is not None:
        if bin_size_has_unit:
            bin_day = float(bin_size.to_value(t.unit))
        else:
            bin_day = float(bin_size)
        edges = np.arange(tmin, tmax + bin_day, bin_day)
        if edges[-1] < tmax:
            edges = np.append(edges, edges[-1] + bin_day)
    else:
        edges = np.linspace(tmin, tmax, num_bin + 1)

    # Bin stats
    y_bin, _, _ = binned_statistic(t_vals, y, statistic="mean", bins=edges)
    n_bin, _, _ = binned_statistic(t_vals, y, statistic="count", bins=edges)
    mean_var_bin, _, _ = binned_statistic(t_vals, yerr**2, statistic="mean", bins=edges)

    # Error on binned mean: sqrt(mean(err^2)/N)
    with np.errstate(invalid="ignore", divide="ignore"):
        yerr_bin = np.sqrt(mean_var_bin / n_bin)

    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = n_bin > 0

    centers = centers[valid]
    y_bin = y_bin[valid]
    yerr_bin = yerr_bin[valid]
    n_bin = n_bin[valid]

    if unit_out is not None:
        centers = centers * unit_out

    return centers, y_bin, yerr_bin, n_bin