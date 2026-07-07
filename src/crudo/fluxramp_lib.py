# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from . import ipe_scipy


def phi_highpass(frequency, cutoff_frequency):
    """
    returns phase shift of frequency in a highpass with cutoff_frequency
    """
    return np.arctan(cutoff_frequency / frequency)


def gen_ramp_data(
    ramp_freq,
    rel_amplitude,
    highpass_cutoff_freq=4000,
    insertion_loss_correction_factor=1,
    fall_time_ns=100,
    sample_rate=500e6,
):
    """
    generates corrected ramp shape to write into Stimulation Module

    Note:
        Insertion Loss Correction not implemented
    """
    steps = 200
    f_adtt1 = np.array([10, 20, 40, 70, 100, 460, 20600, 30400, 50000, 110000])
    il_adtt1 = np.array([0.38, 0.22, 0.13, 0.11, 0.11, 0.10, 0.28, 0.37, 0.59, 1.59])
    # Insertion Loss from Data Sheet: https://www.minicircuits.com/pages/s-params/ADTT1-6_VIEW.pdf
    f_k = np.linspace(ramp_freq, steps * ramp_freq, steps)
    il_k = np.interp(f_k, f_adtt1 * 1000, il_adtt1)
    length = int(np.around(sample_rate / ramp_freq))
    fourier_series = np.zeros(length, dtype=np.float64)
    t = np.linspace(0, 1 / ramp_freq, length, endpoint=False)
    for k in range(1, steps + 1):
        fourier_series += (
            np.sin(
                2 * np.pi * ramp_freq * k * t
                - phi_highpass(k * ramp_freq, highpass_cutoff_freq)
            )
            / k
            * 10 ** (il_k[k - 1] * insertion_loss_correction_factor)
        )
    # spare out edges in fit (cutoff)
    fit_co = 50
    fit_par = np.polyfit(t[fit_co:-fit_co], fourier_series[fit_co:-fit_co], 3)
    polyfit = np.poly1d(fit_par)
    offset = -(np.amax(polyfit(t)) + np.amin(polyfit(t))) / 2
    pp_val = np.amax(polyfit(t)) - np.amin(polyfit(t))
    # maybe offset needs to be adjusted
    ramp = np.array((polyfit(t) + offset) * 2 * rel_amplitude / pp_val, dtype=float)
    # factor of 2 since output range is from -1 to 1

    edge_smpl = int(fall_time_ns / 2)  # default sample time is 2ns
    intersect = ramp[edge_smpl]
    end_val = ramp[-1]
    edge = np.linspace(end_val, intersect, edge_smpl)
    for i in range(edge_smpl):
        ramp[i] = edge[i]
    return ramp


def residual_phase(time, period):
    """
    returns residual phase of wave oscillating for time t with a periode of T
    """
    r = np.fmod(time, period) / period
    if r < 0.5:
        return 2 * np.pi * r
    return 2 * np.pi * (1 - r)


def get_best_fluxramp_length(c_demod, frequency_list, lmin=500, lmax=2000):
    """
    returns FluxRampLength with the least squares of residual phases inside the given limits.
    """
    frequencies = np.float64(frequency_list)
    samples_per_periode = c_demod.GetChannelSampleRate() / frequencies
    steps = len(frequencies)
    min_residual = (steps * 2 * np.pi) ** 2
    best_length = -1
    # brute force
    for index in range(lmin, lmax):
        residual = 0
        for p in samples_per_periode:
            res_phase = residual_phase(index, p)
            residual += res_phase**2
        if residual < min_residual:
            min_residual = residual
            best_length = index
    print(
        f"BestL = {best_length} with average deviation = {np.rad2deg(np.sqrt(min_residual / steps))}°"
    )
    for p in samples_per_periode:
        print(
            f"periodes = {best_length/p:.1f}, dev = {np.rad2deg(residual_phase(best_length, p))}°"
        )
    return best_length


def shifted_errorbars(axis, x_arr, y_arr, err, color, offset=0, label=None):
    """
    Shifts the errorbars by a specific offset
    """
    if err is None:
        err = [None] * y_arr.shape[-1]
    for index, y_value in enumerate(y_arr):
        if index != 0:
            label = None
        axis.errorbar(
            x_arr[index] + offset,
            y_value,
            yerr=err[index],
            marker="o",
            color=color,
            label=label,
        )
