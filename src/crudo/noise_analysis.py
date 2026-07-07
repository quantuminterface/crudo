import numpy as np
import matplotlib.pyplot as plt
import pylpsd
from scipy import signal
from os import listdir
from os.path import isfile, join
from crudo.conversion_tools import ADC_to_dBm


def calculate_noise_welch(data, sample_rate):
    N = len(data) / 400  # 4096 * 128  # window USRP-like
    window = "blackman"  # BULLKID group uses "welch" window, does not exist?
    f, Pxx = signal.welch(data, sample_rate, window, N, scaling="density")
    return f, Pxx


def calculate_noise_lpsd(data, sample_rate):
    window = "blackman"  # BULLKID group uses "welch" window, does not exist?
    f, Pxx = pylpsd.lpsd(data, sample_rate, window=window)
    return f, Pxx


# Magnitude PSD [dBc/Hz]
def noise_magnitude(input_signal, sample_rate, method=calculate_noise_welch):
    magnitude = np.abs(input_signal)
    f_mag, Pxx_mag = method(magnitude, sample_rate)
    Pxx_mag = 10 * np.log10(
        Pxx_mag / np.mean(magnitude**2)
    )  # P[dBc] = 10*log10(P/P0) = 10*log10(V**2/V0**2)
    return f_mag, Pxx_mag


# Phase PSD [dBc/Hz]
def noise_phase(input_signal, sample_rate, method=calculate_noise_welch):
    phase = np.angle(input_signal)
    f_phase, Pxx_phase = method(phase, sample_rate)
    Pxx_phase = 10 * np.log10(Pxx_phase)
    return f_phase, Pxx_phase


# I PSD [dBm/Hz]
def noise_i(input_signal, sample_rate, conf_factor, method=calculate_noise_welch):
    i_volts = np.asarray([sample.real / conf_factor for sample in input_signal])
    f_i, Pxx_i = method(i_volts, sample_rate)
    Z = 50
    Pxx_i = 10 * np.log10(Pxx_i * 1000 / Z)  # P[dBm] = 10 * log10(V_rms**2 * 1000 / Z)
    return f_i, Pxx_i


# Q PSD [dBm/Hz]
def noise_q(input_signal, sample_rate, conf_factor, method=calculate_noise_welch):
    q_volts = np.asarray([sample.imag / conf_factor for sample in input_signal])
    f_q, Pxx_q = method(q_volts, sample_rate)
    Z = 50
    Pxx_q = 10 * np.log10(Pxx_q * 1000 / Z)  # P[dBm] = 10 * log10(V_rms**2 * 1000 / Z)
    return f_q, Pxx_q


def noise_analysis(data, sample_rate, adc_counts_per_v, method="welch", filepath=None):

    # Select Method
    if method == "welch":
        analysis_method = calculate_noise_welch
    elif method == "lpsd":
        analysis_method = calculate_noise_lpsd
    else:
        analysis_method = calculate_noise_welch

    # Create Plot
    plt.rcParams.update({"font.size": 20})

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 2)
    (ax1, ax2), (ax3, ax4) = gs.subplots()

    f_i, Pxx_i = noise_i(data, sample_rate, adc_counts_per_v, analysis_method)
    ax1.plot(f_i, Pxx_i)

    f_q, Pxx_q = noise_q(data, sample_rate, adc_counts_per_v, analysis_method)
    ax2.plot(f_q, Pxx_q)

    f_mag, Pxx_mag = noise_magnitude(data, sample_rate, analysis_method)
    ax3.plot(f_mag, Pxx_mag)

    f_phase, Pxx_phase = noise_phase(data, sample_rate, analysis_method)
    ax4.plot(f_phase, Pxx_phase)

    ax1.set_xscale("log")
    ax1.set_ylabel("I [dBm/Hz]")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.grid()

    ax2.set_xscale("log")
    ax2.set_ylabel("Q [dBm/Hz]")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.grid()

    ax3.set_xscale("log")
    ax3.set_ylabel("Magnitude [dBc/Hz]")
    ax3.set_xlabel("Frequency [Hz]")
    ax3.grid()

    ax4.set_xscale("log")
    ax4.set_ylabel("Phase [dBc/Hz]")
    ax4.set_xlabel("Frequency [Hz]")
    ax4.grid()

    if not filepath is None:
        plt.savefig(filepath)

    plt.show()


def noise_power_scan(dfs, sample_rate, adc_counts_per_v, method="welch", filepath=None):

    # Select Method
    if method == "welch":
        analysis_method = calculate_noise_welch
    elif method == "lpsd":
        analysis_method = calculate_noise_lpsd
    else:
        analysis_method = calculate_noise_welch

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(dfs)))

    fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True, constrained_layout=True)
    ax1, ax2, ax3, ax4 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

    for i, df in enumerate(dfs):

        data = np.array(df["I"]) + 1j * np.array(df["Q"])
        PdBm = ADC_to_dBm(data, adc_counts_per_v)

        f_i, Pxx_i = noise_i(data, sample_rate, adc_counts_per_v, analysis_method)
        ax1.plot(f_i, Pxx_i, label=f"P = {PdBm:.0f} dBm", color=colors[i], lw=1)
        ax1.legend(fontsize=8)

        f_q, Pxx_q = noise_q(data, sample_rate, adc_counts_per_v, analysis_method)
        ax2.plot(f_q, Pxx_q, label=f"P = {PdBm:.0f} dBm", color=colors[i], lw=1)
        ax2.legend(fontsize=8)

        f_mag, Pxx_mag = noise_magnitude(data, sample_rate, analysis_method)
        ax3.plot(f_mag, Pxx_mag, label=f"P = {PdBm:.0f} dBm", color=colors[i], lw=1)
        ax3.legend(fontsize=8)

        f_phase, Pxx_phase = noise_phase(data, sample_rate, analysis_method)
        ax4.plot(f_phase, Pxx_phase, label=f"P = {PdBm:.0f} dBm", color=colors[i], lw=1)
        ax4.legend(fontsize=8)

        ax1.set_xscale("log")
        ax1.set_ylabel("I [dBm/Hz]")
        ax1.set_xlim(f_i[1], f_i[-1])
        ax1.grid(which="both", alpha=0.3)

        ax2.set_xscale("log")
        ax2.set_ylabel("Q [dBm/Hz]")
        ax2.set_xlim(f_q[1], f_q[-1])
        ax2.grid(which="both", alpha=0.3)

        ax3.set_xscale("log")
        ax3.set_ylabel("Magnitude [dBc/Hz]")
        ax3.set_xlabel("Frequency [Hz]")
        ax3.set_xlim(f_mag[1], f_mag[-1])
        ax3.grid(which="both", alpha=0.3)

        ax4.set_xscale("log")
        ax4.set_ylabel("Phase [dBc/Hz]")
        ax4.set_xlabel("Frequency [Hz]")
        ax4.set_xlim(f_phase[1], f_phase[-1])
        ax4.grid(which="both", alpha=0.3)

    if not filepath is None:
        plt.savefig(filepath, bbox_inches="tight", dpi=600)

    plt.show()
