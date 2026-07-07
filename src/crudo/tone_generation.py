from cirque import stimulation
from cirque import rfdc
from crudo import resonator
import pandas as pd
import numpy as np


class ReadoutTones:
    def __init__(self):
        self.resonators = []

    def create_tone(
        self, frequency, amplitude, phase=0.0, phase_iqi=0.0, amplitude_iqi=1.0
    ):
        self.resonators.append(
            resonator.Resonator(frequency, amplitude, phase, phase_iqi, amplitude_iqi)
        )

    def create_tones(
        self, frequencies, amplitudes, phases=None, phases_iqi=None, amplitudes_iqi=None
    ):
        num_tones = len(frequencies)
        if phases is None:
            phases = [0.0 for _ in range(num_tones)]
        if phases_iqi is None:
            phases_iqi = [0.0 for _ in range(num_tones)]
        if amplitudes_iqi is None:
            amplitudes_iqi = [1.0 for _ in range(num_tones)]

        for i in range(num_tones):
            self.resonators.append(
                resonator.Resonator(
                    frequencies[i],
                    amplitudes[i],
                    phases[i],
                    phases_iqi[i],
                    amplitudes_iqi[i],
                )
            )


def import_tones_from_file(filename: str) -> ReadoutTones:
    df = pd.read_csv(filename, sep=",", header=0)
    tones: ReadoutTones = ReadoutTones()

    try:
        frequencies = df.frequency.tolist()
    except (AttributeError, KeyError):
        print("Could not load frequencies from file. No tones imported")
        return tones
    try:
        indices = df.index.tolist()
    except (AttributeError, KeyError):
        print("Could not load indices from file. Use ascending order for indexing.")
        indices = [i for i in range(len(frequencies))]
    try:
        amplitudes = df.amplitude.tolist()
    except (AttributeError, KeyError):
        print(
            "Could not load amplitudes from file, use default values 1/noChannels for all tones."
        )
        amplitudes = [1 / len(frequencies) for _ in frequencies]
    try:
        phases = df.phase.tolist()
    except (AttributeError, KeyError):
        print("Could not load phases from file, use default values 0.0 for all tones.")
        phases = [0.0 for _ in frequencies]
    try:
        phases_iqi = df.phases_iqi.tolist()
    except (AttributeError, KeyError):
        phases_iqi = [0.0 for _ in frequencies]
    try:
        amplitudes_iqi = df.amplitudes_iqi.tolist()
    except (AttributeError, KeyError):
        amplitudes_iqi = [1.0 for _ in frequencies]

    tones.create_tones(frequencies, amplitudes, phases, phases_iqi, amplitudes_iqi)

    return tones


def export_tones_to_file(filename: str, tones: ReadoutTones):
    df = pd.DataFrame()
    df["frequency"] = tones.frequencies
    df["amplitude"] = tones.amplitudes
    df["phase"] = tones.phases
    df["phase_iqi"] = tones.phases_iqi
    df["amplitude_iqi"] = tones.amplitudes_iqi
    df.to_csv(filename, sep="\t", index=False, encoding="utf-8")


def spread_resonators(
    resonators: "resonator.Resonator", stimulations: stimulation.Stimulation
) -> resonator.Resonator:
    distributed_resonators = []
    for stim in stimulations:
        stim_lo = stim.get_rf_center_frequency()
        stim_resonators = []
        for resonator in resonators:
            if np.abs(stim_lo - resonator.frequency) < 400e6:
                stim_resonators.append(resonator)
        distributed_resonators.append(stim_resonators)
    return distributed_resonators


def set_tones(
    stim: "stimulation.Stimulation",
    frequencies,
    amplitudes=None,
    phases=None,
    phases_iqi=None,
    amplitudes_iqi=None,
    mix_freq=0,
    invert_sideband=False,
):
    stim.disable()
    stim.clear_samples()
    if amplitudes is None:
        amplitudes = [1.0 / len(frequencies) for _ in range(len(frequencies))]
    if phases is None:
        phases = [0.0 for _ in range(len(frequencies))]
    if phases_iqi is None:
        phases_iqi = [0.0 for _ in range(len(frequencies))]
    if amplitudes_iqi is None:
        amplitudes_iqi = [1.0 for _ in range(len(frequencies))]
    bb_frequencies = calc_baseband_freqs(frequencies, mix_freq, invert_sideband)
    actual_bb_frequencies = stim.add_tones(
        bb_frequencies, amplitudes, phases, phases_iqi, amplitudes_iqi
    )
    stim.enable()
    actual_frequencies = [frequency - mix_freq for frequency in actual_bb_frequencies]
    return actual_frequencies


def set_tone(
    stim: "stimulation.Stimulation",
    frequency,
    amplitude=None,
    phase=None,
    phase_iqi=None,
    amplitude_iqi=None,
    mix_freq=0,
    invert_sideband=False,
):
    if amplitude is None:
        amplitude = 1.0
    if phase is None:
        phase = [0.0]
    if phase_iqi is None:
        phase_iqi = [0.0]
    if amplitude_iqi is None:
        amplitude_iqi = [1.0]
    if invert_sideband:
        bb_freq = mix_freq - frequency
    else:
        bb_freq = frequency - mix_freq
    actual_frequencies = set_tones(
        stim, [bb_freq], [amplitude], [phase], [phase_iqi], [amplitude_iqi]
    )
    return actual_frequencies[0] + mix_freq


def set_tones_from_object(
    stim: "stimulation.Stimulation",
    resonators: "resonator.Resonator",
    mix_freq=0,
    invert_sideband=False,
):
    resonator_frequencies = [resonator.frequency for resonator in resonators]
    resonator_amplitudes = [resonator.amplitude for resonator in resonators]
    resonator_phases = [resonator.phase for resonator in resonators]
    resonator_phases_iqi = [resonator.phase_iqi for resonator in resonators]
    resonator_amplitudes_iqi = [resonator.amplitude_iqi for resonator in resonators]
    bb_frequencies = calc_baseband_freqs(
        resonator_frequencies, mix_freq, invert_sideband
    )
    actual_bb_frequencies = set_tones(
        stim,
        bb_frequencies,
        resonator_amplitudes,
        resonator_phases,
        resonator_phases_iqi,
        resonator_amplitudes_iqi,
    )
    return actual_bb_frequencies


def calc_baseband_freqs(rf_frequencies, mix_freq, invert_sideband):
    if invert_sideband:
        return [mix_freq - frequency for frequency in rf_frequencies]
    else:
        return [frequency - mix_freq for frequency in rf_frequencies]
