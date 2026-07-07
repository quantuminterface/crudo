from cirque import stimulation
from cirque import multichannelddc
from cirque import pp_channelizer
from cirque import fluxrampdemod
from cirque import eventdetection
from cirque import dmacontroller

from crudo import channel_configuration
from crudo import fluxramp_calibration
from crudo import tone_generation
from crudo import resonator

import numpy as np
import matplotlib.pyplot as plt


class ECHoReadoutChain:
    def __init__(self, connection, bands: int = 5, subbands: int = 4):

        self.index = 0
        self.bands = bands
        self.subbands = subbands

        self.stims = []
        self.fluxrampdemods = []
        self.triggers = []

        for i in range(self.bands):
            stim_name = f"stimulation_{i//2}_{i%2}"
            # stim_name = f"stimulation_{4-i}"
            self.stims.append(stimulation.Stimulation(connection, stim_name))

        self.ddc = channel_configuration.DownConversion()
        for i in range(self.bands * self.subbands):
            channelizer_name = f"processing_chain_{i//4}_signal_demodulation_chain_{i//2}_pp_channelizer_{i//2}_{i%2}"
            mcddc_name = f"processing_chain_{i//4}_signal_demodulation_chain_{i//2}_multi_channel_ddc_{i//2}_{i%2}"
            fluxrampdemod_name = f"processing_chain_{i//4}_signal_demodulation_chain_{i//2}_fluxramp_demodulation_{i//2}_{i%2}"
            event_name = f"signal_selection_chain_{i//4}_event_detection_{i//4}_{i%4}"

            pp_chan = pp_channelizer.PPChannelizer(connection, channelizer_name)
            mcddc = multichannelddc.MultiChannelDDC(connection, mcddc_name)
            self.fluxrampdemods.append(
                fluxrampdemod.FluxrampDemod(connection, fluxrampdemod_name)
            )
            self.triggers.append(eventdetection.EventDetection(connection, event_name))

            ddc_chain = channel_configuration.Subchain([pp_chan], mcddc)
            self.ddc.add_chain(ddc_chain)

        self.resonators: resonator.Resonator = []

    def setup_mux_from_csv(self, filename: str, emulator_frequencies: float):
        self.resonators = tone_generation.import_tones_from_file(filename).resonators
        self.resonators = tone_generation.spread_resonators(self.resonators, self.stims)
        for chain_index in range(self.bands):
            inv_band = True
            center_freq = self.stims[chain_index].get_rf_center_frequency()
            bb_frequencies = tone_generation.set_tones_from_object(
                self.stims[chain_index],
                self.resonators[chain_index],
                center_freq,
                invert_sideband=inv_band,
            )
            if inv_band:
                actual_frequencies = [
                    -frequency + center_freq for frequency in bb_frequencies
                ]
            else:
                actual_frequencies = [
                    frequency + center_freq for frequency in bb_frequencies
                ]

            for resonator_index, resonator in enumerate(self.resonators[chain_index]):
                resonator.baseband_frequency = bb_frequencies[resonator_index]
                resonator.actual_frequency = actual_frequencies[resonator_index]

        emu_resonators = tone_generation.ReadoutTones()
        for emu_freq in emulator_frequencies:
            emu_resonators.create_tone(frequency=emu_freq, amplitude=1)
        emu_resonators = emu_resonators.resonators
        emu_resonators = tone_generation.spread_resonators(emu_resonators, self.stims)
        for band_index, band in enumerate(emu_resonators):
            for emu_res in band:
                emu_res.baseband_frequency = emu_res.frequency - center_freq
                emu_res.actual_frequency = emu_res.frequency - center_freq
                self.resonators[band_index].append(emu_res)

        for chain_index in range(self.bands):
            channel_configuration.initialize_ddc(
                self.ddc.chains[4 * chain_index : 4 * chain_index + 4],
                self.resonators[chain_index],
                chain_index,
            )

    def setup_mux_from_object(self, resonators, invert_sideband: bool = True):
        self.resonators = resonators
        for chain_index in range(self.bands):
            center_freq = self.stims[chain_index].get_rf_center_frequency()
            bb_frequencies = tone_generation.set_tones_from_object(
                self.stims[chain_index],
                self.resonators[chain_index],
                center_freq,
                invert_sideband=invert_sideband,
            )
            if invert_sideband:
                actual_frequencies = [
                    -frequency + center_freq for frequency in bb_frequencies
                ]
            else:
                actual_frequencies = [
                    frequency + center_freq for frequency in bb_frequencies
                ]

            for resonator_index, resonator in enumerate(self.resonators[chain_index]):
                resonator.baseband_frequency = bb_frequencies[resonator_index]
                resonator.actual_frequency = actual_frequencies[resonator_index]

        for chain_index in range(self.bands):
            channel_configuration.initialize_ddc(
                self.ddc.chains[4 * chain_index : 4 * chain_index + 4],
                self.resonators[chain_index],
                chain_index,
            )

    def setup_fluxramp_demod(
        self,
        dma: dmacontroller.DMAController,
        fluxramp_frequency: float,
        accumulation_range=None,
        delay: int = 0,
    ):

        fluxramp_length = int(500e6 / 32 / fluxramp_frequency)

        if accumulation_range is None:
            accumulation_range = (0, fluxramp_length - 1)

        for fluxrampdemod in self.fluxrampdemods:
            fluxrampdemod.set_bypass_act(1)
            fluxrampdemod.set_fluxramp_length(fluxramp_length)
            fluxrampdemod.set_sync_delay(fluxrampdemod.get_channel_count() * delay)

        for band_index, band_resonators in enumerate(self.resonators):
            for resonator in band_resonators:
                subchain = self.subbands * band_index + resonator.subchain
                data, _ = dma.snapshot(
                    int(fluxramp_length * 4),
                    subchains=[subchain],
                    channels=[resonator.channel],
                    deinterleave=False,
                    is_iq=True,
                    dtype=np.int16(),
                    axis_stream_selection=2,
                    package_mode=False,
                )
                data = data[0::2][5:]

                fig, ax = plt.subplots(1, 3, figsize=(18, 3))
                plt.suptitle(
                    f"Resonator: {resonator.actual_frequency/1e9} GHz, subband {band_index}, channel {resonator.channel}"
                )

                fluxramp_calibration.plot_sync_flag(
                    data, fluxramp_length, delay, figure=ax[0]
                )

                sliced_data = fluxramp_calibration.cut_signal(
                    data,
                    fluxramp_length,
                    accumulation_range=accumulation_range,
                    plot=ax[1],
                    verbose=False,
                )

                if sliced_data is None:
                    continue

                try:
                    (
                        carrier_frequency_fit,
                        modulation_offset,
                    ) = fluxramp_calibration.find_modulation_freq(
                        np.abs(sliced_data), sample_rate=500e6 / 32, plot=[_, ax[2]]
                    )
                except:
                    continue

                carrier_frequency_fit = np.abs(carrier_frequency_fit)

                resonator.squid_frequency = carrier_frequency_fit

                flux_demod = self.fluxrampdemods[subchain]
                flux_demod.set_nco_frequency(resonator.channel, carrier_frequency_fit)
                flux_demod.set_nco_phase(
                    resonator.channel, resonator.fluxramp_phase_offset
                )
                flux_demod.set_offset(resonator.channel, int(modulation_offset))

        for fluxrampdemod in self.fluxrampdemods:
            fluxrampdemod.set_bypass_act(0)

    def setup_trigger(
        self, threshold_pos, threshold_neg, event_length, pretrigger_values
    ):
        for index, trigger in enumerate(self.triggers):
            trigger.set_trigger_engine(0)
            trigger.set_event_threshold_pos(threshold_pos)
            trigger.set_event_threshold_neg(threshold_neg)
            trigger.set_event_length(event_length)
            trigger.set_pretrigger_values(pretrigger_values)

            channels = []
            for band_index, band_resonators in enumerate(self.resonators):
                for resonator in band_resonators:
                    if (band_index * 4 + resonator.subchain == index) and (
                        not (resonator.channel in channels)
                    ):
                        channels.append(resonator.channel)

            trigger.set_active_channels(channels)

            trigger.enable()
