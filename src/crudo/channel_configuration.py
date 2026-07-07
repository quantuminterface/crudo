import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from dataclasses import dataclass

from crudo import channelizer
from crudo import data_acquisition
from crudo import resonator
from cirque import multichannelddc
from cirque import stimulation
from cirque import pp_channelizer
from cirque import tdmchannelmultipick
from cirque import stimulation


class Subchain:
    def __init__(
        self,
        chan: "pp_channelizer.PP_Channelizer",
        ddc: "multichannelddc.MultiChannelDDC",
        binselect: "tdmchannelmultipick.TDMChannelMultipick" = None,
    ):
        self.chan = []
        for ppc in chan:
            self.chan.append(channelizer.Channelizer(ppc))
        self.binselect = binselect
        self.ddc = ddc


class DownConversion:
    def __init__(self):
        self.chains = []
        self.channel_offsets = []

    def add_chain(self, chain: Subchain, channel_offset=0):
        self.chains.append(chain)
        self.channel_offsets.append(channel_offset)


@dataclass
class PPCMap:
    chain_index: int
    ppc_index: int
    channel: int
    delta_frequency: float


def spread_resonators(
    resonators: resonator.Resonator, stimulations: stimulation.Stimulation
):
    for resonator in resonators:
        for band_index, stim in enumerate(stimulations):
            stim_lo = stim.get_rf_center_frequency()
            if np.abs(stim_lo - resonator.frequency) < 400e6:
                resonator.band = band_index
                continue


def setup_ddc(chains: DownConversion, resonators: resonator.Resonator, offset=0):

    # Step 1: Allocate appropriate bins for each resonator
    for resonator in resonators:
        rx_frequency = resonator.actual_frequency - offset

        for chain_index, chain in enumerate(chains.chains):
            n_ppc = len(chain.chan)
            for ppc_index, ppc in enumerate(chain.chan):
                try:
                    bin, delta_frequency = ppc.calculate_delta_frequency(rx_frequency)
                    print(
                        f"Chain index: {chain_index}, PPC index: {ppc_index}, Bin: {bin}, delta_frequency: {delta_frequency}"
                    )
                except:
                    continue

                if np.abs(delta_frequency) < np.abs(resonator.delta_frequency):
                    interleaved_bin = n_ppc * bin + ppc_index
                    print(f"Interleaved bin: {interleaved_bin}")
                    resonator.set_calibration_values(
                        chain_index, interleaved_bin, delta_frequency, True
                    )

    # Step 2: Configure binselect and multi_channel_ddc
    global_bins = []
    for chain_index, chain in enumerate(chains.chains):
        chain_resonators = []

        for resonator in resonators:
            if resonator.subchain == chain_index:
                chain_resonators.append(resonator)

        # Sort resonators by increasing bin number
        chain_resonators.sort(key=lambda x: x.bin)

        bins = []

        for index, resonator in enumerate(chain_resonators):
            bins.append(resonator.bin)
            resonator.set_readout_channel(index)
            chain.ddc.set_nco(index, resonator.delta_frequency, 0.0)

        chain.ddc.enable()

        # chain.binselect.set_active_channels(bins)

        global_bins.append(bins)

    for chain_index, chain in enumerate(chains.chains):
        chain.binselect.set_active_channels(global_bins[chain_index])


# Check which ppcs are suitable for the current frequency and return ppc index, channel index and delta_frequency
def map_frequency(frequency: float, chains: DownConversion):
    ppc_map = []
    for chain_index, chain in enumerate(chains):
        for ppc_index, ppc in enumerate(chain.chan):
            try:
                channel, delta_frequency = ppc.calculate_delta_frequency(frequency)
                my_map = PPCMap(chain_index, ppc_index, channel, delta_frequency)
                ppc_map.append(my_map)
            except:
                continue

    # Sort map with increasing delta_frequencies
    if len(ppc_map) > 0:
        return sorted(ppc_map, key=lambda PPCMap: np.abs(PPCMap.delta_frequency))
    else:
        return None


def initialize_ddc(chains, resonators: resonator.Resonator, band: int):

    if chains[0].binselect is None:
        limit_bins = True
    else:
        limit_bins = False

    for resonator in resonators:
        ppc_map = map_frequency(resonator.baseband_frequency, chains)

        if not (ppc_map is None):
            chain_index = ppc_map[0].chain_index
            ppc_index = ppc_map[0].ppc_index
            channel = ppc_map[0].channel
            delta_frequency = ppc_map[0].delta_frequency

            chains[chain_index].ddc.set_nco(channel, delta_frequency)
            chains[chain_index].ddc.enable()

            resonator.set_calibration_values(
                chain_index, channel, delta_frequency, True, band
            )
            resonator.set_readout_channel(channel)


def initialize_ddc_backup(chains: DownConversion, frequencies, offset=0):
    rx_frequencies = [frequency - offset for frequency in frequencies]

    global_active_channels = []
    global_delta_frequencies = []

    for chain_index in range(len(chains.chains)):
        tdm_channels = chains.chains[chain_index].ddc.get_tdm_channels()
        nco_phases = [0.0 for _ in range(tdm_channels)]

        filtered_frequencies, rx_frequencies = (
            chains.chains[chain_index].chan[0].filter_frequencies(rx_frequencies)
        )
        # print(f"Filtered_frequencies: {filtered_frequencies}")
        delta_frequencies = (
            chains.chains[chain_index]
            .chan[0]
            .set_readout_frequencies(filtered_frequencies)
        )

        active_channels = [
            True if delta_frequency != 0 else False
            for delta_frequency in delta_frequencies
        ]

        # Workaround due to DMA issue: Sometimes there is an offset on the desired channel
        active_channels = deque(active_channels)
        active_channels.rotate(chains.channel_offsets[chain_index])
        delta_frequencies = deque(delta_frequencies)
        delta_frequencies.rotate(chains.channel_offsets[chain_index])

        global_active_channels.append(active_channels)
        global_delta_frequencies.append(delta_frequencies)

        for channel_index in range(tdm_channels):
            chains.chains[chain_index].ddc.set_nco(
                channel_index, delta_frequencies[channel_index], 0.0
            )
        chains.chains[chain_index].ddc.enable()

    return global_active_channels, global_delta_frequencies


def calibrate_iq(chains: DownConversion, dma, frequencies):

    active_channels, delta_frequencies = initialize_ddc(chains, frequencies)

    for chain_index, chain in enumerate(active_channels):
        for channel_index, channel in enumerate(chain):
            if channel == True:
                data, loss = dma.snapshot(
                    1000,
                    dtype=np.int16(),
                    subchains=[chain_index],
                    channels=[channel_index],
                    package_mode=False,
                )
                i = data[0::4]
                q = data[1::4]
                i = i[10:]
                q = q[10:]
                plt.plot(i, q)
                mean_i = np.mean(i)
                mean_q = np.mean(q)
                phase = np.angle(mean_i + mean_q * 1.0j)
                print(
                    f"Chain: {chain_index}, Channel: {channel_index}, Delta freq: {delta_frequencies[chain_index][channel_index]}, Phase: {phase}"
                )
                chains.chains[chain_index].ddc.set_nco(
                    channel_index, delta_frequencies[chain_index][channel_index], -phase
                )
        # chains.chains[chain_index].ddc.enable()
    plt.show()
