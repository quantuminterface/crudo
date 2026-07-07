from cirque import stimulation
from cirque import multichannelddc
from cirque import dmacontroller
from cirque import rfdc
from cirque import pp_channelizer
from cirque import tdmchannelmultipick
from cirque import vna
from cirque import bidirectionalmixer
from cirque import eventdetection
from crudo import channel_configuration
from crudo import tone_generation
from crudo import resonator


class BullkidReadoutChain:
    def __init__(self, connection, index: int, interleaved: bool, center_freq: float):
        self.index = index
        self.center_freq = center_freq
        self.interleaved = interleaved

        # Stimulation module for tone generation
        self.stim = stimulation.Stimulation(
            connection, f"processing_chain_{index}_stimulation_{index}"
        )

        self.vna = vna.VNA(connection, f"processing_chain_{index}_vna_{index}")

        self.bidir_mixer = bidirectionalmixer.BidirectionalMixer(
            connection, f"processing_chain_{index}_bidir_mixer_{index}"
        )

        ## Polyphase channelizer & MultiChannelDDC for channel separation
        channelizer_centered = pp_channelizer.PPChannelizer(
            connection, f"processing_chain_{index}_pp_channelizer_{index}_0"
        )
        channelizer_shifted = pp_channelizer.PPChannelizer(
            connection, f"processing_chain_{index}_pp_channelizer_{index}_1"
        )

        if not interleaved:
            binselect_centered = tdmchannelmultipick.TDMChannelMultipick(
                connection, f"processing_chain_{index}_bin_select_{index}_0"
            )
            binselect_shifted = tdmchannelmultipick.TDMChannelMultipick(
                connection, f"processing_chain_{index}_bin_select_{index}_1"
            )
            ddc_centered = multichannelddc.MultiChannelDDC(
                connection, f"processing_chain_{index}_multi_channel_ddc_{index}_0"
            )
            ddc_shifted = multichannelddc.MultiChannelDDC(
                connection, f"processing_chain_{index}_multi_channel_ddc_{index}_1"
            )

            chain_0 = channel_configuration.Subchain(
                channelizer_centered, ddc_centered, binselect_centered
            )
            chain_1 = channel_configuration.Subchain(
                channelizer_shifted, ddc_shifted, binselect_shifted
            )

            self.ddc = channel_configuration.DownConversion()
            self.ddc.add_chain(chain_0)
            self.ddc.add_chain(chain_1)

            # Workaoround: update center frequency of channelizers
            # ToDo: it would be better to change the calibration system
            # so that the center frequency of the channelizer is always 0.
            self.ddc.chains[0].chan[0].set_center_frequency(center_freq)
            self.ddc.chains[1].chan[0].set_center_frequency(center_freq)

            self.triggers = []
            self.triggers.append(
                eventdetection.EventDetection(
                    connection, f"processing_chain_{index}_event_detection_{index}_0"
                )
            )
            self.triggers.append(
                eventdetection.EventDetection(
                    connection, f"processing_chain_{index}_event_detection_{index}_1"
                )
            )

        else:
            binselect = tdmchannelmultipick.TDMChannelMultipick(
                connection, f"processing_chain_{index}_bin_select_{index}"
            )

            ddc = multichannelddc.MultiChannelDDC(
                connection, f"processing_chain_{index}_multi_channel_ddc_{index}"
            )

            chain = channel_configuration.Subchain(
                [channelizer_centered, channelizer_shifted], ddc, binselect
            )

            self.ddc = channel_configuration.DownConversion()
            self.ddc.add_chain(chain)

            # Workaoround: update center frequency of channelizers
            # ToDo: it would be better to change the calibration system
            # so that the center frequency of the channelizer is always 0.
            self.ddc.chains[0].chan[0].set_center_frequency(center_freq)
            self.ddc.chains[0].chan[1].set_center_frequency(center_freq)

            self.triggers = []
            self.triggers.append(
                eventdetection.EventDetection(
                    connection, f"processing_chain_{index}_event_detection_{index}"
                )
            )
        self.resonators: resonator.Resonator = []

    def setup_mux(self, filename: str):
        self.resonators = tone_generation.import_tones_from_file(filename).resonators
        bb_frequencies = tone_generation.set_tones_from_object(
            self.stim, self.resonators, self.center_freq, invert_sideband=False
        )
        actual_frequencies = [
            frequency + self.center_freq for frequency in bb_frequencies
        ]
        for resonator_index, resonator in enumerate(self.resonators):
            resonator.mux = self.index
            resonator.actual_frequency = actual_frequencies[resonator_index]
        channel_configuration.setup_ddc(self.ddc, self.resonators, 0)

    def setup_mux_from_object(self, resonators):
        self.resonators = resonators
        bb_frequencies = tone_generation.set_tones_from_object(
            self.stim, self.resonators, self.center_freq, invert_sideband=False
        )
        actual_frequencies = [
            frequency + self.center_freq for frequency in bb_frequencies
        ]
        for resonator_index, resonator in enumerate(self.resonators):
            resonator.mux = self.index
            resonator.actual_frequency = actual_frequencies[resonator_index]
        channel_configuration.setup_ddc(self.ddc, self.resonators, 0)

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
            for resonator in self.resonators:
                if (resonator.subchain == index) and (
                    not (resonator.channel in channels)
                ):
                    channels.append(resonator.channel)

            trigger.set_active_channels(channels)

            trigger.enable()
