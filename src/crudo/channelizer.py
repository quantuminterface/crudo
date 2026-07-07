from math import ceil, log2
from bitarray.util import int2ba, ba2int
from cirque import pp_channelizer


class Channelizer:
    """
    The Channelizer is used to separate the readout tones from each other.
    For efficiency reasons, the channelizer does not have a programming interface and is not runtime changeable.
    This class is used to prepare the the signals for further processing by mapping them to their specific channel.
    """

    def __init__(self, pp_channelizer):
        """
        For initialization, the parameters of the channelizer have to be set.
        These parameters are set in the hardware and cannot be changed at runtime.
        Depending on these parameters, the channel mapping is performed.

        Args:
            ppc_decimation (int): Number of channels the spectrum is split into
            ppc_bandwidth (float): Single sided bandwidth of the individual channels
            reverse_iq (bool): ??
            shifted (bool): Defines whether the chain has been frequency shifted in order to fill the blind intervalls
            sample_rate (int): Input sample rate of the module
            center_frequency (int): Center Frequency of the input spectrum. Is set by the ADC or the DDC
            bb_passband (float): ??
        """
        self._ppc_decimation = pp_channelizer.get_decimation()
        self._channel_bandwidth = pp_channelizer.get_channel_bandwidth()
        self._ppc_bandwidth = pp_channelizer.get_ppc_bandwidth()
        self._bb_passband = pp_channelizer.get_passband()

        # Members
        self._readout_frequencies = []
        self._delta_frequencies = [0] * self._ppc_decimation
        self._channel_selection = 0x0
        self._sample_rate = pp_channelizer.get_sample_rate()
        if pp_channelizer.get_shifted():
            self._shift_frequency = self._sample_rate / self._ppc_decimation / 2
        else:
            self._shift_frequency = 0
        if pp_channelizer.get_reverse_iq():
            self._reverse_freq = -1
        else:
            self._reverse_freq = 1

        self._center_frequency = pp_channelizer.get_center_frequency()

    def calculate_delta_frequency(self, input_frequency):
        """
        Calculates the channel and delta frequency for the input frequency
        """
        delta_frequency = self._reverse_freq * self._get_channel_delta_frequency(
            input_frequency
        )

        if abs(delta_frequency) > self._channel_bandwidth:
            raise RuntimeError(
                f"Given frequency {input_frequency} is in a blind interval, {delta_frequency} Hz away from next center."
            )

        bin = self.get_channel_for_frequency(input_frequency)
        delta_frequency = delta_frequency

        return bin, delta_frequency

    def set_readout_frequencies(self, list_of_frequencies):
        """
        Calculates the delta frequencies in the individual channels based on the input frequencies
        """
        self._delta_frequencies = [0] * self._ppc_decimation
        self._channel_selection = 0x0
        self._readout_frequencies = list_of_frequencies

        for frequency in self._readout_frequencies:
            current_delta_frequency = (
                self._reverse_freq * self._get_channel_delta_frequency(frequency)
            )

            if self._delta_frequencies[self.get_channel_for_frequency(frequency)] != 0:
                # TODO be more precise
                # raise RuntimeError(
                #    "Only one readout frequency per channel is permitted."
                # )
                print(
                    f"Warning: two readout tones in channel {self.get_channel_for_frequency(frequency)}. Discarded delta frequency is {current_delta_frequency}."
                )
                continue

            if abs(current_delta_frequency) < self._channel_bandwidth:
                self._delta_frequencies[
                    self.get_channel_for_frequency(frequency)
                ] = current_delta_frequency
            else:
                # TODO be more precise
                raise RuntimeError(
                    f"Given frequency {frequency} is in a blind interval, {current_delta_frequency} Hz away from next center."
                )

            self._channel_selection = (
                self._channel_selection | 1 << self.get_channel_for_frequency(frequency)
            )

        return self._delta_frequencies

    def get_channel_selection(self):
        """
        Returns an array with the active channels
        """
        return self._channel_selection

    def get_channel_map(self, selected_frequencies):
        """
        Returns the mapping of the frequencies in the channels
        """
        # Map: String(Frequency) -> Data index is returned.
        ret = {}

        start_iter = 0
        double_check = True

        for channel in range(self._ppc_decimation):
            double_check = False
            for frequency in selected_frequencies:
                if self._channel_number_reverse(
                    channel
                ) == self.get_channel_for_frequency(frequency):

                    if double_check:
                        raise RuntimeError("Frequency occuping the same channel.")
                    # Key must be hashable
                    ret[str(frequency)] = start_iter
                    start_iter += 1
                    double_check = True

        return ret

    def get_channel_count(self):
        """
        Returns the number of channels the pp_channelizer is configured with
        """
        return self._ppc_decimation

    def set_ddc_passthrough(self):
        """
        Activates passthrough mode, without channelizing the data
        """
        raise NotImplementedError("Method not implemented")

    def get_sample_rate(self):
        """
        Returns the sample rate the channelizer operates with
        """
        return self._sample_rate

    def get_channel_for_frequency(self, frequency, verbose=False):
        """
        Returns the Channel number for a given frequency
        """
        channel_spacing = self.get_sample_rate() / self._ppc_decimation

        relative_frequency = frequency - self._center_frequency

        passband_frequency = (self.get_sample_rate() / 2) * self._bb_passband

        if abs(relative_frequency) > passband_frequency:
            raise RuntimeError(
                f"Given frequency over nyquist frequency, or filter passband which is: {(self.get_sample_rate() / 2) * self._bb_passband} from center."
            )

        relative_shifted_frequency = self._reverse_freq * (
            relative_frequency + self._shift_frequency
        )

        channel_number_natural_order = round(
            relative_shifted_frequency / channel_spacing
        )

        if channel_number_natural_order < 0:
            channel_number_natural_order = (
                self._ppc_decimation + channel_number_natural_order
            )
        if channel_number_natural_order > self._ppc_decimation - 1:
            channel_number_natural_order = (
                channel_number_natural_order - self._ppc_decimation
            )

        if verbose:
            print(f"{channel_number_natural_order} (natural) for {frequency} ")

        # if relative_frequency > -(self._sample_rate/self._ppc_decimation/2):
        #    # Channel 0 to N/2 for positive frequencies
        #    channel_number_natural_order = round(relative_shifted_frequency / channel_spacing)
        #    print("%d for %f " % (channel_number_natural_order,frequency))

        # else:
        #    # Channel N/2(min) to N-1(max) for negative frequencies
        #    channel_number_natural_order = round((self.get_sample_rate()+relative_shifted_frequency) / channel_spacing)
        #    print("%d for %f " % (channel_number_natural_order,frequency))

        return self._channel_number_reverse(channel_number_natural_order)

    def _channel_number_reverse(self, channel_number):
        """
        Returns the natural order of the channels
        """
        reversed_ba = int2ba(channel_number, length=ceil(log2(self._ppc_decimation)))
        reversed_ba.reverse()
        # print("Got channel %d, reversed %d."  %(channel_number,ba2int(reversed_ba)))
        return ba2int(reversed_ba)

    def _get_channel_center_frequency(self, channel_number):
        """
        Returns the Center frequnency of a channel (relative to the defined center frequency)
        """
        channel_spacing = self.get_sample_rate() / self._ppc_decimation

        if channel_number >= self._ppc_decimation:
            raise RuntimeError(f"Channel Number {channel_number} out of bounds")

        natural_channel_number = self._channel_number_reverse(channel_number)

        # return_frequency = 0

        channel_center_frequency = None

        if natural_channel_number <= self._ppc_decimation / 2:
            channel_center_frequency = (
                self._reverse_freq * (channel_spacing * natural_channel_number)
                - self._shift_frequency
            )
        else:
            channel_center_frequency = (
                self._reverse_freq
                * (-channel_spacing * (self._ppc_decimation - natural_channel_number))
                - self._shift_frequency
            )

        # if self._shift_frequency:
        #    # Shifted frequency requires the last band to mapped to negative frequencies
        #    if reverse_channel_number > self._ppc_decimation/2-1:
        #        return_frequency = self._reverse_freq*channel_spacing*(reverse_channel_number-self._ppc_decimation)+self._shift_frequency
        #    else:
        #        return_frequency = self._reverse_freq*channel_spacing*reverse_channel_number+self._shift_frequency
        # else:
        #    if reverse_channel_number > self._ppc_decimation/2:
        #        return_frequency = self._reverse_freq*channel_spacing*(reverse_channel_number-self._ppc_decimation)+self._shift_frequency
        #    else:
        #        return_frequency = self._reverse_freq*channel_spacing*reverse_channel_number+self._shift_frequency

        return channel_center_frequency

    def _get_channel_delta_frequency(self, frequency, verbose=False):
        channel_center = self._get_channel_center_frequency(
            self.get_channel_for_frequency(frequency)
        )
        if verbose:
            print(f"Channel center: {channel_center}")

        # if the channel is on the lower sideband and the center of the last channel must be negated
        # (The last channel of the unshifted channelizer is in upper and lower sideband)

        # difference = (frequency+self._center_frequency)-channel_center
        difference = (frequency - self._center_frequency) - channel_center

        if verbose:
            print(f"Difference: {difference}")

        if difference > self._sample_rate / 2:
            return difference - self._sample_rate
        elif difference < -self._sample_rate / 2:
            return difference + self._sample_rate
        else:
            return difference

        # if abs(channel_center) == self.get_sample_rate()/2+self._center_frequency and (frequency - self._center_frequency)*self._reverse_freq < 0:
        #    return frequency - self._get_channel_center_frequency(
        #        self.get_channel_for_frequency(frequency))+self._reverse_freq*self.get_sample_rate()
        # else:
        #    return frequency - self._get_channel_center_frequency(
        #        self.get_channel_for_frequency(frequency))

    def filter_frequencies(self, frequency_array, verbose=False):
        """
        Filters a list of frequencies in selected frequencies, that ly inside the channelizer
        and a list of discarded, which are either in blind intervals or outside of the nyquist band.
        """
        selected_frequencies = []
        discarded_frequencies = []

        for frequency in frequency_array:
            relative_frequency = frequency - self._center_frequency

            if (
                abs(relative_frequency)
                >= (self.get_sample_rate() / 2) * self._bb_passband
            ):
                if verbose:
                    print(f"Discarded {frequency}: Out of nyquist/passband.")
                discarded_frequencies.append(frequency)
                continue

            current_delta_frequency = (
                self._reverse_freq * self._get_channel_delta_frequency(frequency)
            )
            if abs(current_delta_frequency) >= self._channel_bandwidth:
                if verbose:
                    print(f"Discarded {frequency}: In blind interval.")
                discarded_frequencies.append(frequency)
                continue

            if verbose:
                print(
                    f"Selected {frequency}: in  Channel {self.get_channel_for_frequency(frequency)}"
                )
            selected_frequencies.append(frequency)

        return [selected_frequencies, discarded_frequencies]

    def set_center_frequency(self, center_frequency):
        # In some configurations, the center frequency of the channelizer might be updated after booting the system
        # This is especially true for RFSoC systems that allow changing the mixing frequency at runtime
        # To be able to accurately calculate the channel frequencies, the users need to be able to update center frequency accordingly
        self._center_frequency = center_frequency
