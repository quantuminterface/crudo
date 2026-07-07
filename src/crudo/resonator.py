from dataclasses import dataclass


@dataclass
class Resonator:
    frequency: float
    amplitude: float
    phase: float
    phase_iqi: float
    amplitude_iqi: float
    mux: int = -1

    baseband_frequency: float = float("inf")
    actual_frequency: float = float("inf")

    band: int = -1
    subchain: int = -1
    bin: int = -1
    delta_frequency: float = float("inf")
    active: bool = False
    channel: int = -1

    squid_frequency: float = -1
    fluxramp_phase_offset: float = 0.0

    def set_calibration_values(
        self, subchain, channel, delta_frequency, active, band=1
    ):
        self.band = band
        self.subchain = subchain
        self.bin = channel
        self.delta_frequency = delta_frequency
        self.active = active

    def set_readout_channel(self, channel):
        self.channel = channel
