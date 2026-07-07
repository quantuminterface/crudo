from cirque import dmacontroller
from crudo import resonator
from crudo import bullkid_readout_chain
import numpy as np


def _define_dma_channels(
    muxes: bullkid_readout_chain,
):
    n_channels = 64
    if len(muxes) == 1:
        channels = [res.channel for res in muxes[0].resonators]
    else:
        channels = []
        for mux in muxes:
            mux_index = mux.index
            for res in mux.resonators:
                channels.append(mux_index * n_channels + res.channel)
    channels.sort()
    return channels


def map_dma_data(
    data: np.array,
    muxes: bullkid_readout_chain,
    is_iq: bool,
):
    n_channels = 64
    readout_channels = _define_dma_channels(muxes)

    if is_iq == False:
        data = data[0::2] + 1j * data[1::2]

    # Crop data
    n_channel_samples = int(np.floor(len(data) / len(readout_channels)))
    data = data[: len(readout_channels) * n_channel_samples]

    output = []
    for mux in muxes:
        if len(muxes) == 1:
            mux_index = 0
        else:
            mux_index = mux.index
        resonators = mux.resonators
        mux_output = []
        for resonator in resonators:
            mux_output.append(
                data[
                    readout_channels.index(
                        mux_index * n_channels + resonator.channel
                    ) :: len(readout_channels)
                ]
            )
        output.append(mux_output)

    return output


def raw_iq_snapshot(
    dma: dmacontroller.DMAController, n_samples, muxes: bullkid_readout_chain
):
    # Workaround: add offset to remove first samples of snapshot
    offset = 10

    n_chains = dma.get_parallel_streams()

    if len(muxes) > n_chains:
        raise Exception(f"Only {n_chains} number of lines can be acquired in parallel")

    subchains = [mux.index for mux in muxes]
    subchains.sort()

    # Read raw data from ADC
    data, _ = dma.snapshot(
        (n_samples + offset),
        subchains=subchains,
        axis_stream_selection=0,
        package_mode=False,
        is_iq=True,
        dtype=np.int16(),
        timeout=30,
    )

    output = np.zeros((len(muxes), n_samples))
    output = []
    for index, mux in enumerate(muxes):
        data_index = subchains.index(mux.index)
        output.append((data[data_index::n_chains])[offset:])

    return output


def tdm_iq_snapshot(
    dma: dmacontroller.DMAController,
    n_samples,
    muxes: bullkid_readout_chain,
):
    # ToDo: Read number of channels from FPGA
    n_channels = 64
    fs_channel = int(125e6 / 32 / 80)

    if len(muxes) == 1:
        acquisition_mode = 1
        subchains = [muxes[0].index]
    else:
        acquisition_mode = 3
        subchains = [0]

    readout_channels = _define_dma_channels(muxes)

    data, _ = dma.snapshot(
        n_samples * len(readout_channels),
        channels=readout_channels,
        subchains=subchains,
        dtype=np.int16(),
        axis_stream_selection=acquisition_mode,
        package_mode=False,
        sample_rate=fs_channel * len(readout_channels),
        is_iq=True,
    )

    # Cut data of upper parallel streams
    data = data[0 :: dma.get_parallel_streams()]

    output = map_dma_data(data, muxes, True)
    return output


def tdm_iq_file_snapshot(
    dma: dmacontroller.DMAController,
    muxes: bullkid_readout_chain,
    samples: int,
    file_path: str,
    file_name: str,
):
    # ToDo: Read number of channels from FPGA
    n_channels = 64
    fs_channel = int(125e6 / 32 / 80)

    if len(muxes) == 1:
        acquisition_mode = 1
        subchains = [muxes[0].index]
    else:
        acquisition_mode = 3
        subchains = [0]

    readout_channels = _define_dma_channels(muxes)

    dma.file_snapshot(
        sample_count=samples * len(readout_channels),
        channels=readout_channels,
        subchains=subchains,
        package_mode=False,
        axis_stream_selection=acquisition_mode,
        file_name=file_name,
        sub_path=file_path,
        sample_rate=fs_channel * len(readout_channels),
        disk_type="SD",
    )


def tdm_iq_stream(
    dma: dmacontroller.DMAController,
    muxes: bullkid_readout_chain,
    acquisition_time: int,
    file_path: str,
    file_name: str,
):
    # ToDo: Read number of channels from FPGA
    n_channels = 64
    fs_channel = int(125e6 / 32 / 80)

    if len(muxes) == 1:
        acquisition_mode = 1
        subchains = [muxes[0].index]
    else:
        acquisition_mode = 3
        subchains = [0]

    readout_channels = _define_dma_channels(muxes)

    dma.continuous_grpc_stream_to_file(
        channels=readout_channels,
        subchains=subchains,
        file_name=file_name,
        target_path=file_path,
        package_mode=False,
        sample_rate=fs_channel * len(readout_channels),
        axis_stream_selection=acquisition_mode,
        acquisition_time=acquisition_time,
    )


def tdm_iq_file_stream(
    dma: dmacontroller.DMAController,
    muxes: bullkid_readout_chain,
    acquisition_time: int,
    file_path: str,
    file_name: str,
):
    # ToDo: Read number of channels from FPGA
    fs_channel = int(125e6 / 32 / 80)
    n_channels = 64

    if len(muxes) == 1:
        acquisition_mode = 1
        subchains = [muxes[0].index]
    else:
        acquisition_mode = 3
        subchains = [0]

    readout_channels = _define_dma_channels(muxes)

    dma.continuous_file_stream(
        channels=readout_channels,
        subchains=subchains,
        file_name=file_name,
        sub_path=file_path,
        package_mode=False,
        sample_rate=fs_channel * len(readout_channels),
        axis_stream_selection=acquisition_mode,
        acquisition_time=acquisition_time,
    )


def triggered_iq_snapshot(
    dma: dmacontroller.DMAController,
    n_samples,
    wafer: bullkid_readout_chain,
    resonators: resonator.Resonator,
):

    # set active channels in event_detection
    for index, event_det in enumerate(wafer.triggers):
        active_channels = []
        for resonator in resonators:
            if resonator.subchain == index:
                active_channels.append(resonator.channel)
        event_det.set_active_channels(active_channels)

    # Start measurement
    data, _ = dma.snapshot(
        n_samples,
        dtype=np.int32(),
        timeout=10,
        package_mode=True,
        axis_stream_selection=2,
    )
    return data


def acquire_triggered_data(
    dma: dmacontroller.DMAController,
    wafer: bullkid_readout_chain,
    resonators: resonator.Resonator,
    file_path: str,
    file_name: str,
):

    # set active channels in event_detection
    for index, event_det in enumerate(wafer.triggers):
        active_channels = []
        for resonator in resonators:
            if resonator.subchain == index:
                active_channels.append(resonator.channel)
        event_det.set_active_channels(active_channels)

    channels = [i for i in range(32)]
    subchains = [0, 1]
    dma.continuous_grpc_stream_to_file(
        channels=channels,
        subchains=subchains,
        file_name=file_name,
        target_path=file_path,
        package_mode=True,
        axis_stream_selection=2,
        timeout=120,
    )
