import math
import numpy as np


class Event:
    def __init__(
        self,
        channel,
        chain,
        timestamp,
        filterresult,
        eventlength,
        iq,
        pileup,
        trigger_type,
        trigger_engine,
        samples,
    ):
        self.channel = channel
        self.chain = chain
        self.timestamp = timestamp
        self.filterresult = filterresult
        self.eventlength = eventlength
        self.trigger_type = trigger_type
        self.trigger_engine = trigger_engine
        self.iq = iq
        self.pileup = pileup
        self.samples = samples


metadata_dtype = np.dtype(
    [
        ("chain", np.int32),
        ("channel", np.int32),
        ("timestamp", np.int64),
        ("filterresult", np.int32),
        ("eventlength", np.int32),
        ("pileup", bool),
        ("iq", bool),
        ("trigger_type", np.int32),
        ("trigger_engine", np.int32),
    ]
)


def extract_flags(register):
    pileup = bool(register & 0x0001)
    iq = bool(register & 0x0002)
    trigger_type = np.int32((register & 0x0004) >> 2)
    trigger_engine = np.int32((register & 0x0038) >> 3)
    return pileup, iq, trigger_type, trigger_engine


def convert_to_iq(samples):
    samples_q = [np.int16((sample >> 16) & 0xFFFF) for sample in samples]
    samples_i = [np.int16(sample & 0xFFFF) for sample in samples]
    cdata = np.empty(len(samples_i), dtype=np.complex64)
    cdata.real = samples_i
    cdata.imag = samples_q
    return cdata


def extract_events(raw_data):
    """extracts individual events out of the data stream"""
    headerlength = 6

    event_startindex = 0
    eventlist = []
    while event_startindex < len(raw_data):
        try:
            channel = np.int16(raw_data[event_startindex] & 0xFFFF)
            chain = np.int16((raw_data[event_startindex] >> 16) & 0xFF)
            timestamp_msb = np.uint64(raw_data[event_startindex + 1]) << 32
            timestamp_lsb = np.uint64(np.uint32(raw_data[event_startindex + 2]))
            timestamp = timestamp_msb | timestamp_lsb
            eventlength = np.int32(raw_data[event_startindex + 4]) - 2
            pileup, iq, trigger_type, trigger_engine = extract_flags(
                raw_data[event_startindex + 5]
            )
            samples = []
            for j in range(eventlength):
                if (event_startindex + headerlength + j) < len(raw_data):
                    samples.append(raw_data[event_startindex + headerlength + j])
            if iq:
                filterresult = np.int16((raw_data[event_startindex + 3]) & 0xFFFF)
                # filterresult_q = np.int16((raw_data[i*eventsamples+2] >> 16) & 0xffff)
                samples = convert_to_iq(samples)
            else:
                filterresult = np.int32(raw_data[event_startindex + 3])
            eventlist.append(
                Event(
                    channel,
                    chain,
                    timestamp,
                    filterresult,
                    eventlength,
                    iq,
                    pileup,
                    trigger_type,
                    trigger_engine,
                    samples,
                )
            )

            event_startindex = event_startindex + headerlength + eventlength
        except:
            break

    return eventlist


def organize_events(raw_data):
    eventlist = extract_events(raw_data)
    events_metadata = []
    events_samples = []

    for event in eventlist[:-1]:
        metadata = [
            event.chain,
            event.channel,
            event.timestamp,
            event.filterresult,
            event.eventlength,
            event.iq,
            event.trigger_type,
            event.trigger_engine,
            event.pileup,
        ]
        events_metadata.append(metadata)
        samples = event.samples.real + 1j * event.samples.imag
        events_samples.append(samples.tolist())
    return events_metadata, events_samples
