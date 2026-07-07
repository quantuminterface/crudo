#!/usr/bin/env python3
"""
Module for storing VNA sweeps, resonance circle data, and events into HDF5 files.

Updated per chef’s feedback:
  - VNA sweeps now store S21 as a complex number (I + 1j*Q).
  - Resonance circles: each resonator’s data (multiple circles) is stored in its own subgroup,
    with the circle measurements stored as rows (complex data computed as I + 1j*Q) and with the
    corresponding frequency matrix.
  - Events: the function accepts a single NumPy array (from event_parse) that already contains
    the full metadata. No info_dtype is defined here.
"""

import os
import numpy as np
import h5py
from crudo import event_parse


# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def get_or_create_group(h5file, group_path):
    """
    Given an open h5py.File object and a group_path like "VNA_0/Measurements",
    create (if needed) and return the group.
    """
    grp = h5file
    for subgrp in group_path.strip("/").split("/"):
        if subgrp in grp:
            grp = grp[subgrp]
        else:
            grp = grp.create_group(subgrp)
    return grp


def create_or_overwrite_dataset(group, ds_name, data, dtype=None, shape=None):
    """
    In the given group, if a dataset with name ds_name exists, delete it.
    Then create a new dataset with the given data.
    If shape is provided, it will be used (otherwise data.shape is used).
    """
    if ds_name in group:
        del group[ds_name]
    if shape is None:
        shape = data.shape
    group.create_dataset(
        ds_name, data=data, shape=shape, dtype=dtype if dtype else data.dtype
    )


# --------------------------------------------------------------------
# 1. VNA Sweeps
# --------------------------------------------------------------------
def vna_sweep_to_hdf5(h5_path, group_name, frequencies, amplitude, phase):
    """
    Compute I and Q from amplitude and phase (I = amp*cos(phase), Q = amp*sin(phase))
    and store the result as a single complex number (I + 1j*Q).

    Datasets written (under group_name):
      - S21: 1D complex array (np.complex128) with S21 = I + 1j*Q.
      - frequency: 1D float64 array with frequency values.
    """
    # Convert amplitude & phase to I and Q
    I = amplitude * np.cos(phase)
    Q = amplitude * np.sin(phase)
    # Compute complex S21
    s21 = I + 1j * Q

    # Ensure numpy arrays and proper dtype:
    frequencies = np.asarray(frequencies, dtype=np.float64)
    s21 = np.asarray(s21, dtype=np.complex128)

    if frequencies.shape != s21.shape:
        raise ValueError("Frequencies and computed S21 must have the same shape.")

    with h5py.File(h5_path, "a") as h5file:
        grp = get_or_create_group(h5file, group_name)
        create_or_overwrite_dataset(grp, "S21", s21, dtype=np.complex128)
        create_or_overwrite_dataset(grp, "frequency", frequencies, dtype="float64")
    print(
        f"VNA sweep data (S21 as complex numbers) written to {h5_path} under group {group_name}"
    )


# --------------------------------------------------------------------
# 2. Resonance Circles
# --------------------------------------------------------------------
def resonance_circle_to_hdf5(
    h5_path, group_name, resonance_frequencies, resonance_data_list
):
    """
    Store resonance circle data by merging all provided datasets into a single dataset.

    Parameters:
      h5_path             : Path to the HDF5 file.
      group_name          : Group name under which the combined data will be stored.
      resonance_data_list : List of resonance circle data. Each entry should be a tuple:
                             (frequencies, I_array, Q_array)
                             where each of these is a 2D numpy array with each row corresponding
                             to one circle measurement.

    The function merges all resonance data by vertically stacking:
      - "data"     : A 2D array of complex numbers computed as I + 1j*Q.
      - "frequency": A 2D array of float64 values corresponding to the frequencies.

    The merged datasets are stored directly under the specified group.
    """
    complex_data_list = []
    frequency_data_list = []

    for idx, (freq_arr, I_arr, Q_arr) in enumerate(resonance_data_list):
        # Ensure inputs are numpy arrays of type float64.
        freq_arr = np.asarray(freq_arr, dtype=np.float64)
        I_arr = np.asarray(I_arr, dtype=np.float64)
        Q_arr = np.asarray(Q_arr, dtype=np.float64)

        if not (freq_arr.shape == I_arr.shape == Q_arr.shape):
            raise ValueError(
                f"Resonance dataset {idx}: frequencies, I, and Q arrays must have the same shape."
            )

        # Compute complex data for each circle measurement.
        data_complex = I_arr + 1j * Q_arr

        complex_data_list.append(data_complex)
        frequency_data_list.append(freq_arr)

    # Merge all data by vertically stacking.
    # Each row corresponds to one circle measurement from any resonator.
    merged_data = np.vstack(complex_data_list)
    merged_frequency = np.vstack(frequency_data_list)

    with h5py.File(h5_path, "a") as h5file:
        grp = get_or_create_group(h5file, group_name)
        # Write the merged datasets directly under the group.
        create_or_overwrite_dataset(grp, "data", merged_data, dtype=np.complex128)
        create_or_overwrite_dataset(grp, "data_freq", merged_frequency, dtype="float64")
        create_or_overwrite_dataset(
            grp,
            "resonator",
            np.asarray(resonance_frequencies, dtype="float64"),
            dtype="float64",
        )

    print(
        f"Resonance circle data merged (data shape: {merged_data.shape}) and written to {h5_path} under group {group_name}."
    )


# --------------------------------------------------------------------
# 3. Events
# --------------------------------------------------------------------
def events_to_hdf5(h5_path, group_name, data):
    """
    Store events (as parsed by event_parse) into the HDF5 file.

    The function expects events_array to be a single numpy array with a structured dtype that
    contains all metadata and samples (e.g. a field "samples" holding the event samples).

    The entire events array is stored in a dataset named "data" under the given group.

    (Note: No info_dtype is defined here so that updates to the event metadata need only be made
    in event_parse.)
    """
    # events_array is assumed to already have the proper dtype (metadata fields, etc.)

    events_metadata, events_samples = event_parse.organize_events(data)
    event_samples = np.vstack(events_samples)
    event_metadata = np.vstack(events_metadata)

    with h5py.File(h5_path, "a") as h5file:
        grp = get_or_create_group(h5file, group_name)
        create_or_overwrite_dataset(grp, "data", event_samples, dtype=np.complex128)
        create_or_overwrite_dataset(grp, "info", event_metadata, dtype=np.float64)
    print(f"Events data written to {h5_path} under group {group_name}")
