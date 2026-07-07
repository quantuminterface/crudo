#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Davide Quaranta
Date: 2025-10-31
Purpose: Convert signal amplitude to dBm units based on calibration parameters.

Description:
    This script defines a utility function to convert a given amplitude value
    into power expressed in dBm. The conversion uses a logarithmic relationship
    with calibration coefficients that depend on whether an amplifier is used.
"""

import numpy as np

# Volts to ADC counts conversion
# need to computed from ADC calibration
V_2_ADC = 421852


def ADC_to_dBm(data, adc_counts_per_v):
    V_1mW = np.sqrt(0.001 * 50)
    V_1mW_ADC = adc_counts_per_v * V_1mW
    PdBm = 10 * np.log10(np.mean(np.abs(data)) ** 2 / V_1mW_ADC**2)
    return PdBm


def amplitude_to_dBm(a, ampli="on"):
    """
    Convert amplitude to dBm based on calibration constants.
    Need to be recomputed from board to board possibly.

    Parameters
    ----------
    a : float or array-like
        Input amplitude value(s).
    ampli : str, optional
        Amplifier state. Use 'on' if the amplifier is enabled, 'off' otherwise.
        Default is 'on'.

    Returns
    -------
    dBm : float or ndarray
        Power in dBm corresponding to the input amplitude.
    """
    m = 20
    c = 10.83 if ampli == "on" else -18.60

    dBm = m * np.log10(a) + c
    return dBm


def dBm_to_mW(P_dBm):
    return 10 ** (P_dBm / 10)


def dBm_to_volts(P_dBm, Z=50):
    P_W = 1e-3 * dBm_to_mW(P_dBm)
    V = np.sqrt(P_W * Z)
    return V


def mW_to_dBm(P_mW):
    return 10 * np.log10(P_mW)


def volts_to_dBm(V, Z=50):
    P_mW = 1000 * (V**2 / Z)
    P_dBm = 10 * np.log10(P_mW)
    return P_dBm
