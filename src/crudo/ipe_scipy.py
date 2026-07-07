import numpy as np
import scipy.signal
from scipy.optimize import curve_fit


def get_window(name, size):
    """
    Returns the corresponding window as np array given a name of the window

    Args:
        size: (int) size of the window
        name: (string) name of the window

    Returns:
        np array of the window with a given size

    """
    w = {
        "rectangular": np.ones(size),
        "hamming": np.hamming(size),
        "hanning": np.hanning(size),
        "blackman": np.blackman(size),
        "blackmanharris": scipy.signal.windows.blackmanharris(size),
        "bartlett": np.bartlett(size),
        "kaiser": np.kaiser(size, 10),
    }
    try:
        return w[name]
    except KeyError as error:
        raise Exception from error(
            f"{name} is not a supported window. Use {str.join(w.keys())}"
        )


def gaussian(x_vector, amplitude, mu, sigma):
    """
    Produces a gaussian fit onto the data
    """
    return amplitude * np.exp(-0.5 * ((x_vector - mu) / sigma) ** 2)


def fit_gaussian(f_arr, y_arr, index, fit_width):
    """
    fit a gaussian peak to the data (x, y)
    Source : https://dspguru.com/dsp/howtos/how-to-interpolate-fft-peak/

    Args:
        f_arr: mesh points of spectrum
        y_arr: spectral amplitude array
        index: (int) index of local maximum in the y-array
        fit_width: (int) numbers of samples contributing to the fit

    Returns:
        A: Amplitude of fitted gaussian
        mu: mean value of distribution
        sigma: standard deviation
    """
    df = f_arr[1] - f_arr[0]
    hw = fit_width // 2  # half width
    w_odd = fit_width % 2

    # Select fit range
    x_vector = f_arr[index - hw : index + hw + w_odd]
    y_vector = y_arr[index - hw : index + hw + w_odd]
    try:
        p0 = (y_vector[hw], x_vector[hw], df * hw)
        (ampl, mu, sigma), _ = curve_fit(gaussian, x_vector, y_vector, p0=p0)
    except RuntimeError as error:
        print("Fitting failed")
        raise RuntimeError from error
    return ampl, mu, sigma


def interpolate_peaks(f_space, spect, height, peak_width=10, debug=False):
    """Find carrier frequencies by fitting peaks of the passed Spectrum in a small interval

    Args:
        f_space      : (numpy array) frequencies of the given spectrum
        Spect       : (numpy array) amplitude spectrum
        height      : (float) required height of peaks
        PeakWidth   : number of samples used for the peak fitting.
                        further peaks within this range will be ignored.
        debug       : print and plot properties
    Returns:
        PeakFreq    : (numpy array) of peak center frequencies
    """
    # Auto Set threshold/height parameter for find_peaks
    print(f"Height : {height}")
    peak_index, peak_prop = scipy.signal.find_peaks(
        spect, height=height, distance=peak_width // 2
    )
    if debug:
        print(f"PeakIdx : {str(peak_index)}")
        print(f"F(Peaks): {str(f_space[peak_index])}")
        print(f"PeakProp: {str(peak_prop)}")
        f_0 = f_space[peak_index[0]]
        # x_vector = np.linspace(0.9 * f_0, 2.0 * f_0)

    peak_freq = np.zeros(len(peak_index))
    for k, idx in enumerate(peak_index):
        if debug:
            peak, peak_freq[k], sigma = fit_gaussian(
                f_space, spect, idx, peak_width, debug
            )
            print(f"A : {peak:.2g}\t| f : {peak_freq[k]:.3g}\t| sigma : {sigma:.3g}")
            # plt.plot(x_vector, gaussian(x_vector, peak, peak_freq[k], sigma))
        else:
            peak_freq[k] = fit_gaussian(f_space, spect, idx, peak_width)
    return peak_freq


def fft(data, window_name="rectangular"):
    """
    Calculates the fft on the data with a given window

    Args:
        data: array containing the input data
        window_name: (str) name of the window

    Returns:
        FFT of the input data
    """
    size = len(data)
    window = get_window(window_name, size)
    return np.fft.fft((np.array(data) - np.mean(data)) * window) / np.sum(window)


def normalized_fft(input_signal, d_t, window_name):
    """
    generates the normalized one sided power spectrum for a given real signal weighted by the given window function.
    if a 2d array is passed, the fft is computed on the subarrays and averaged
    """
    signal_length = input_signal.shape[-1]
    window = get_window(window_name, signal_length)
    window_sum = np.sum(window**2)
    # Adapt Dimensions
    if len(input_signal.shape) == 2:
        depth = input_signal.shape[0]
        window = np.outer(np.ones(depth), window)
    elif len(input_signal.shape) > 2:
        raise Exception(
            f"Can only handle numpy arrays with one or two dimensions, given array is {len(input_signal.shape)}-d"
        )

    # f_space = scipy.fft.fftfreq(signal_length, d=d_t)[1 : signal_length // 2]
    f_space = scipy.fft.fftfreq(signal_length, d=d_t)
    f_space = scipy.fft.fftshift(f_space)
    p_spect = (2 * d_t / window_sum) * (
        np.abs(scipy.fft.fft(input_signal * window, axis=-1)) ** 2
    )  # calculate the spectrum
    if len(input_signal.shape) == 2:
        p_spect = np.mean(p_spect, axis=0)
    # p_spect = p_spect[1 : signal_length // 2]
    p_spect = scipy.fft.fftshift(p_spect)
    return f_space, p_spect


def ddc(input_signal, fs, fm):
    """
    Performs digital downconversion on the input signal

    Args:
        input_signal: Signal to be downmixed in complex form
        fs: sample rate of the input signal
        fm: mixing frequency
    """
    signal_length = len(input_signal)

    # Time vector
    t = np.arange(0, signal_length / fs, 1 / fs)

    output_signal = input_signal * np.exp(-1j * 2 * np.pi * fm * t)
    return output_signal


def raw_to_deg(data):
    """
    calculates the phase in degrees from int16 values.
    Expects a list of data traces
    """
    if isinstance(data, list):
        out = []
        for trace in data:
            raw = np.array(trace)
            out.append(np.rad2deg(raw / float(2**13)))
        return out
    elif isinstance(data, np.ndarray):
        return np.rad2deg(data / float(2**13))


def raw_to_rad(data):
    """
    calculates the phase in degrees from int16 values.
    Expects a list of data traces
    """
    if isinstance(data, list):
        out = []
        for trace in data:
            raw = np.int16(trace)
            out.append(raw / float(2**13))
        return out
    elif isinstance(data, np.ndarray):
        return data / float(2**13)


def raw_to_phi0(data, unwrap=True):
    """
    converts int16 to Phi0 values.
    Expects a list of data traces or a numpy.ndarray
    """
    if isinstance(data, list):
        out = []
        for trace in data:
            raw = np.int16(trace)
            out.append(raw / float(2**14) / np.pi)
        return out
    elif isinstance(data, np.ndarray):
        return data / float(2**14) / np.pi


def deinterleave(interleaved_data, n_channels):
    """
    returns deinterleaved data as list of np.arrays
    """
    deinterleaved_data = [
        np.array(interleaved_data[k::n_channels]) for k in range(n_channels)
    ]
    return deinterleaved_data


def deinterleave_ndarray(interleaved_data, n_channels):
    """
    returns deinterleaved data as np.ndarray of shape (N_channels, len(interleaved_data)//N_channels)
    """
    deinterleaved = np.asarray(
        [np.array(interleaved_data[k::n_channels]) for k in range(n_channels)]
    )
    return deinterleaved
