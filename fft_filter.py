import numpy as np
import scipy.fftpack as fftpack


def fft_filter(signal, freq_min, freq_max, fps):
    fft = fftpack.fft(signal, axis=0)
    frequencies = fftpack.fftfreq(signal.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    return fft, frequencies
