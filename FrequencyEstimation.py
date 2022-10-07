# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
# Built-in libraries
#
import numpy as np


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
# Scipy
#
from scipy.signal import blackmanharris
from scipy.signal import periodogram
from scipy.signal import correlate


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
# User libraries
#
from utils.parabolic           import *





def freq_from_crossings(sig, fs):
    """
    Estimate frequency by counting zero crossings
    """
    # Find all indices right before a rising-edge zero crossing
    indices = np.nonzero((sig[1:] >= 0) & (sig[:-1] < 0))[0]

    # Naive (Measures 1000.185 Hz for 1000 Hz, for instance)
    # crossings = indices

    # More accurate, using linear interpolation to find intersample
    # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]

    # Some other interpolation based on neighboring points might be better.
    # Spline, cubic, whatever

    return ( fs / np.mean(np.diff(crossings)) )


def freq_from_fft(sig, fs):
    """
    Estimate frequency from peak of FFT
    """
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    f = np.fft.rfft( windowed )

    # Find the peak and interpolate to get a more accurate peak
    i      = np.argmax(abs(f))  # Just use this for less-accurate, naive version
    true_i = parabolic( np.log(abs(f)), i)[0]

    # Convert to equivalent frequency
    return ( fs * true_i / len(windowed) )


def freq_from_autocorr(sig, fs):
    """
    Estimate frequency using autocorrelation
    """
    # Calculate autocorrelation and throw away the negative lags
    corr = correlate(sig, sig, mode='full')
    corr = corr[len(corr)//2:]

    # Find the first low point
    d     = np.diff(corr)
    start = np.nonzero(d > 0)[0][0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak   = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px




def freq_from_periodogram(x, fs):
    # get the frequency and spectrum
    #
    f, Pxx = periodogram(x, fs = fs, window='hanning', scaling='spectrum')

    # Calculate frequency
    #
    freq = f[Pxx==max(Pxx)];

    return (freq)




