# Built-in libraries
#
import math
import numpy   as np
import pandas  as pd
from   typing  import Tuple
from   typing  import List
from   typing  import Dict
from   typing  import Optional

# Scipy libraries
#
from   scipy.signal.windows import get_window
from   scipy.fftpack        import fft
from   scipy.stats          import skew
from   scipy.stats          import kurtosis

# Visualization libraries
#
import matplotlib.pyplot    as plt





def signal_to_units(signal: np.array, units: str = None):
    """Converts the acceleration (g) signal, to the required units."""

    if units in ["mmps2", "mmps", "mm"]:
        return signal * 9.80665 * 1000
    if units in ["um"]:
        return signal * 9.80665 * 1000 * 1000
    if units in ["mil"]:
        return signal * 9.80665 * 1000 * 25.4 * 1000
    if units in ["inch", "inchps", "inchps2"]:
        return signal * 9.80665 * 1000 * 25.4
    return signal




def integration(
    signal: np.array,
    sample_rate: int,
    order: int = 1,
    f_lo   = None,
    f_hi   = None,
    winlen = 0.1,
    unwin  = True,
):
    """
    Numerically integrate a time series in the frequency domain.

    This function integrates a time series in the frequency domain using
    'Omega Arithmetic', over a defined frequency band.

    Parameters
    ----------
    signal : array_like
        Input time series.
    sample_rate : int
        Sampling rate (Hz) of the input time series.
    f_lo : float, optional
        Lower frequency bound over which integration takes place.
        Defaults to 0 Hz.
    f_hi : float, optional
        Upper frequency bound over which integration takes place.
        Defaults to the Nyquist frequency ( = sample_rate / 2).
    order : int, optional
        Number of times to integrate input time series a. Can be either
        0, 1 or 2. If 0 is used, function effectively applies a 'brick wall'
        frequency domain filter to a.
        Defaults to 1.
    winlen : int, optional
        Number of seconds at the beginning and end of a file to apply half a
        Hanning window to. Limited to half the record length.
        Defaults to 0.1 second.
    unwin : Boolean, optional
        Whether or not to remove the window applied to the input time series
        from the output time series.

    Returns
    -------
    out : complex ndarray
        The zero-, single- or double-integrated acceleration time series.

    Versions
    ----------
    1.1 First development version.
        Uses rfft to avoid complex return values.
        Checks for even length time series; if not, end-pad with single zero.
    1.2 Zero-means time series to avoid spurious errors when applying Hanning
        window.

    """

    signal = signal - signal.mean()  # Convert time series to zero-mean
    if np.mod(signal.size, 2) != 0:  # Check for even length time series
        odd = True
        signal = np.append(signal, 0)  # If not, append zero to array
    else:
        odd = False

    if f_lo == None:  # Set upper frequency bound if not specified
        f_lo = 0.0
    if f_hi == None:  # Set lower frequency bound if not specified
        f_hi = 1.0e12

    f_hi = min(sample_rate / 2, f_hi)      # Upper frequency limited to Nyquist

    winlen = min(signal.size / 2, winlen)  # Limit window to half record length

    ni = signal.size  # No. of points in data (int)
    nf = float(ni)  # No. of points in data (float)
    sample_rate = float(sample_rate)  # Sampling rate (Hz)
    df = sample_rate / nf  # Frequency increment in FFT
    stf_i = int(f_lo / df)  # Index of lower frequency bound
    enf_i = int(f_hi / df)  # Index of upper frequency bound

    window = np.ones(ni)  # Create window function
    es = int(winlen * sample_rate)  # No. of samples to window from ends
    edge_win = np.hanning(es)  # Hanning window edge
    window[: int(es / 2)] = edge_win[: int(es / 2)]
    window[-int(es / 2) :] = edge_win[-int(es / 2) :]
    signal_w = signal * window

    fft_spec_signal = np.fft.rfft(signal_w)  # Calculate complex FFT of input
    fft_freq = np.fft.fftfreq(ni, d=1 / sample_rate)[: int(ni / 2 + 1)]

    w = 2 * np.pi * fft_freq  # Omega
    iw = (0 + 1j) * w  # i*Omega

    mask = np.zeros(int(ni / 2 + 1))  # Half-length mask for +ve freqs
    mask[stf_i:enf_i] = 1.0  # Mask = 1 for desired +ve freqs

    if order == 2:    # Double integration
        fft_spec = -fft_spec_signal * w / (w + np.finfo(float).eps) ** 3
    elif order == 1:  # Single integration
        fft_spec = fft_spec_signal * iw / (iw + np.finfo(float).eps) ** 2
    elif order == 0:  # No integration
        fft_spec = fft_spec_signal

    fft_spec *= mask  # Select frequencies to use

    out_w = np.fft.irfft(fft_spec)  # Return to time domain

    if unwin:
        out = (
            out_w * window / (window + np.finfo(float).eps) ** 2
        )  # Remove window from time series
        out = out[es:-es]
    else:
        out = out_w

    if odd:  # Check for even length time series
        return out[:-1]  # If not, remove last entry
    else:
        return out


    
    
    
def calculate_fft(
    signal:      np.array,
    sample_rate: int,
    no_of_lines: int = 400,
    averages:    int = 1,
    window:      str = "hann",
    overlapping: float = 0.0):
    """
    Calculate the Fast Fourier Transform of a signal.

    This method supports overlapping and averages which are not covered by
    default (scipy) methods.

    Parameters
    ----------
    signal : array_like
        Input time series.
    sample_rate : int
        Sampling rate (Hz) of the input time series.
    no_of_lines : int
        Spectrum resolution (default 400)
    averages : int
        Number of averages to combine (default 1 -- no averaging)
    window : str
        Window type (options hann, hamming, flattop -- default hann)
    overlapping : float
        Window overlapping fraction (0-1) (default 0)
    winlen : int
        Number of seconds at the beginning and end of a file to apply half a
        Hanning window to. Limited to half the record length.
        Defaults to 0.1 second.

    References
    ----------
    1 : https://blog.prosig.com/2011/08/30/understanding-windowing-and-overlapping-analysis/
    """
    error = None

    chunk_size      = no_of_lines * 2
    overlap_samples = int(round(overlapping * chunk_size))
    overlap_offset  = chunk_size - overlap_samples

    averages_requested = averages
    averages = min((len(signal) - overlap_samples) // overlap_offset, averages)
    if averages == 0:
        return [], [], "The requested number of lines is too high for this signal."

    if averages != averages_requested:
        error = "The requested averages number is too high for this signal. \
            Averages adjusted to {}. \
            You may want to reduce the number of lines.".format(
            averages
        )
        
    fft_results = []
    for i in range(averages):
        start = overlap_offset * i
        end   = start + chunk_size
        chunk = signal[start:end]

        if window:
            windower = get_window(window, chunk_size, False)
            chunk = chunk * windower

        fft_results.append(2.0 / chunk_size * np.abs(fft(chunk)[0 : chunk_size // 2]))

    yf = np.sum(fft_results, axis=0) / averages
    xf = dict(
        spacing = "linspace",
        start   = 0.0,
        stop    = 1.0 / (2.0 / sample_rate),
        size    = chunk_size // 2,
    )

    return xf, yf, error





def to_spectrums(
    file_path: str,
    integration_order: int = 0,
    high_pass: float = None,
    low_pass: float = None,
    no_of_lines: int = 400,
    averages: int = 1,
    window: str = "hann",
    overlapping: float = 0.0,
    units: str = None,
) -> Tuple[List[Dict[str, np.array]], Optional[str]]:
    """
    Converts a Mechbase file, to a tuple with an array of spectrums and an error message.

    Parameters
    ----------
        file_path : str
            The Mechbase file (may contain multiple signals)
        integration_order : int
            Integration order (default 0, no integration)
        high_pass : float
            Lower frequency bound over which integration takes place.
            Defaults to 0 Hz.
        low_pass : float
            Upper frequency bound over which integration takes place.
            Defaults to the Nyquist frequency ( = sample_rate / 2).
        no_of_lines : int
            Spectrum resolution (default 400)
        averages : int
            Number of averages to combine (default 1)
        window : str
            Window type (options hann, hamming, flattop -- default hann)
        overlapping : float
            Window overlapping fraction (0-1) (default 0)
        units : str
            Measuring units to use (default None)
    Returns
    -------
    out : Tuple ( array_like of complex ndarrays, error)
        array_like of complex ndarrays: the spectrums
        error: Αveraging arguments should agree with the signal resolution.
            If not the user is notified and is given instructions.
    """

    header, signals = convert(file_path)

    spectrums = []
    error = None
    for signal in signals:
        signal = signal_to_units(signal, units)
        signal = integration(
            signal,
            header.sample_rate,
            f_lo=high_pass,
            f_hi=low_pass,
            order=integration_order,
        )

        fft_x, fft_y, fft_error = calculate_fft(
            signal,
            sample_rate=header.sample_rate,
            no_of_lines=no_of_lines,
            averages=averages,
            window=window,
            overlapping=overlapping,
        )

        spectrums.append(dict(x=fft_x, y=fft_y))
        error = fft_error

    return spectrums, error





def calculate_RMS( x ):
    """
    Calculate RMS - Root mean square of an array

    """
    try:
        return ( math.sqrt(np.mean(np.square(x))) )
    except:
        return ( np.NaN )


def calculate_Peak_to_Peak(x):
    """
    Calculate Peak to Peak

    """        
    try:        
        return ( max(x) - min(x) )
    except:
        return ( np.NaN )


def calculate_Crest_Factor(x):
    """
    Calculate Crest Factor

    """
    try:        
        RMS          = math.sqrt(np.mean(np.square(x)))
        Max          = np.max( x )
        Min          = np.min( x )
        #
        Zero_to_Peak = max(map(abs, [Max, Min]))
        Crest_factor = Zero_to_Peak / RMS
        
        return ( Crest_factor )
    except:
        return ( np.NaN )
    
    
    
# Descriptive statistics
def Descriptive_Statistics( Signal ):
    """
    Print desciptive statistics about a signal
    RMS
    Min
    Max
    Mean
    std
    Peak-2-Peak
    Zero-2-Peak
    Crest factor
    """
    
    RMS          = calculate_RMS( Signal )
    Max          = np.max( Signal )
    Min          = np.min( Signal )
    Mean         = np.mean( Signal )
    STD          = np.std( Signal )
    Skew         = skew( Signal )
    Kurtosis     = kurtosis( Signal )
    #
    Peak_to_Peak = Max - Min
    Zero_to_Peak = max(map(abs, [Max, Min]))
    Crest_factor = Zero_to_Peak / RMS

    print('[INFO] Min:          %10.4f' % Min)
    print('[INFO] Max:          %10.4f' % Max)
    print('[INFO] Mean:         %10.4f' % Mean)
    print('[INFO] STD:          %10.4f' % STD)
    print('[INFO] Skew:         %10.4f' % Skew)
    print('[INFO] Kurtosis:     %10.4f' % Kurtosis)
    print()
    #
    #
    print('[INFO] RMS:          %10.4f' % RMS)
    print('[INFO] Peak to peak: %10.4f' % (Max - Min))
    print('[INFO] Zero to peak: %10.4f' % Zero_to_Peak)
    print('[INFO] Crest factor: %10.4f' % Crest_factor)
    print('\n')

    return (Min, Max, Mean, STD, Skew, Kurtosis, RMS, (Max - Min), Zero_to_Peak, Crest_factor)

    
# Descriptive statistics
def Descriptive_Statistics_Comparison( Signal1, Signal2 ):
    """
    Print desciptive statistics about two signals
    RMS
    Min
    Max
    Mean
    std
    Peak-2-Peak
    Zero-2-Peak
    Crest factor
    """
    
    # Descriptive statistics for the 1st Signal
    #
    RMS_1          = calculate_RMS( Signal1 )
    Max_1          = np.max( Signal1 )
    Min_1          = np.min( Signal1 )
    Mean_1         = np.mean( Signal1 )
    STD_1          = np.std( Signal1 )
    Skew_1         = skew( Signal1 )
    Kurtosis_1     = kurtosis( Signal1 )
    #
    Peak_to_Peak_1 = Max_1 - Min_1
    Zero_to_Peak_1 = max(map(abs, [Max_1, Min_1]))
    Crest_factor_1 = Zero_to_Peak_1 / RMS_1

    
    
    # Descriptive statistics for the 2nd Signal
    #
    RMS_2          = calculate_RMS( Signal2 )
    Max_2          = np.max( Signal2 )
    Min_2          = np.min( Signal2 )
    Mean_2         = np.mean( Signal2 )
    STD_2          = np.std( Signal2 )
    Skew_2         = skew( Signal2 )
    Kurtosis_2     = kurtosis( Signal2 )
    #
    Peak_to_Peak_2 = Max_2 - Min_2
    Zero_to_Peak_2 = max(map(abs, [Max_2, Min_2]))
    Crest_factor_2 = Zero_to_Peak_2 / RMS_2
    
    
    
    
    print('[INFO] Min:          %10.4f | %10.4f' % (Min_1, Min_2))
    print('[INFO] Max:          %10.4f | %10.4f' % (Max_1, Max_2))
    print('[INFO] Mean:         %10.4f | %10.4f' % (Mean_1, Mean_2))
    print('[INFO] STD:          %10.4f | %10.4f' % (STD_1, STD_2))
    print('[INFO] Skew:         %10.4f | %10.4f' % (Skew_1, Skew_2))
    print('[INFO] Kurtosis:     %10.4f | %10.4f' % (Kurtosis_1, Kurtosis_2))
    print()
    #
    #
    print('[INFO] RMS:          %10.4f | %10.4f' % (RMS_1, RMS_2))
    print('[INFO] Peak to peak: %10.4f | %10.4f' % (Peak_to_Peak_1, Peak_to_Peak_2))
    print('[INFO] Zero to peak: %10.4f | %10.4f' % (Zero_to_Peak_1, Zero_to_Peak_2))
    print('[INFO] Crest factor: %10.4f | %10.4f' % (Crest_factor_1, Crest_factor_2))
    print('\n')
    
