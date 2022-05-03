import numpy as np
from scipy import signal
import pyfftw
import scipy.fftpack
from scipy.signal import ZoomFFT

def fft_baseband_offset(baseband, fs, L):
    # FFT
    spectrum = np.abs(pyfftw.interfaces.scipy_fftpack.fft(baseband, L))
    spectrum = np.fft.fftshift(spectrum)
    # Return max point
    peak = np.argmax(spectrum)
    offset = (peak - L/2)/L * fs
    
    return offset

def zoomfft_baseband_offset_bounds(baseband, fs, bottom, top, L):
    transform = ZoomFFT(len(baseband), [bottom, top], L, fs=fs)
    spectrum = np.abs(transform(baseband))
    
    peak = np.argmax(spectrum)
    offset = (top - bottom)*peak/(L-1) + bottom
    
    return offset

def combined_baseband_offset(baseband, fs, L):
    freq_est = fft_baseband_offset(baseband, fs, L[0])
    offset_zoom = zoomfft_baseband_offset_bounds(baseband, fs, freq_est-(1.1)*fs/L[0], freq_est+(1.1)*fs/L[0], L[1])

    return offset_zoom

#This function performs the estimation of the carrier frequency offset using the FFT

#Input:

#segments_of_data_for_fft: an array contains the signal samples to be used as the input for FFT
#num_fft_point: number of points for FFT (this affects the resolution of the estimate)
#fs_in: ADC input sampling frequency

#Output:

#coarse_frequency: This is the frequency offset estimated


def freq_sync(segments_of_data_for_fft, num_fft_point, fs_in):
        num_fft_point = int(num_fft_point)
        # Output: 
        #Obtain the estimated carrier frequency offset
        #coarse_frequency = fft_baseband_offset(segments_of_data_for_fft, fs_in, num_fft_point)
        coarse_frequency = combined_baseband_offset(baseband=segments_of_data_for_fft, fs=fs_in, L=[num_fft_point, num_fft_point])
        
        return coarse_frequency
