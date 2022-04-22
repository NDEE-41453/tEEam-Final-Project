import numpy as np
from scipy import signal
import pyfftw
import scipy.fftpack

#This function performs the estimation of the carrier frequency offset using the FFT

#Input:

#segments_of_data_for_fft: an array contains the signal samples to be used as the input for FFT
#num_fft_point: number of points for FFT (this affects the resolution of the estimate)
#fs_in: ADC input sampling frequency

#Output:

#coarse_frequency: This is the frequency offset estimated


def freq_sync(segments_of_data_for_fft, num_fft_point, fs_in):

        spectrum = abs(pyfftw.interfaces.scipy_fftpack.fft(segments_of_data_for_fft, num_fft_point))

        # FFT shift so that DC component (zero freq) is in the middle of the array of FFT result                
        spectrum = np.fft.fftshift(spectrum)

        # Hint: You may find np.argmax() useful
        peak_position = np.argmax(spectrum)

        # Output: 

        #Obtain the estimated carrier frequency offset
        coarse_frequency = (peak_position-len(spectrum)/2) / len(spectrum) * fs_in
        
        return coarse_frequency
