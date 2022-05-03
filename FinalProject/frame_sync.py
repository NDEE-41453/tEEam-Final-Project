import numpy as np
from scipy import signal
#This function performs the crosscorrelation / matched-filtering based frame sychronization by
#match filtering the received, zero-mean / ac-coupled data with the matched filter response
#The output is the corrcorrelation / Matched filtering result that is then fed to the peak search
#function which returns the location(s) of the payload beginning

#Input:
# data_bb_ac: Zero-mean input data that contains the preamble(s)
# known_preamble_ac: This array contains samples of the known preamble

#Output:
# crosscorr: Result of the matched filtering


def frame_sync(data_bb_ac, known_preamble_ac):
        
        matched_filter_coef = np.flip(known_preamble_ac)
        crosscorr = signal.fftconvolve(data_bb_ac,matched_filter_coef)

        return crosscorr
