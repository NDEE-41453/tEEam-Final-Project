import numpy as np
from scipy import signal
#This function performs the estimate of the carrier phase offset (using the methods described in the writeup) and correct the phase offset in the packet data samples

# Input:

# packet_data: packet data samples with frequency and phase offset, \hat{x}[n] in the writeup

# Digital_LO: locally generated complex LO for frequency error correction, \hat{LO}[n] in the writeup

# payload_start: sample index denoting the beginning of the payload and end of the preamble

# preamble_length: numebr of symbols in preamble

# samples_perbit: number of sample per symbol

# Output:

# phase_corrected_packet: an array contains packet data samples with both carrier frequency and phase offsets corrected


def phase_sync(packet_data, Digital_LO, payload_start, preamble_length, samples_perbit):

        #First, correct the frequency offset from packet_data
        #Hint: You may find np.multiply() useful
        packet_data_freq_corrected = np.multiply(packet_data, Digital_LO)
        
        #remove the BB voltage offset at the payload due to non-idealities
        packet_data_freq_corrected = packet_data_freq_corrected - np.mean(packet_data_freq_corrected[payload_start:])

        #Extract the preamble only from the corrected packet (preamble + payload)
        preamble = packet_data_freq_corrected[0:int(preamble_length*samples_perbit)]

        #Extract carrier phase offset using "preamble" above
        #Hint: You may find np.angle() useful
        angles = np.angle(preamble)

        #Averaging for better estimate
        phase_estimated = np.mean(angles)

        #Correct the carrier phase offset in "packet_data_freq_corrected" to obtain signal samples with both frequency and phase offsets corrected
        #Hint: You may find np.multiply() helpful and you may want to construct a complex exponential using "phase_estimated"
        phase_corrected_packet = np.multiply(packet_data_freq_corrected, np.exp(-1j*phase_estimated))

        return phase_corrected_packet
