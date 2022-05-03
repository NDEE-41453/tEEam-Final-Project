import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import signal
import matched_filtering
import pyfftw
import phase_sync

#Define the demodulation function

#The demodulation function takes the following arguments as inputs:

# a_in1:                        This array contains the received samples to be demodulated, samples here include both the preamble and the payload.

# samples_perbit:               This is the oversampling factor on the receiver side (= 4/3*oversampling factor of the transmitter), each symbol you would like to demodulate
#                               is represented by "samples_perbit" bits

# carrier_frequency:            This is the estimated frequency of the IF carrier in Hz

# carrier_phase:                This is the estimated phase of the IF carrier in radian

# payload_start:                This is the index of the received samples that indicates the beginning of the payload
#                               For example, if an array "a_in1" contains both the preamble and payload, the a_in1[payload_start] should be the first sample of the payload

# N                             This value is equal to the length of the payload in bits.

# Ts_in:                        This is the ADC input sampling period in sec, Ts_in = 1/fs_in

# channel_gain                  This is the gain of the channel, channel impulse response is simply modeled as g(t)= channel_gain*delta(t)

# pulse_shaping                 This is a string indicating whether pulse shaping is applied on the transmitter side

# scheme                        Modulation Scheme

#The demodulation function returns the following arguments as outputs:


#(with pulse_shaping == "On" )

# baseband_symbols              This is the array of baseband symbols


#(with pulse_shaping == "Off" )

# a_demodulated                 This is the bit array containing the demodulated payload



                
def coherent_demodulation(a_in1, samples_perbit, carrier_frequency, payload_start, N, fs_in, Ts_in, pulse_shaping, scheme, preamble_length, showplot):

        debugging = 1 # print out statistics for the developer

        t1 = time.time()

        if(scheme == "QPSK"):
                N = int(N/2)

        if(scheme == "16QAM"):
                N = int(N/4)

        payload_before_correction = a_in1[payload_start:(payload_start + N*samples_perbit)]

##        plt.plot(np.abs(a_in1))
##        plt.show()



##        preamble_ones_and_pay = a_in1[int(payload_start-180*samples_perbit):(payload_start-samples_perbit)]
##        k = np.arange(len(preamble_ones))
##
##        Digital_LO = np.exp(-1j*2*np.pi*carrier_frequency*(k*Ts_in))
##
##        preamble_freq_corrected = np.multiply(preamble_ones,Digital_LO)
##
##        angles = np.angle(preamble_freq_corrected)
##        phase_estimated = np.mean(angles)
        

        ones_length = preamble_length - 20
        

        payload_and_ones = a_in1[int(payload_start-ones_length*samples_perbit):(payload_start + N*samples_perbit)]
        k = np.arange(len(payload_and_ones))
        Digital_LO = np.exp(-1j*2*np.pi*carrier_frequency*(k*Ts_in))

        #Correct the frequency and then extract & correct the phase

        payload_and_ones_corrected = phase_sync.phase_sync(payload_and_ones, Digital_LO, payload_start, ones_length, samples_perbit)

# Save the frame sync result
        with open('phasesync_result.txt', 'w') as fs:
                fs.write('Phase Synchronization result:\n')
                fs.write('Packet Data: '+str(payload_and_ones)+'\n')
                fs.write('Digital LO: '+str(Digital_LO)+'\n')
                fs.write('Payload Start: '+str(payload_start)+'\n')
                fs.write('Preamble Length: '+str(ones_length)+'\n')
                fs.write('Oversampling Factor: '+str(samples_perbit)+'\n')
                fs.write('Phase Correct Length: '+str(payload_and_ones_corrected)+'\n')

        # save result to a numpy array as well
        np.savez('PhaseSync_data', payload_and_ones = payload_and_ones, Digital_LO = Digital_LO, payload_start = payload_start, \
                 ones_length = ones_length, samples_perbit = samples_perbit, payload_and_ones_corrected= payload_and_ones_corrected)

        payload_corrected = payload_and_ones_corrected[ones_length*samples_perbit:]


        baseband_signal_I_new = np.real(payload_corrected)
        baseband_signal_Q_new = np.imag(payload_corrected)


##(Outdated)
##        if freq_offset_positive:
##                baseband_signal_I =  - np.multiply(np.real(signal_before_correction),Digital_LO_I) -  np.multiply(np.imag(signal_before_correction),Digital_LO_Q)
##
##                baseband_signal_Q = - np.multiply(np.real(signal_before_correction),Digital_LO_Q) +  np.multiply(np.imag(signal_before_correction),Digital_LO_I)
##        else:
##                baseband_signal_I =  - np.multiply(np.real(signal_before_correction),Digital_LO_I) +  np.multiply(np.imag(signal_before_correction),Digital_LO_Q)
##
##                baseband_signal_Q = - np.multiply(np.real(signal_before_correction),Digital_LO_Q) -  np.multiply(np.imag(signal_before_correction),Digital_LO_I)
##
##        if(np.max(baseband_signal_I)>-np.min(baseband_signal_I)):
##                I_value = np.max(baseband_signal_I)
##        else:
##                I_value = np.min(baseband_signal_I)
##        if(np.max(baseband_signal_Q)>-np.min(baseband_signal_Q)):
##                Q_value = np.max(baseband_signal_Q)
##        else:
##                Q_value = np.min(baseband_signal_Q)
##
##        if(scheme=="OOK" or scheme=="BPSK"):
##                actual_phase = np.angle(I_value+1j*Q_value)
##        else:
##                actual_phase = 2*np.angle(I_value+1j*Q_value)
##                
##        baseband_signal_I_new = baseband_signal_I*np.cos(actual_phase) + baseband_signal_Q*np.sin(actual_phase)
##        baseband_signal_Q_new = baseband_signal_I*np.sin(actual_phase) - baseband_signal_Q*np.cos(actual_phase)

        #correct for the DC offset through AC coupling
##        if(scheme == "QPSK"):
##                baseband_signal_I_new = baseband_signal_I_new - np.mean(baseband_signal_I_new)
##                baseband_signal_Q_new = baseband_signal_Q_new - np.mean(baseband_signal_Q_new)
     
##        plt.plot(baseband_signal_I_new)
##        plt.plot(baseband_signal_Q_new)
##        plt.title('baseband after frequency and phase correction')
##        plt.show()


        
        



#Matched filtering
        if(pulse_shaping == "On"):

                #Use Matched filter receiver realization when pulse shaping is applied on the transmitter side

                alpha = 0.9 #roll-off factor of the RRC matched-filter

                L = 8

                #frame sync error detected
                if(len(baseband_signal_I_new)==0):
                        return 0,0

##                plt.plot(baseband_signal_I_new)
##                plt.title('I BB before MF')
##                plt.show()
##
##                plt.plot(baseband_signal_Q_new)
##                plt.title('Q BB before MF')
##                plt.show()
                        
                symbols_I = matched_filtering.matched_filtering(baseband_signal_I_new, samples_perbit, fs_in, alpha, L)
                
                symbols_Q = matched_filtering.matched_filtering(baseband_signal_Q_new, samples_perbit, fs_in, alpha, L)

##                plt.plot(symbols_I)
##                plt.title('I BB after MF')
##                plt.show()
##
##                plt.plot(symbols_Q)
##                plt.title('Q BB after MF')
##                plt.show()
                
##                np.random.seed(2020)
##
##                generated_segment = np.random.randint(0,2,20)
##
##                print(generated_segment)
                
                #Remove the L/2 samples at the beginning and L/2 samples at the end caused by group delay of the filter

                symbols_I = symbols_I[int(L/2):(len(symbols_I)-int(L/2))]

                symbols_Q = symbols_Q[int(L/2):(len(symbols_Q)-int(L/2))]
                

##                plt.plot(symbols_I)
##                plt.show()
                
##                plt.stem(symbols_I,use_line_collection=True)
##                plt.title('I channel samples at symbol rate (after MF)')
##                plt.show()
##
##                plt.stem(symbols_Q,use_line_collection=True)
##                plt.title('Q channel samples at symbol rate (after MF)')
##                plt.show()

                #Plot the received signal constellation
                if(showplot==True):
##                        plt.ylim((-0.8, 0.8))
##                        plt.xlim((-0.8, 0.8))
                        for i in range(len(symbols_I)):
                                plt.plot(symbols_I[i],symbols_Q[i], color='blue', marker='o', markersize=1)
                        plt.show()

##                plt.plot(np.abs(a_in1))
##                plt.title('packet before correction')
##                plt.show()
##
##                plt.stem(symbols_I,use_line_collection=True)
##                plt.title('I channel samples at symbol rate (after MF)')
##                plt.show()

##                plt.plot(np.real(payload_and_ones_freq_corrected))
##                plt.plot(np.imag(payload_and_ones_freq_corrected))
##                plt.title('angle: '+str(phase_estimated))
##                plt.show()
                
                channel_gain = np.max(symbols_I)/1.0

                if(scheme == "QPSK"):
                        channel_gain = np.mean(np.abs(symbols_I))/(2/3)

                symbols = np.array([symbols_I,symbols_Q])

                return symbols, channel_gain




