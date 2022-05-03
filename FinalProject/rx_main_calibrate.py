#Import ADALM Python libraries
import iio
import adi
import libm2k

#Import other dependencies
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import socket
import sys
import multiprocessing
import struct
from scipy import signal
import math

#Import Radioware RX BB DSP modules
import symbol_mod
import preamble_generator
import pulse_shaping
import mode_preconfiguration
import data_sink

#Import the HAL for ADALM (2000/Pluto)
import HAL

#Import the GUI/CLI parser modules
import UI


def getsamples(q1, Mode, fs_in,total_samples, debugging, same_machine, ftuned, samples_perbit, device, rx_gain):

        #Initialize the receiver device
        ain, spi_desc, sdr, dc_offset_data_I, dc_offset_data_Q = HAL.setup_rx_device_control(device, same_machine, ftuned, fs_in, total_samples, rx_gain)
##        print("I offset estimated: ", np.mean(dc_offset_data_I))
##        plt.plot(dc_offset_data_I)
##        plt.title('dc_offset_data_I')
##        plt.show()
        
        #Perform calibration based on collected data
        I_offset = np.mean(np.array(dc_offset_data_I))
        I_pp = np.max(np.array(dc_offset_data_I)) - np.min(np.array(dc_offset_data_I))
        
        Q_offset = np.mean(np.array(dc_offset_data_Q))
        Q_pp = np.max(np.array(dc_offset_data_Q)) - np.min(np.array(dc_offset_data_Q))

        IQ_gain_ratio = I_pp/Q_pp
        #Print out the calibration results
        print("I_offset: ", I_offset)
        print("Q_offset: ", Q_offset)
        print("I/Q gain ratio: ", IQ_gain_ratio)

        #Save the calibration result
        with open('calibration_result.txt', 'w') as f:
                f.write('I/Q imbalance calibration result:\n')
                f.write('I_offset: '+str(I_offset)+'\n')
                f.write('Q_offset: '+str(Q_offset)+'\n')
                f.write('I/Q gain ratio: '+str(IQ_gain_ratio)+'\n')

        #save result to a numpy array as well
        calibration_data = np.array([I_offset,Q_offset,IQ_gain_ratio])
        np.save('calibration_data.npy', calibration_data) # save
##        calibration_data = np.load('calibration_data.npy') # load

        print("Calibration completed!")
        time.sleep(100000)

        I_data_corrected = np.array(dc_offset_data_I) - I_offset
        Q_data_corrected = np.array(dc_offset_data_Q) - Q_offset

##        plt.plot(dc_offset_data_Q)
##        plt.title('dc_offset_data_Q')
##        plt.show()


        off_set_data_complex = I_data_corrected + 1j*Q_data_corrected
        plt.plot(np.abs(off_set_data_complex))
        plt.title('dc_offset_envelop')
        plt.show()        

        
        number_packets_in_buffer = 6
        rx_buffer_size = int(total_samples*number_packets_in_buffer) #to be adjusted


        #Generate the known copy of the preamble for packet detection
        known_preamble_bits = preamble_generator.preamble_generator()
        known_preamble_symbols = symbol_mod.symbol_mod(known_preamble_bits, "OOK", len(known_preamble_bits))
        known_preamble = np.abs(pulse_shaping.pulse_shaping(known_preamble_symbols, samples_perbit, fs_in, 'rect', None, None))
        known_preamble_ac = known_preamble - np.mean(known_preamble)

        #Reverse the preamble sample sequence to obtain coefficients for the matched filter
        matched_filter_coef = np.flip(known_preamble_ac)

        #Number of packets transmitted during an iperf test (not the number of datagrams!)
        test_packet_num = 20
        
        #Counter for received packet in Iperf test
        rx_test_packet_counter = 0
        test_end_flag = False 

        
        #Estimate the peak correlation level (in order to determine the threshold for packet detection)

        #Detect the beginning of transmission by inspecting the ratio of |xcorr_max|/|xcorr_min| 
        transmission_begin = 0

        while(transmission_begin == 0):
                test_data_raw = HAL.get_buffer(device, ain, sdr, rx_buffer_size)
                test_data_raw_0 = np.array(test_data_raw[0])
                test_data_raw_1 = np.array(test_data_raw[1])
                test_data = test_data_raw_0 + 1j*test_data_raw_1
                test_data_bb = np.abs(test_data) # convert to baseband/envelope
                test_data_bb_ac = test_data_bb - np.mean(test_data_bb)
                crosscorr = signal.fftconvolve(test_data_bb_ac,matched_filter_coef)[0:len(test_data_bb_ac)] #pass the test samples to the matched filter
                max_corr = np.max(crosscorr)
##                print("max_corr: ", max_corr)
##                plt.plot(crosscorr)
##                plt.title('crosscorr')
##                plt.show()
                min_corr = np.min(crosscorr)
                max_min_ratio_threshold = 3.0
                print("ratio: ", np.abs(max_corr/min_corr))
                if(np.abs(max_corr/min_corr)>max_min_ratio_threshold):
                        transmission_begin = 1
                        #transmission detected, need to push out all packets in the first detected buffer
                        first_buffer_packetized = 0
                        first_buffer_data = test_data
                        first_buffer_corr = crosscorr
                else:
                        print("waiting for transmission to begin.")


##        plt.plot(crosscorr)
##        plt.title('crosscorr')
##        plt.show()

##        plt.plot(test_data_bb)
##        plt.title('transmission detected data')
##        plt.show()
##        time.sleep(100000)

        allowed_fluctuation = 0.2
        peak_threshold = (1-allowed_fluctuation)*max_corr

        payload_start = len(known_preamble)        
        start_flag = 0
        

        #freq hop occurs at (recursive update): t_next_hop = t_next_hop + (dwell_time + transition_time)    
        while(Mode == 1 or Mode == 2 or Mode == 4):                

                if(first_buffer_packetized==0):
                        data = first_buffer_data 
                        crosscorr = first_buffer_corr
                        first_buffer_packetized = 1
                else:
                        #Start processing received buffers                     
                        raw_data = HAL.get_buffer(device,ain,sdr,rx_buffer_size)


                        raw_data_0 = np.array(raw_data[0]) 
                        raw_data_1 = np.array(raw_data[1])

                        data = raw_data_0 + 1j*raw_data_1

                        if(start_flag != 0):
                                data = np.append(stored_segment,data)

                        #Before conducting the crosscorrelation-based packet detection, we convert the samples to the baseband.

                        data_bb = np.abs(data)

##                        plt.plot(data_bb)
##                        plt.title('data_bb')
##                        plt.show()

                        data_bb_ac = data_bb - np.mean(data_bb)

                        #Pass the data samples to the matched filter


                        crosscorr = signal.fftconvolve(data_bb_ac,matched_filter_coef)[0:len(data_bb_ac)]
                        #figure out why the length is limited

##                        plt.plot(crosscorr)
##                        plt.title('crosscorrelation')
##                        plt.show()

                
                
                peak_indices, _ = signal.find_peaks(crosscorr, height =  peak_threshold, distance = int(0.8*total_samples))

##                plt.plot(data_bb)
##                plt.title("data_bb")
##                plt.show()

##                print("length of known preamble: ", len(known_preamble))

                if(len(peak_indices)==0): #no packets detected
                        print("no packets detected in this buffer")
                        continue

                        
                start_indices = peak_indices - len(known_preamble)

                        
                #do not process the last peak and save enough data to append to the next working buffer
                if(len(start_indices)>1):
                        start_indices_but_last = start_indices[0:(len(start_indices)-1)]
                        start_index_last = start_indices[(len(start_indices)-1)]
                        
                        for index in start_indices_but_last:

                                final_index = index + total_samples

                                if(final_index<len(data)):
                                        final_data = data[index:final_index]

                                        q1.put([payload_start, final_data])
                                        print("q size: ", q1.qsize())
                                        if(Mode == 2):
                                                rx_test_packet_counter = rx_test_packet_counter + 1
##                                                print("rx iperf test packet_count: ",rx_test_packet_counter)
                                                if(rx_test_packet_counter == test_packet_num + 1):
                                                        test_end_flag = True
##                                                        print("break because counter suggests received all packets")
                                                        break
                                
                        stored_segment = data[int(start_index_last-0.5*total_samples):]
                        start_flag = 1
                        if(test_end_flag==True):
                                break

                        
                if(len(start_indices)==1 and start_flag ==0):
                        start_index_last = start_indices
                        stored_segment = data[int(start_index_last-0.5*total_samples):]
                        start_flag = 1





                        
def visualization(q3, visualize, fs_in):

        
        plt.ion()

        #Visualize the symbols in the constellation

        while(visualize == 1):
                
                symbols,packet_BER,long_term_BER = q3.get()
                symbols_I = symbols[0]
                symbols_Q = symbols[1]
                plt.clf()
                for i in range(len(symbols_I)):
                        plt.plot(symbols_I[i],symbols_Q[i], color='blue', marker='o', markersize=1)
##                        plt.ylim((-0.6, 0.6))
##                        plt.xlim((-0.6, 0.6))
                plt.title('Rx Constellation of Received Symbol\n'+' Packet BER: '+packet_BER+' Long term BER: '+long_term_BER)
                plt.draw()
                plt.pause(0.002)

                start_time = time.time()
                while(q3.qsize()==0):
                    if(time.time()-start_time>3.0):
                        plt.clf()
                        plt.draw()
                        plt.pause(0.1)


#The main function
if __name__ == "__main__":

        #Creating the command line interfaces (CLIs)
        parser = UI.create_rx_CLI_parser()        
        args = parser.parse_args()


        #System Parameters and default values

        fs = 1000000.0	       # maximum input sampling rate for sustainable data streaming (Hz)
        Ts = 1.0 / fs          # sampling period in seconds
        f0 = 0.0               # homodyne (0 IF)
        M = 8                 # oversampling factor
        T = M*Ts               # symbol period in seconds
        Rs = 1/T               # symbol rate
        segment_size = 1504    # One Transport Stream (TS) packet=188Bytes=1504 bits
        R = 20                 # Packet Ratio: number of segments contained in our larger OOK packet 
        N = R*segment_size     # OOK Packet Length (equals R* segment_size)
        
        #Automatically determine the connected M2K device using iio scan context API
        device = HAL.identify_device()

        #Addtional parameters and default values

        Mode = 4                 # Default is BER mode for calibration
        rx_gain = 30             # Default Rx hardware gain for ADALM-Pluto

        #Default parameters for signal visualization
        visualize = 0            # Visualize the signal received? 1: Yes 0: No 
        visualized_bits = 100    # Number of bits to be plotted (including the header)

        debugging = 0            # Debugging mode disabled by default
        showplot = False

        #Update parameters based on user inputs

        #The GUI will be displayed if the user starts the program without additional command line arguments
        if (len(sys.argv) == 1):
   
                window = UI.create_rx_GUI_parser()
                event, values = window.Read()    
                window.Close()

                #Processing GUI inputs
                if(values[0] != ''):
                        Mode = int(values[0])
                if(values[1] != ''):
                        R = int(values[1])
                        N = R*segment_size
                if(values[2] != ''):
                        M = int(values[2])

                available_freq=["75000000","7500000","750000","75000","7500","750"]
                for i in range(3,9):
                        if (values[i] == True):
                                fs = float(available_freq[i-3])

                if(values[11] == True):
                        visualize = 1

                if(values[12] != ''):
                        visualized_bits = int(values[12])


        #If the user enters command line arguments, the GUI will be skipped the command line arguments will be processed
        if (len(sys.argv) > 1):

                #Processing command line inputs
                if(args.mode != None):
                        Mode = args.mode

                if(args.ratio != None):
                        R = args.ratio
                        N = R*segment_size

                if(args.oversampling_factor != None):
                        M = args.oversampling_factor

                if(args.sampling_rate != None):
                        fs = args.sampling_rate
                        Ts = 1.0 / fs

                if(args.visualized_bits != None):
                        visualize = 1
                        visualized_bits = args.visualized_bits

                if(args.debugging == True):
                        debugging = 1

                if(args.showplot == True):
                        showplot = True

                if(args.rx_gain != None):
                        rx_gain = args.rx_gain


        #Print out received parameter settings specified either through the command line or GUI
        if(Mode == 1):
                print("Streaming mode initiated!")
        if(Mode == 2):
                print("Iperf mode initiated!")
        if(Mode == 4):
                print("BER test mode initiated!")

        #System initialization
                
        #Determine if the "same machine mode" is enabled

        same_machine = False

        if(len(sys.argv) > 1): #command line mode 
                same_machine = args.same_machine
        else:                  #GUI mode
                same_machine = values[9]

        #Specify the symbol modulation scheme (availabel schemes: OOK, BPSK, QPSK, 16QAM)  
        scheme = "QPSK" #default is QPSK for calibration

        preamble_length = 200                             # Preamble length in bits

        #Currently the manual LO frequency tuning and system parameter display via a runtime GUI is only implemented for M2K
        #TODO: Implement the same GUI for pluto as well
        ftuned = 2375 #Initial frequency tuned to for the PLL
        freq = ftuned
        step_freq = 5

        #Manual LO frequency tuning and system parameter display via a runtime GUI is currently only implemented for M2K
        #TODO: Implement the same GUI for pluto as well
        if(device == 'm2k'):                
                q7 = multiprocessing.Queue() #  tx_runtime_GUI to main        
                p6 = multiprocessing.Process(target=UI.tx_runtime_GUI, args=(q7, fs, M, N, ftuned, preamble_length))  #GUI used for frequency hopping and key parameters display
                p6.start()

        #Compute the number of samples per packet and per symbol
        total_samples, samples_perbit, fs_in, Ts_in =  mode_preconfiguration.rx_mode_preconfig(scheme,N,preamble_length,M,fs)
        
        print("R: ",R)
        print("M: ",M)
        print("fs: ",fs)


        q1 = multiprocessing.Queue() # "getsamples" process uses this queue to send to main

        q3 = multiprocessing.Queue() # main process uses this queue to send samples to "visualization" process

        q4 = multiprocessing.Queue() # "data_sink" uses this queue to receive demodulated bits from the main process

        q8 = multiprocessing.Queue() #  queue used to send throughput statistics to the GUI

        p1 = multiprocessing.Process(target=getsamples, args=(q1, Mode,fs_in,total_samples, debugging, same_machine, ftuned, samples_perbit,device, rx_gain)) # "getsamples" process

        p3 = multiprocessing.Process(target=data_sink.data_sink, args=(Mode,q4, R, N, debugging, q3, visualize, q8, segment_size)) #"post-processing" process, handling demodulated bits (sending UDP datagrams/Check BER&FER)

       
        p3.start()
        p1.start()
        

        if(visualize == 1):
                p2 = multiprocessing.Process(target = visualization, args = (q3,visualize,fs_in))

                frame_rate_factor = 30
                visualize_counter = 0

                p2.start()
                print("visualization process started!")


        while(Mode == 1 or Mode == 2 or Mode == 4):


                payload_start, data = q1.get()

                #Perform the non-coherent demodulation
 
                a_demodulated = noncoherent_demodulation.noncoherent_demodulation_wireless(data, samples_perbit, payload_start, N, fs_in, Ts_in)

                #Map the received symbols into bits                
                
                q4.put([a_demodulated, None, data]) #sending demodulated bits to data_sink
