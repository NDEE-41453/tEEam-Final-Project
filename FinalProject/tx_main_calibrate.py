#Import ADALM Python libraries
import adi
import iio
import libm2k

#Import other dependencies
import time
import numpy as np
import os
import socket
import sys
import multiprocessing
import matplotlib.pyplot as plt
import math
from scipy import signal

#Import Radioware TX BB DSP modules
import pulse_shaping
import preamble_generator
import symbol_mod
import mode_preconfiguration
import data_source

#Import the HAL for ADALM (2000/Pluto)
import HAL

#Import the GUI/CLI parser modules
import UI

#The main function/process
if __name__ == "__main__":

        #Creating the command line interfaces (CLIs)
        parser = UI.create_tx_CLI_parser()        
        args = parser.parse_args()

        #System Parameters and default values

        fs = 750000.0	       # maximum output sampling rate sustainable for pluto streaming: 2750000 (Hz), for m2k streaming: 750000 (Hz) 
        Ts = 1.0 / fs          # sampling period in seconds
        f0 = 0.0               # homodyne (0 HZ IF)
        M = 6                  # oversampling factor
        T = M*Ts               # symbol period in seconds
        Rs = 1/T               # symbol rate
        segment_size = 1504    # One Transport Stream (TS) packet=188Bytes=1504 bits
        R = 20                 # Packet Ratio: number of segments contained in our larger OOK packet 
        N = R*segment_size     # OOK Packet Length (equals R* segment_size)


        #Automatically determine the connected M2K device using iio scan context API
        device = HAL.identify_device()
            
        #Addtional parameters and default values

        #default tx gain
        tx_gain = 0            
        
        b = "40k"             #bandwidth of Iperf client (UDP test)

        test_packet_num = 20  #number of ADALM packets transmitted in Iperf test

        Mode = 4               #Default is BER mode for calibration purpose

        debugging = 0          #developer debugging mode disabled by default

        #Update parameters based on user inputs (in GUI or CLI)

        #The GUI will be displayed if the user starts the program without additional command line arguments
        if (len(sys.argv) == 1):
 
                window = UI.create_tx_GUI_parser()     
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

                available_freq = ["75000000","7500000","750000","75000","7500","750"]
                for i in range(3,9):
                        if (values[i] == True):
                                fs = float(available_freq[i-3])

                if(values[9] != ''):
                        b = values[9]
                        

        #If the user enters command line arguments, the GUI will be skipped and the command line arguments will be processed
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

                if(args.bandwidth != None):
                        b = args.bandwidth

                if(args.debugging == True):
                        debugging = 1

                if(args.tx_gain != None):
                        tx_gain = args.tx_gain

        #Print out received parameter settings specified either through the command line or GUI
        if(Mode == 1):
                print("Streaming mode initiated!")
        if(Mode == 2):
                print("Iperf test mode initiated!")
        if(Mode == 4):
                print("BER test mode initiated!")

        print("system parameters: ")
        print("R: ",R)
        print("M: ",M)
        print("fs: ",fs)
        
        if(Mode == 2):
                print("b: ",b)

        #System initialization
                
        #Determine if the "same machine mode" is selected
        same_machine = False

        if(len(sys.argv) > 1): #command line mode 
                same_machine = args.same_machine
        else:                #GUI mode
                same_machine = values[10]

        #Specify the symbol modulation scheme (availabel schemes: OOK, BPSK, QPSK, 16QAM)  
        scheme = "QPSK"  #use QPSK for calibration
        

        #Length of the preamble sequence (in bits)
        preamble_length = 200

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

        #Initialize the transmiter device
        aout, spi_desc, sdr = HAL.setup_tx_device_control(device, same_machine, ftuned, tx_gain, fs)

        #TX device initialized and ready after initial spurs (due to ADC & DAC calibration)

        #For calibration, directly start TX transmission without waiting

        #Initialize and precompute for each operation mode (e.g., create socket, generate pseudo random packet payload)
        serverSock, generated_sequence, sequence_counter, l = mode_preconfiguration.tx_mode_preconfig(Mode, R, segment_size, N, b, test_packet_num)
        iperf_mode_packet_counter = 0
                
        #Parameters for throughput calculation
        packet_kbits = N/1024
        total_kbits = 0
        window_counter = 0

        #Specify the number of packets contained in one transmitter buffer pushed
        if(device == 'm2k'):
                final_buffer = [[],[]]
                packet_counter = 0
                packets_per_buffer = 15

        while(Mode == 1 or Mode == 2 or Mode == 4):

                if(device == 'm2k'):

                        if(q7.qsize()!=0):
                                tune_command = q7.get()
                                if(tune_command=='up'):
                                    print("LO tune up command get ")
                                    freq = freq + step_freq
                                else:
                                    print("LO tune down command get")
                                    freq = freq - step_freq
                                HAL.freq_tune(device, spi_desc, sdr, freq)
                                
                
                if(window_counter == 0):
                        start = time.time()
                        total_kbits = 0

                if(Mode == 4):
                        sequence_counter = sequence_counter + 1

                if(Mode == 2):
                        iperf_mode_packet_counter = iperf_mode_packet_counter + 1
                
                Bits = data_source.data_source(Mode, serverSock, generated_sequence, sequence_counter, l)

                preamble = preamble_generator.preamble_generator()  

                packet_bits = np.append(preamble, Bits)

                preamble_length = len(preamble)
                
                baseband_symbols = symbol_mod.symbol_mod(packet_bits, scheme, preamble_length)

                #Specify the pulse shape ('rect' or 'rrc')
                pulse_shape = 'rect'
                baseband = pulse_shaping.pulse_shaping(baseband_symbols, M, fs, pulse_shape, None, None)
                
                bb_amplitude = 0.2
                buffer = bb_amplitude*baseband

                buffer_IQ = buffer
                if(device == 'm2k'):
                        I_data = np.real(buffer_IQ)
                        Q_data = np.imag(buffer_IQ)

                        if(scheme=="OOK" or scheme=="BPSK"):
##                                plt.plot(I_data)
##                                plt.title('I_data before push')
##                                plt.show()
##
##                                #crosscorrelate with the MF for packet detection (bypassing physical layer)
##                                #Generate the known copy of the preamble for packet detection
##                                known_preamble_bits = preamble_generator.preamble_generator()
##                                known_preamble_symbols = symbol_mod.symbol_mod(known_preamble_bits, "OOK", len(known_preamble_bits))
##                                known_preamble = np.abs(pulse_shaping.pulse_shaping(known_preamble_symbols, M, fs, 'rect', None, None))
##
##                                known_preamble_ac = known_preamble - np.mean(known_preamble)
##
##                                #Reverse the preamble sample sequence to obtain coefficients for the matched filter
##                                matched_filter_coef = np.flip(known_preamble_ac)
##                                I_data = I_data - np.mean(I_data)
##                                crosscorr = signal.fftconvolve(I_data,matched_filter_coef)[0:len(I_data)] #pass the test samples to the matched filter
##                                plt.plot(crosscorr)
##                                plt.title('after MF')
##                                plt.show()
                                
                                HAL.push_buffer(device,aout,sdr,I_data,scheme)

                                #Push dummy packets at the end of iPerf test
                                if(iperf_mode_packet_counter==test_packet_num+1):
                                        HAL.push_buffer(device,aout,sdr,I_data,scheme)
                                        HAL.push_buffer(device,aout,sdr,I_data,scheme)



                        else:
                                packet_counter = packet_counter + 1

                                final_buffer[0] = np.append(final_buffer[0],I_data)
                                final_buffer[1] = np.append(final_buffer[1],Q_data)
                                
                                if(packet_counter%packets_per_buffer==0):     
                                        HAL.push_buffer(device,aout,sdr,final_buffer,scheme)
                                        final_buffer = [[],[]]
                                        #note: for 
                                        time.sleep(0.1)
                                else:
                                        pass
                                #Note: For streaming via m2k with maximum rate possible
                                #Multiple packets need to be stored per buffer when both I,Q are used                                    
                                #Need some wait time between push buffer calls in order to not crush the program

                if(device == 'pluto'):
                        HAL.push_buffer(device,aout,sdr,buffer,scheme)
                        
                #Print out throughput information

                window_counter = window_counter+1
                total_kbits = total_kbits+packet_kbits

                if(window_counter == 40): #another x packets sent
                        window_throughput = total_kbits/(time.time()-start)
                        if(debugging == 0):
                                print("Average Tx throughput(last 40 packets):",window_throughput)
                        window_counter = 0

