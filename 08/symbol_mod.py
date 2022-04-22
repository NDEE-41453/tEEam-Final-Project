import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#Define the symbol_mod module

#The symbol_mod module takes the following arguments as inputs:

# packet_bits                 The bit array to be mapped into symbols (including both the preamble bits and the
#                             payload bits)

# scheme                      A string indicating which scheme is adopted (e.g.: "OOK", "QPSK")

# preamble_length             Length of the preamble (in bits)

#The symbol_mod function returns the following argument as output:

# baseband_symbols:           The baseband symbols obtained after mapping the bits


def symbol_mod(packet_bits, scheme, preamble_length): 


        if(scheme == 'OOK'):

                preamble = packet_bits[0:preamble_length]
                payload = packet_bits[preamble_length:len(packet_bits)]
                preamble_symbols = 1.0*preamble
                payload_symbols = 1.0*payload                
                baseband_symbols = np.append(preamble_symbols,payload_symbols)

        if(scheme == 'BPSK'):

                preamble = packet_bits[0:preamble_length]
                payload = packet_bits[preamble_length:len(packet_bits)]
                preamble_symbols = 1.0*preamble 
                payload_symbols = 2.0*payload - 1.0
                payload_symbols = 1/np.sqrt(2)*payload_symbols
##                print("length of preamble: ", len(preamble_symbols))
##                print("length of payload: ", len(payload_symbols))
                baseband_symbols = np.append(preamble_symbols,payload_symbols)
                

        if(scheme == 'QPSK'):

                preamble = packet_bits[0:preamble_length]
                payload = packet_bits[preamble_length:len(packet_bits)]
                baseband_symbols_I = 1.0*preamble
                baseband_symbols_Q = np.zeros(preamble_length)

                #Map the payload
                I_bits = payload[0:int(len(payload)/2)]
                Q_bits = payload[int(len(payload)/2):len(payload)]
                I_symbols = (2.0*I_bits - 1.0)*(1/np.sqrt(2))
                Q_symbols = (2.0*Q_bits - 1.0)*(1/np.sqrt(2))

                preamble_symbols = baseband_symbols_I + 1j*baseband_symbols_Q
                data_symbols = I_symbols + 1j* Q_symbols
                #Scale QPSK payload to have the same per channel average signal power as OOK
                data_symbols = 1/np.sqrt(2)*data_symbols
                baseband_symbols = np.append(preamble_symbols, data_symbols)

##                print("I symbols :", I_symbols[0:10])
##                print("Q symbols :", Q_symbols[0:10])
               
        return baseband_symbols
        
        
