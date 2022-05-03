import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

#Define the symbol_mod module

#The symbol_demod module takes the following arguments as inputs:

# baseband symbols:            The symbols array to be mapped into bits

# scheme:                      A string indicating which scheme is adopted (e.g.: "OOK", "QPSK")

# channel_gain                 This is the gain of the channel, channel impulse response is simply modeled as g(t)= channel_gain*delta(t)

#The symbol_demod module returns the following argument as output:

# a_demodulated:              An array containing the demodulated bits


def symbol_demod(baseband_symbols, scheme, channel_gain): #"a" is the bit array to be modulate

        a_demodulated = []
        
        if(scheme == 'OOK'):

                s_on = 1.0 * channel_gain

                s_off = 0* channel_gain

                baseband_symbols_I = baseband_symbols[0]

                baseband_symbols_Q = baseband_symbols[1]


                baseband_symbols_complex = baseband_symbols_I + 1j * baseband_symbols_Q


                for i in range( len(baseband_symbols_complex) ):

                        #Coherent: finding the minimum distance between the received symbol and all reference symbols 
                        # in the constellation plot.

                                if (np.abs(baseband_symbols_complex[i] - s_on) < np.abs(baseband_symbols_complex[i] - s_off)):
                                       a_demodulated.append(1)
                                else:
                                       a_demodulated.append(0)


        if(scheme == 'BPSK'):

                baseband_symbols_I = baseband_symbols[0]

                baseband_symbols_Q = baseband_symbols[1]

                reference_plus = 1.0*channel_gain

                reference_minus = -reference_plus

                baseband_symbols_complex = baseband_symbols_I + 1j * baseband_symbols_Q

                for i in range(len(baseband_symbols_complex)):

                    #Find the minimum distance between the received symbol and all reference symbols in the constellation plot.

                        if (np.abs(baseband_symbols_complex[i] - reference_plus) < np.abs(baseband_symbols_complex[i] - \
                                                                                          reference_minus)):
                               a_demodulated.append(0)
                        else:
                               a_demodulated.append(1)


        if(scheme == 'QPSK'):

                baseband_symbols_I = baseband_symbols[0]

                baseband_symbols_Q = baseband_symbols[1]


                I_demodulated = []
                Q_demodulated = []

                #Construct the received symbols on the complex plane (signal space constellation)
                baseband_symbols_complex = baseband_symbols_I + 1j * baseband_symbols_Q

                #Compute and define the reference signals in the signal space (4 constellation points)
                reference_00 = -1.0*channel_gain  -1j* channel_gain

                reference_11 = 1.0*channel_gain + 1j* channel_gain

                reference_01 = -1.0*channel_gain + 1j* channel_gain

                reference_10 = 1.0*channel_gain  -1j* channel_gain

                #Start a for-loop to iterate through all complex symbols and make a decision on 2-bits of data 

                for i in range(len(baseband_symbols_complex)):

                        symbol = baseband_symbols_complex[i]

                        #Find the minimum distance between the received symbol and all symbols in the constellation plot.

                        if (  np.abs(symbol - reference_11) == np.amin(  [np.abs(symbol - reference_11),  \
                                                                          np.abs(symbol - reference_10), \
                                                                          np.abs(symbol - reference_00), \
                                                                          np.abs(symbol - reference_01)]  )  ):
                               I_demodulated.append(1)
                               Q_demodulated.append(1)
                        elif(np.abs(symbol - reference_10) == np.amin(  [np.abs(symbol - reference_11),  \
                                                                          np.abs(symbol - reference_10), \
                                                                          np.abs(symbol - reference_00), \
                                                                          np.abs(symbol - reference_01)]  )  ):
                               I_demodulated.append(1)
                               Q_demodulated.append(0)
                        elif(np.abs(symbol - reference_01) == np.amin(  [np.abs(symbol - reference_11),  \
                                                                          np.abs(symbol - reference_10), \
                                                                          np.abs(symbol - reference_00), \
                                                                          np.abs(symbol - reference_01)]  )  ):
                               I_demodulated.append(0)
                               Q_demodulated.append(1)
                        elif(np.abs(symbol - reference_00) == np.amin(  [np.abs(symbol - reference_11),  \
                                                                          np.abs(symbol - reference_10), \
                                                                          np.abs(symbol - reference_00), \
                                                                          np.abs(symbol - reference_01)]  )  ):
                               I_demodulated.append(0)
                               Q_demodulated.append(0)

                a_demodulated = np.append(I_demodulated, Q_demodulated)


        return a_demodulated


#Helper function

#Rotating a vector in constellation diagram:

#Take a complex number (2-D vector) as input (referenced through x, y coordinate) and return the coordinate of the rotated vector (by angle radians)

def rotate(vector, angle):

        x = np.real(vector)

        y = np.imag(vector)

        x_new = np.cos(angle)*x - np.sin(angle)*y

        y_new = np.sin(angle)*x + np.cos(angle)*y

        vector_new = x_new + 1j*y_new
        

        return vector_new
