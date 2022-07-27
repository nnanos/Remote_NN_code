

import numpy as np
import scipy
import time
import os



#Auxilliary funcs------------------------------------------------------------

def timeis(func):
    '''Decorator that reports the execution time.'''
  
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
          
        print(func.__name__, end-start)
        return result
    return wrap

def cputime():
    utime, stime, cutime, cstime, elapsed_time = os.times()
    return utime

#_-----------------------------------------------------------------------------

@timeis
def HPSS(mix_spectr,filter_len_harm,filter_len_per,alpha):
    #FUNCTIONS IMPLEMENTING THE ORACLE METHODS 
    #(each function estimates the oracle mask for the corresponding orcale method
    # and then applys it to the spectrogram of the mix )

    #Inputs:
    #   mix_spectr: the spectrogram (complex) of the mix  (SHAPE IS ())
    #   filter_len_harm: the lenth of the horizontal median median filter 
    #   filter_len_per: the lenth of the horizontal median median filter

    #Outputs:
    #   extracted_spectr: extracted source spectr   

 

    mag_mix_spectr = np.abs(mix_spectr)**alpha


    #MEDIAN FILTERING STEP--------------------------------------------------------
    #Executing median filtering on each row (hence enchansing horizontal lines or harmonic part)
    h_mag = np.array([scipy.signal.medfilt(i,filter_len_harm) for i in mag_mix_spectr])

    #Executing median filtering on each column (hence enchansing vertical lines or percussive part)
    p_mag = np.array([scipy.signal.medfilt(i,filter_len_per) for i in mag_mix_spectr.T]).T
    #----------------------------------------------------------------------------    


    #BINARY SPECTRAL MASKING STEP-----------------------
    #Creating the percussive mask
    nb_rows,nb_cols = mix_spectr.shape
    bin_mask_p = np.zeros((nb_rows,nb_cols))
    for i in range(nb_rows):
        for j in range(nb_cols):
            if h_mag[i,j]<p_mag[i,j]:
                bin_mask_p[i,j]=1 


    #Apply binary mask
    extracted_source_spectr = np.multiply(mix_spectr, bin_mask_p)

    return extracted_source_spectr

    
    