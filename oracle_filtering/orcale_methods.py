

import numpy as np


#FUNCTIONS IMPLEMENTING THE ORACLE METHODS 
#(each function estimates the oracle mask for the corresponding orcale method
# and then applys it to the spectrogram of the mix )

#Inputs:
#   source_spectr: the spectrogram (complex) of the target source
#   mix_spectr: the spectrogram (complex) of the mix
#   alpha: list with the source names

#Outputs:
#   extracted_spectr: extracted source spectr   




def IRM(source_spectr,mix_spectr,alpha):

    eps = np.finfo(np.float).eps
    mag_mix_spectr =np.abs(mix_spectr)**alpha + eps

    # compute soft mask as the ratio between source spectrogram and total
    Mask = np.divide(np.abs(source_spectr)**alpha, mag_mix_spectr )    


    #Apply soft mask
    extracted_source_spectr = np.multiply(mix_spectr, Mask)    

    return extracted_source_spectr



    
def IBM(source_spectr,mix_spectr,theta,alpha):

    eps = np.finfo(np.float).eps
    mag_mix_spectr =np.abs(mix_spectr)**alpha + eps

    # Create Binary Mask
    Mask = np.divide(np.abs(source_spectr)**alpha, mag_mix_spectr )
    Mask[np.where(Mask >= theta)] = 1
    Mask[np.where(Mask < theta)] = 0

    #Apply binary mask
    extracted_source_spectr = np.multiply(mix_spectr, Mask)    

    return extracted_source_spectr
