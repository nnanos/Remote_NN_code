from Time_Frequency_Analysis.NSGT_CQT import NSGT_cqt
from Time_Frequency_Analysis.SCALE_FRAMES import scale_frame
from Time_Frequency_Analysis.STFT_custom import STFT_CUSTOM
import nsgt as nsg        
import librosa
import scipy
import numpy as np
import scipy.signal as sg



#FRONT_ENDs available (its scalable)----------------------------------------------------------------
front_end_lookup ={
    "STFT_custom":STFT_CUSTOM,
    "librosa":librosa,
    "scipy":scipy.signal,
    "nsgt_grr":nsg,
    "NSGT_CQT":NSGT_cqt,
    "NSGT_SCALE_FRAMES":scale_frame
}


def pick_front_end(args,front_end_params,musdb_track):  

    #A FUNCTION to get front_end forward and backward methods---
    #Inputs:
    #   front_end_params : A dict containing the front end params
    #   musdb_track

    
    # #DETERMINING FRONT END----------------------

    #The STFTs and CQTs are L (signal len) dependend ie the only thing needed to construct the transform windows is L
    L = musdb_track.audio.shape[0]
    #scale_frame is SIGNAL DEPENDEND i.e. in order to determine the windows positions (and consecuently construct them) you need the onsets
    #of the particular signal (its more complicated for the stereo chanel case so we test the mono)
    mono_mix = librosa.to_mono(musdb_track.audio.T)


    front_end = front_end_lookup[front_end_params["front_end_name"]]

    if front_end==scipy.signal:
        g = np.hanning(front_end_params["support"])
        forward = lambda y : scipy.signal.stft( y , window=g, nfft=front_end_params["M"] , noverlap=front_end_params["a"] ,nperseg=front_end_params["support"])[-1]
        backward = lambda Y : scipy.signal.istft( Y, window=g, nfft=front_end_params["M"] , noverlap=front_end_params["a"]  ,nperseg=front_end_params["support"])[1]

    elif front_end==librosa:
        g = np.hanning(front_end_params["support"])
        #X = np.array( list( map( lambda x :  librosa.stft(x,n_fft=front_end_params["M"],hop_length=front_end_params["a"],win_length=front_end_params["support"],window=g) , track.audio.T ) ) )
        forward = lambda y : librosa.stft( y=y , n_fft=front_end_params["M"],hop_length=front_end_params["a"],win_length=front_end_params["support"],window=g ) 
        backward = lambda Y : librosa.istft( stft_matrix=Y ,hop_length=front_end_params["a"]  ,win_length=front_end_params["support"],window=g )
    
    elif front_end==STFT_CUSTOM:
        g = np.hanning(front_end_params["support"])
        stft = front_end(g,front_end_params["a"],front_end_params["M"],front_end_params["support"],L)
        forward = stft.forward  
        backward = stft.backward
    
    elif front_end==nsg:
        scale = nsg.LogScale
        scl = scale(front_end_params["ksi_min"], front_end_params["ksi_max"], front_end_params["B"]*7 )
        nsgt = nsg.NSGT(scl, front_end_params["ksi_s"], L=L, real=1, matrixform=1, reducedform=0 ,multithreading=0)
        forward = nsgt.forward
        backward = nsgt.backward


    elif front_end==NSGT_cqt:
        nsgt = front_end(ksi_s=front_end_params["ksi_s"],ksi_min=front_end_params["ksi_min"], ksi_max=front_end_params["ksi_max"], B=front_end_params["B"],L=L,matrix_form=1)
        forward = nsgt.forward
        backward = nsgt.backward


    elif front_end==scale_frame:

        if front_end_params["onset_det"]=="custom":

            #Onset det custom using hpss to estimate the drums:
            D = librosa.stft(mono_mix)
            H, P = librosa.decompose.hpss(D, margin=(1.0,7.0))
            y_perc = librosa.istft(P)
            onsets = librosa.onset.onset_detect(y=y_perc, sr=front_end_params["ksi_s"], units="samples")

        else:
            onsets = librosa.onset.onset_detect(y=mono_mix, sr=front_end_params["ksi_s"], units="samples")


        middle_window = np.hanning if front_end_params["middle_window"]=="np.hanning" else sg.tukey
        
        scl_frame_object = front_end(ksi_s=front_end_params["ksi_s"],min_scl=front_end_params["min_scl"],overlap_factor=front_end_params["ovrlp_fact"],onset_seq=onsets,middle_window=middle_window,L=L,matrix_form=front_end_params["matrix_form"],multiproc=front_end_params["multiproc"])
        forward = scl_frame_object.forward
        backward = scl_frame_object.backward



    front_end_repr = {
        "forward" : forward,
        "backward" : backward
    }

    return front_end_repr