

import numpy as np
import musdb
import museval
import librosa
import scipy
import nsgt as nsg
import Time_Frequency_Analysis.STFT_custom as STFT_custom
import Time_Frequency_Analysis.NSGT_custom as NSGT_custom







#Auxilliary funcs------------------------------------------------------------
def timeis(func):
    '''Decorator that reports the execution time.'''
    import time
  
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
          
        print(func.__name__, end-start)
        return result
    return wrap

def cputime():
    import os
    utime, stime, cutime, cstime, elapsed_time = os.times()
    return utime




@timeis
def IRM(track, front_end  , alpha=2 , params = None , eval_dir=None ):
    """Ideal Ratio Mask:
    processing all channels inpependently with the ideal ratio mask.
    this is the ratio of spectrograms, where alpha is the exponent to take for
    spectrograms. usual values are 1 (magnitude) and 2 (power)"""

    #params["a"]  


    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    L = track.audio.shape[0]  # remember number of samples for future use
    

    #DETERMINING THE FRONTEND AND THE ASOSIATED PARAMS---------------------------------------------------------
    
    if front_end==scipy.signal:
        g = np.hanning(params["support"])
        X = scipy.signal.stft(track.audio.T, window=g, nfft=params["M"] , noverlap=params["a"] ,nperseg=params["support"])[-1]
        #X = np.array( list( map( lambda x : scipy.signal.stft( x, window=g, nfft=params["M"] , noverlap=params["a"] ,nperseg=params["support"])[-1] , track.audio.T ) ) )
    elif front_end==librosa:
        g = np.hanning(params["support"])
        X = np.array( list( map( lambda x :  librosa.stft(x,n_fft=params["M"],hop_length=params["a"],win_length=params["support"],window=g) , track.audio.T ) ) )
    elif front_end==STFT_custom:
        g = np.hanning(params["support"])
        support = params["support"]
        stft = STFT_custom.STFT_CUSTOM(g,params["a"],params["M"],params["support"],L)
        X = np.array( list( map( lambda x :  stft.forward(x) , track.audio.T ) ) )
    
    elif front_end==nsg:
        scale = nsg.LogScale
        scl = scale(params["ksi_min"], params["ksi_max"], params["B"]*7 )
        nsgt = nsg.NSGT(scl, params["ksi_s"], L, real=1, matrixform=1, reducedform=0 ,multithreading=0)
        X = np.array( list( map( lambda x :  nsgt.forward(x) , track.audio.T ) ) )
    elif front_end==NSGT_custom:
        nsgt = NSGT_custom.NSGT_CUSTOM(ksi_s=params["ksi_s"],ksi_min=params["ksi_min"], ksi_max=params["ksi_max"], B=params["B"],L=L,matrix_form=1)
        X = np.array( list( map( lambda x :  nsgt.forward(x) , track.audio.T ) ) )

    #------------------------------------------------------------------------------


    # Compute sources spectrograms
    P = {}
    # compute model as the sum of spectrograms
    model = eps

    for name, source in track.sources.items():
        # compute spectrogram of target source:
        #DETERMINING THE FRONTEND AND THE ASOSIATED PARAMS---------------------------------------------------------
        if front_end==scipy.signal:
            P[name] = np.abs(scipy.signal.stft(source.audio.T, window=g, nfft=params["M"] , noverlap=params["a"] ,nperseg=params["support"])[-1])**alpha
            #P[name] = np.abs( np.array( list( map( lambda x :  scipy.signal.stft(x, window=g, nfft=params["M"] , noverlap=params["a"] ,nperseg=params["support"])[-1] , source.audio.T ) ) ) )**alpha

        elif front_end==librosa:
            P[name] = np.abs( np.array( list( map( lambda x :  librosa.stft(x,n_fft=params["M"],hop_length=params["a"],win_length=params["support"],window=g) , source.audio.T ) ) ) )**alpha
        elif front_end==STFT_custom:
            P[name] = np.abs( np.array( list( map( lambda x :  stft.forward(x) , source.audio.T ) ) ) )**alpha
        
        elif front_end==nsg:
            P[name] = np.abs( np.array( list( map( lambda x :  nsgt.forward(x) , source.audio.T ) ) ) )**alpha
        else:
            P[name] = np.abs( np.array( list( map( lambda x :  nsgt.forward(x) , source.audio.T ) ) ) )**alpha

        #------------------------------------------------------------------------------        
        model += P[name]


    # now performs separation
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():
        # compute soft mask as the ratio between source spectrogram and total
        Mask = np.divide(P[name], model)

        # multiply the mix by the mask
        Yj = np.multiply(X, Mask)

        # invert to time domain
        #DETERMINING THE FRONTEND AND THE ASOSIATED PARAMS---------------------------------------------------------
        if front_end==scipy.signal:
            target_estimate = scipy.signal.istft(Yj, window=g, nfft=params["M"] , noverlap=params["a"]  ,nperseg=params["support"])[1].T[:L, :]
            #target_estimate = np.array( list( map( lambda y :  scipy.signal.istft(y, window=g, nfft=params["M"]  , noverlap=params["a"] ,nperseg=params["support"])[-1] , Yj ) ) ).T[:L, :]
        elif front_end==librosa:
            target_estimate = np.array( list( map( lambda y :  librosa.istft(y,hop_length=params["a"]  ,win_length=params["support"],window=g) , Yj ) ) ).T[:L, :]
        elif front_end==STFT_custom:
            target_estimate = np.array( list( map( lambda y :  stft.backward(y) , Yj ) ) ).T[:L, :]
        
        elif front_end==nsg:
            target_estimate = np.array( list( map( lambda y :  nsgt.backward(y) , Yj ) ) ).T[:L, :]
        else:
            target_estimate = np.array( list( map( lambda y :  nsgt.backward(y) , Yj ) ) ).T[:L, :]

        #------------------------------------------------------------------------------   

        #target_estimate = istft(Yj)[1].T[:L, :]

        # set this as the source estimate
        estimates[name] = target_estimate


    if eval_dir is not None:
        museval.eval_mus_track(
            track,
            estimates,
            output_dir=eval_dir,
        )

    return estimates



if __name__ =='__main__':

    import numpy as np
    import musdb
    import museval
    import librosa
    import scipy
    import nsgt as nsg
    import Time_Frequency_Analysis.STFT_custom as STFT_custom
    import Time_Frequency_Analysis.NSGT_custom as NSGT_custom
    import time
    import yaml
    import argparse



    parser = argparse.ArgumentParser(description='Command description.')


    parser.add_argument('--front_end', type=str, default="STFT",
                            help='provide Transform name')

    parser.add_argument('-p', '--params', type=yaml.load,
                            help='provide Transform parameters as a quoted json sting')    

    args, _ = parser.parse_known_args()

    front_end_lookup ={
        "STFT_custom":STFT_custom,
        "librosa":librosa,
        "scipy":scipy.signal,
        "nsgt":nsg,
        "NSGT_custom":NSGT_custom
    }


    #DATASET
    subset = 'train'
    #mus = musdb.DB(download=True,subsets=subset)
    root = "/home/nnanos/open-unmix-pytorch/musdb18"
    mus = musdb.DB(root=root,
    is_wav=False,
    subsets=subset,
    )
    track = mus[10]    


    track = mus[0]
    front_end = front_end_lookup[args.front_end]
    params = args.params
    estimates=IRM(track,front_end,params=params)


    a = 0