# from turtle import forward
import musdb
import museval
import numpy as np
import functools
import argparse
import time
import os
import librosa
# from scipy.signal import stft, istft



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
def IRM(track , sources_names_list , nb_chan , front_end , alpha=1, eval_dir=None):
    """Ideal Ratio Mask:
    processing all channels inpependently with the ideal ratio mask.
    this is the ratio of spectrograms, where alpha is the exponent to take for
    spectrograms. usual values are 1 (magnitude) and 2 (power)"""


    sources_dict = track.sources
    

    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # compute STFT of Mixture
    N = track.audio.shape[0]  # remember number of samples for future use
    X = np.array( list( map( lambda chanell : front_end["forward"](chanell) , track.audio.T if nb_chan==2 else np.array([librosa.to_mono(track.audio.T)]) ) ) )
    #X = np.array( list( map( lambda chanell : front_end["forward"](chanell) , track.audio.T ) ) )

    
    # Compute sources spectrograms--------------------------------------------------------------------------------------------------------------------------------
    P = {}
    model = eps

    for name in sources_names_list: 
        # compute spectrogram of target source:
        # magnitude of STFT to the power alpha
        source = sources_dict[name]
        P[name] = np.abs( np.array( list( map( lambda chanell :  front_end["forward"](chanell) , source.audio.T if nb_chan==2 else np.array([librosa.to_mono(source.audio.T)]) ) ) ) )**alpha
        #P[name] = np.abs( np.array( list( map( lambda chanell :  front_end["forward"](chanell) , source.audio.T ) ) ) )
        #model += P[name]


    model += np.abs(X)  



    # now performs separation
    estimates = {}
    accompaniment_source = 0
    for name in sources_names_list:
        # compute soft mask as the ratio between source spectrogram and total
        Mask = np.divide(np.abs(P[name])**alpha, model**alpha)

        # multiply the mix by the mask
        Yj = np.multiply(X, Mask)

        # invert to time domain
        target_estimate = np.array( list( map( lambda Yj_chanell :  front_end["backward"](Yj_chanell) , Yj ) ) ).T[:N, :]

        # set this as the source estimate
        estimates[name] = target_estimate if nb_chan==2 else target_estimate.reshape(-1)
        #estimates[name] = target_estimate 

        # accumulate to the accompaniment if this is not vocals
    #     if name != 'vocals':
    #         accompaniment_source += target_estimate

    # estimates['accompaniment'] = accompaniment_source

    if eval_dir is not None:
        museval.eval_mus_track(
            track,
            estimates,
            output_dir=eval_dir,
        )

    return estimates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Ideal Ratio Mask'
    )
    parser.add_argument(
        '--audio_dir',
        nargs='?',
        help='Folder where audio results are saved'
    )

    parser.add_argument(
        '--eval_dir',
        nargs='?',
        help='Folder where evaluation results are saved'
    )

    parser.add_argument(
        '--alpha',
        type=int,
        default=2,
        help='exponent for the ratio Mask'
    )

    args = parser.parse_args()

    alpha = args.alpha

    # initiate musdb
    mus = musdb.DB(root = "/home/nnanos/open-unmix-pytorch/musdb18")

    mus.run(
        functools.partial(
            IRM, alpha=alpha, eval_dir=args.eval_dir
        ),
        estimates_dir=args.audio_dir,
        subsets='test',
        parallel=True,
        cpus=2
    )



