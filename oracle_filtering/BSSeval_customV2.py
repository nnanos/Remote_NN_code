import numpy as np
import librosa
import museval


def compute_energy_ratios(s_target,e_interf,e_artif):

    #computing the energy ratios
    # metrics = {}
    # metrics['SDR'] = 10*np.log10( np.dot( s_target,s_target ) / np.dot( (e_interf+e_artif) , (e_interf+e_artif) ) )
    # metrics['SIR'] = 10*np.log10( np.dot( s_target,s_target ) / np.dot( e_interf,e_interf ) )
    # metrics['SAR'] = 10*np.log10( np.dot( s_target + e_interf , s_target + e_interf ) / np.dot( e_artif,e_artif ) )

    return [10*np.log10( np.dot( s_target,s_target ) / np.dot( (e_interf+e_artif) , (e_interf+e_artif) ) ), 
            10*np.log10( np.dot( s_target,s_target ) / np.dot( e_interf,e_interf ) ),
            10*np.log10( np.dot( s_target + e_interf , s_target + e_interf ) / np.dot( e_artif,e_artif ) )
            ]  



def evaluation(estimates,musdb_track,target,dur=None,orthogonality_assumption=False):
    #EVALUATION FUNCTION IMPLEMENTED BASED ON THE BSSeval metrics paper
    #estimate : mono signal estimate
    #musdb_track : the object that represents a Test track from the musdb dataset 
    #target : string options ['vocals','drums','bass','other']
    #dur : duration of the song (for faster processing (FOR TESTING PURPOSES))


    vocals = librosa.to_mono(musdb_track.targets['vocals'].audio.T)
    drums = librosa.to_mono(musdb_track.targets['drums'].audio.T)
    bass = librosa.to_mono(musdb_track.targets['bass'].audio.T)
    other = librosa.to_mono(musdb_track.targets['other'].audio.T)

    target_source = librosa.to_mono(musdb_track.targets[target].audio.T)
    s_target = ( np.dot( target_source,estimate ) / np.dot(target_source,target_source) ) * target_source


    S = np.c_[vocals,bass,other,drums]

    if orthogonality_assumption:
        #obtainig the dot products of the estimate with all the sources (i.e. the geometric representation of the estimate in the basis defined by the sources)
        cords = np.dot(S.T,estimate)

        #obtaining the basis signals (normalizing each column by its energy)
        #S = np.c_[vocals/(np.linalg.norm(vocals)**2) , bass/(np.linalg.norm(bass)**2),other/(np.linalg.norm(other)**2),drums/(np.linalg.norm(drums)**2) ]
        #each row contains the sources normalized by its energy
        S = np.array(list(map(lambda s : s/np.dot(s,s), S.T)))

        #obtaining the projection of the estimate in the basis subspace
        Proj_to_S_subspace = np.dot(S.T,cords)

        #we can obtain the e_interf by subtraction because we assume an orthogonal basis (the source signals are mutualy orthogonal)
        e_interf = Proj_to_S_subspace - s_target

        #by orthogonality principle we have that e=x-x_hat is orthogonal to the subspace that we projected our data(estimates==x)
        e_artif = estimate - Proj_to_S_subspace

    else:
        Rss_inv = np.linalg.inv( np.dot(S.T,S) ) 
        tmp = np.array( list( map( lambda x : np.dot(estimate,x)  , S.T ) ) )
        c = np.dot( Rss_inv,tmp )

        Proj_to_S_subspace = np.dot(S,c)

        e_interf = Proj_to_S_subspace - s_target
        e_artif = estimate - Proj_to_S_subspace



    '''
    #case window/fs = 1sec and the number of secs in the song is an int
    #and hop=len(window)//2 (overlap=2)
    window_len = 44100
    n_windows = len(s_target)//window_len
    hop=window_len//2

    #create the inds
    inds = [ np.arange(0,window_len) ]
    for i in range(2*n_windows-1):
        inds.append(inds[i]+hop)

    list(map(lambda ind : compute_energy_ratios(s_target[ind],e_interf[ind],e_artif[ind]),inds))
    '''

    #computing the energy ratios
    metrics = compute_energy_ratios(s_target,e_interf,e_artif)

    return metrics