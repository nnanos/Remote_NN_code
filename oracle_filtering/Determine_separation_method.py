import numpy as np
import librosa
from orcale_methods import IRM,IBM
from HPSS import HPSS




#EST_METHOD-----------------------------------------------------------------------
est_mthd_lookup ={
    "IBM":IBM,
    "IRM":IRM,
    "HPSS":HPSS,

}    



def pick_sep_mthd_and_est_soucres(args,est_mthd_params,track,source_targets,front_end,nb_chanels):

    #FUNCTION TO PRODUCE THE SOURCE ESTIMATES FOR ONE TRACK
    #Inputs:
    #   est_mthd_params: a dict containig the estimation method params
    #   track: musb track object 
    #   source_targets: list with the source names
    #   front_end: forward backward methds
    #   nb_chanels: int       
    #Outputs:
    #   estimates_dict: dict containing the estimates     
    



    est_mthd = est_mthd_lookup[est_mthd_params["est_mthd_name"]]

    sources_dict = track.sources



    # compute STFT of Mixture--------------------------------------------------------------------------------------------------------------------------------------
    alpha = est_mthd_params["alpha"]
    N = track.audio.shape[0]  # remember number of samples for future use
    X = np.array( list( map( lambda chanell : front_end["forward"](chanell) , track.audio.T if nb_chanels==2 else np.array([librosa.to_mono(track.audio.T)]) ) ) )
    (I, F, T) = X.shape




    #SEPARATION---------------------------------------------------------------------

    target_estimates = {}
    source_spectr_dict = {}         
    
    #ORACLE FILTERING CASE--------------------------------------------------------------------------------------------------
    if est_mthd == IRM or est_mthd == IBM:

        

        # Compute GROUND TRUTH sources spectrograms--------------------------------------------------------------------------------------------------------------------------------

        for name in source_targets: 
            # compute spectrogram of target source:
            # magnitude of TF representation to the power alpha
            target_source = sources_dict[name]
            source_spectr_dict[name] = np.array( list( map( lambda chanell :  front_end["forward"](chanell) , target_source.audio.T if nb_chanels==2 else np.array([librosa.to_mono(target_source.audio.T)]) ) ) ) 
            #P[name] = np.abs( np.array( list( map( lambda chanell :  front_end["forward"](chanell) , source.audio.T ) ) ) )
            #model += P[name]

            if est_mthd==IBM:
                theta = est_mthd_params["theta"]


                #ESTIMATION---
                tmp = []
                for i in range(nb_chanels):
                    tmp.append( est_mthd(source_spectr=source_spectr_dict[name][i],mix_spectr=X[i],alpha=alpha,theta=theta) )

            else:
                #ESTIMATION---
                tmp = []
                for i in range(nb_chanels):
                    tmp.append(est_mthd(source_spectr=source_spectr_dict[name][i],mix_spectr=X[i],alpha=alpha))    
        

            extracted_source_spectr = np.array(tmp)

            # invert to time domain
            target_estimate = np.array( list( map( lambda Yj_chanell :  front_end["backward"](Yj_chanell) , extracted_source_spectr ) ) ).T[:N, :]
   
            target_estimates[name] = target_estimate if nb_chanels==2 else target_estimate.reshape(-1)



    #HPSS CASE----------------------------------------------------------------------------------------------------------------------------------
    elif est_mthd == HPSS:
        filter_len_harm = est_mthd_params["filter_len_harm"]
        filter_len_per = est_mthd_params["filter_len_per"]

         

        #ESTIMATION---
        #for name in source_targets:
        tmp = []
        for i in range(nb_chanels):
            tmp.append(est_mthd(mix_spectr=X[i],filter_len_harm=filter_len_harm,filter_len_per=filter_len_per,alpha=alpha))


        # invert to time domain
        #DRUMS:
        extracted_drums_spectr = np.array(tmp)
        target_estimate = np.array( list( map( lambda Yj_chanell :  front_end["backward"](Yj_chanell) , extracted_drums_spectr ) ) ).T[:N, :]
        target_estimates["drums"] = target_estimate if nb_chanels==2 else target_estimate.reshape(-1)


        #OTHER:
        if len(source_targets)!=1:
            extracted_other_spectr = np.array(X-tmp)
            target_estimate = np.array( list( map( lambda Yj_chanell :  front_end["backward"](Yj_chanell) , extracted_other_spectr ) ) ).T[:N, :]
            target_estimates["other"] = target_estimate if nb_chanels==2 else target_estimate.reshape(-1)

        

   
    
    return target_estimates 

    