import museval
import numpy as np
import pandas as pd
import mir_eval
import BSSeval_custom
import librosa
from orcale_methods import IRM,IBM
from HPSS import HPSS


#EVAL_METHOD-----------------------------------------------------------------------
eval_mthd_lookup ={
    "BSS_eval_mus_track":museval.eval_mus_track,
    "BSS_evaluation":museval.evaluate,
    #"mir_eval":mir_eval.separation.bss_eval_sources,
    "mir_eval":mir_eval.separation.bss_eval_images_framewise,
    "BSSeval_custom":BSSeval_custom.evaluation

}    







def pick_eval_mthd_and_eval_track(args,eval_mthd_params,musdb_track,targets,estimates_dict):

    #FUNCTION TO PRODUCE BSS_EVAL METRICS FOR ONE TRACK
    #Inputs:
    #   eval_mthd_params: a dict containig the bss_eval method params
    #   musdb_track: 
    #   estimates_dict:
    #Outputs:
    #   

    eval_mthd = eval_mthd_lookup[eval_mthd_params["eval_mthd"]] 



    # if eval_mthd_params["eval_mthd"] == "BSS_eval_mus_track":
    #     results = museval.EvalStore( frames_agg=eval_mthd_params["aggregation_method"], tracks_agg=eval_mthd_params["aggregation_method"] )

    #EVALUATION---------------------------------------------------------------------------------------------------------------------------------------
    if eval_mthd_params["nb_chan"] == 2 :
        #STEREO eval methods------------------------------------------------------------------------------------------------------------------

        if eval_mthd_params["eval_mthd"] == "BSS_eval_mus_track":
            #EVALUATE METH1 BSS_eval (eval_mus_track)--------------------------------------------------------------------
            #results.add_track(eval_mthd(track, estimates,hop=eval_mthd_params["hop"],win=eval_mthd_params["win"]))   
            track_scores = eval_mthd(musdb_track, estimates_dict,hop=eval_mthd_params["hop"],win=eval_mthd_params["win"])
          


        if eval_mthd_params["eval_mthd"] == "BSS_evaluation":
            #EVALUATE METH2  BSS_eval (evaluate)--------------------------------------------------------------------------
            # targets = list(musdb_track.sources.keys())
            refrences = np.array(list( map( lambda x : musdb_track.sources[x].audio , targets) ))        
            estimates_ndarray = np.array(list( map( lambda x : estimates_dict[x] , targets) ))
            tmp = eval_mthd(refrences,estimates_ndarray,hop=eval_mthd_params["hop"],win=eval_mthd_params["win"] )
            track_scores = np.array(tmp).T 
              

        if eval_mthd_params["eval_mthd"] == "mir_eval":
            #EVALUATE METH3  BSS_eval (mir_eval.separation.bss_eval_images_framewise)--------------------------------------------------------------------------
            # targets = list(musdb_track.sources.keys())
            refrences = np.array(list( map( lambda x : musdb_track.sources[x].audio , targets) ))        
            estimates_ndarray = np.array(list( map( lambda x : estimates_dict[x] , targets) ))
            tmp = eval_mthd(refrences,estimates_ndarray,hop=eval_mthd_params["hop"],window=eval_mthd_params["win"] ,compute_permutation=False )
            track_scores = np.array(tmp).T 


        # if eval_mthd_params["eval_mthd"] == "BSSeval_custom":
        #     #EVALUATE METH3  BSS_eval (mir_eval.separation.bss_eval_images_framewise)--------------------------------------------------------------------------
        #     # targets = list(musdb_track.sources.keys())
        #     refrences = np.array(list( map( lambda x : musdb_track.sources[x].audio , targets) ))        
        #     estimates_ndarray = np.array(list( map( lambda x : estimates_dict[x] , targets) ))
        #     tmp = eval_mthd(refrences,estimates_ndarray,hop=eval_mthd_params["hop"],window=eval_mthd_params["win"] ,compute_permutation=False )
        #     track_scores = np.array(tmp).T                  
                          


    else:
        #WE PERFORM THE SEPARATION IN THE SINGLE chanel mix (mono) 

        #MONO eval methods------------------------------------------------------------------------------------------------------------------

        if args.est_mthd_params["est_mthd_name"] == HPSS:
            refrences = np.concatenate(([np.array(musdb_track.sources["vocals"].audio+musdb_track.sources["bass"].audio+musdb_track.sources["other"].audio)],[musdb_track.sources["drums"].audio]))    
        else:    
            refrences = np.array(list( map( lambda x : musdb_track.sources[x].audio , targets) ))

        
        
        estimates_ndarray = np.array(list( map( lambda x : estimates_dict[x] , targets) ))    
        #converting the references to mono      
        refrences = np.array(list(map(lambda x : librosa.to_mono(x.T) , refrences)))      
        #estimates_ndarray = np.array(list(map(lambda x : librosa.to_mono(x.T) , estimates_ndarray)))

        #ADDING THE LAST (nb_chanels) DIMENSION
        refrences = np.array([refrences.T]).T
        estimates_ndarray = np.array([estimates_ndarray.T]).T                          

        if eval_mthd_params["eval_mthd"] == "BSS_evaluation":
            #EVALUATE METH2  BSS_eval (evaluate)--------------------------------------------------------------------------
            tmp = eval_mthd(refrences,estimates_ndarray,hop=eval_mthd_params["hop"],win=eval_mthd_params["win"])
            track_scores = np.array(tmp).T 
                

        
        if eval_mthd_params["eval_mthd"] == "mir_eval":
            #EVALUATE METH3 mir----------------------------------------------  

            tmp = eval_mthd(refrences, estimates_ndarray, hop=eval_mthd_params["hop"],window=eval_mthd_params["win"],compute_permutation=False )[:-1]
            track_scores = np.array(tmp).T 
              


        if eval_mthd_params["eval_mthd"] == "BSSeval_custom":
            #EVALUATE METH4----------------------------------------------  

            #REMOVING THE LAST (nb_chanels) DIMENSION
            refrences = refrences[:,:,0]
            estimates_ndarray = estimates_ndarray[:,:,0]   

            track_scores = np.array(  list(map( lambda tmp : eval_mthd(tmp[0],musdb_track,tmp[1],win=eval_mthd_params["win"]) , zip(estimates_ndarray,targets) ))  )

            if len(track_scores.shape)==3:
                #THEN number of sources to estimate is greater than 1 
                track_scores = track_scores.transpose(1,0,2)
            else:
                track_scores = np.array([track_scores])



    return track_scores
              