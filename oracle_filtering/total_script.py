from statistics import median
#from turtle import backward
import musdb
import museval
import numpy as np
import functools
import argparse
import nsgt as nsg
import librosa
import scipy.signal
import time
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
import mir_eval
import BSSeval_custom
import pandas as pd
import scipy.signal as sg
from Audio_proc_lib.audio_proc_functions import load_music,sound_write
from Determine_Front_end import *
from Determine_Evaluation_method import *
from Determine_separation_method import *


#from create_and_save_boxplot import create_box_plot


#COMPLEXITY (FOR ONE SONG):
#2*5 STFTS + 2*4 ISTFTS (2X because the method acts on dual chanel signal) 

#COMPARE ALL THE FRONT_ENDS FOR THE VARIOUS TF PARAMETERS 
# IN THE SMALL DATASET AND IF YOU SEE THAT SOME TRANSFORM IS BETTER THAN ALL
#  THE OTHERS (MAINLY THE STFT) THEN GO TO FULL LENGTH EVALUATION (BECAUSE ITS SO MEMORY DEMANDING) 

#TODO 
#2) VIZUALIZE THE EVAL WITH BOXPLOTS (for any eval method)
#3) PROPER JSON LOG OUTPUT 
#4)RANDOMIZE THE EXPERTS (WITH A MAGIC WAY)
   

#SXOLIO :
# (YPO8ESH)! PLHROFORIA KAI 8ORYBOS SE DIAFORETIKES BADES 
#KAI EPEIDH TA SHMATA MOUSIKHS (TA VLEPOYME SAN STOXASTIKH DIADIKASIA) EINAI MH STASIMA XRONIKA KAI SUXNOTIKA  (8ELEI EKSHGHSH)
# EPOMENWS PROSEGGISH TOU PROBLHMATOS ME (GRAMMIKO XRONIKA METAVALLOMENO FILTRARISMA)! 
# OPOU AUTOU TOU EIDOUS TO FILTRARISMA YLOPOIEITAI ME TO TF-MASKING (YLOPOIEI MIA KLASH APO XRONIKA METAVALLOMENOUS CONVOLUTIONAL OPERATORS) 
#DES 
  

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






if __name__ =='__main__':

    import time
    import yaml
    import argparse
    import json
    # from IRM import IRM
    # from IBM import IBM
    # from MWF import MWF
    # from GT import GT
    # from MIX import MIX
    # from IRM_method import IRM
    



    parser = argparse.ArgumentParser(description='Command description.')



    parser.add_argument('--add_sisec', type=str, default="False",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ')                            


    parser.add_argument('-a', '--est_mthd_params', type=yaml.safe_load,
                            help='provide the estimation method (for the mask) parameters as a quoted json sting')    


    parser.add_argument('-b', '--eval_mthd_params', type=yaml.safe_load,
                            help='provide evaluation method parameters as a quoted json sting')    

    parser.add_argument('-c', '--Dataset_params', type=yaml.safe_load,
                            help='provide Dataset parameters as a quoted json sting')    


    parser.add_argument('-d', '--front_end_params', type=yaml.safe_load,
                            help='provide Transform parameters as a quoted json sting')    


    #yaml.unsafe_load

    args, _ = parser.parse_known_args()




    #DATASET----------------------------------------
    nb_tracks = args.Dataset_params["nb_tracks"]
    subset = args.Dataset_params["subset"]
    if not(args.Dataset_params["Full_songs"]):
        
        mus = musdb.DB(download=True,subsets=subset)
    else:
        root = "/home/nnanos/musdb18_wav_small"
        mus = musdb.DB(root=root,
        is_wav=True,
        subsets=subset,
        )


    sources_targets = list( map( lambda key : key , args.Dataset_params["sources_targtets"].keys() ) )




    #ESTIMATION_METHOD-----------------------------------------------------------------------
    # oracle_mthd_lookup ={
    #     "IRM":IRM,
    #     "IBM":IBM,
    #     "MWF":MWF,
    #     "MIX":MIX
    #     #"HPSS":HPSS

    # }    
    #oracle_mthd = oracle_mthd_lookup[args.est_mthd_params["est_mthd_name"]]
    
    est_mthd_params = args.est_mthd_params
    
    #------------------------------------------------------------------------------------------    


    #SEPARATE and EVALUATE-----------------------------------------------------
    t1 = cputime()


    #ITERATE OVER THE MUS DATASET_-------------------------------------------------------------------------------------------------------------

    k=0
    #scores contains the metrics for each track
    metrics_per_track = []

    for track in mus:

        

        # #DETERMINING FRONT END---------------------------------------------------------------------------------------------------
        front_end_for_back = pick_front_end( args=args , front_end_params = args.front_end_params , musdb_track=mus[k] ) 
        #-------------------------------------------------------------------------

        
        #SEPARATION----------------------------------------------------------------------------------------------
        #mask = pick_sep_mthd_and_estimate_mask(sources_names_list=sources_targets,track=track,front_end=front_end_for_back,nb_chan=args.eval_mthd_params["nb_chan"])
        
        estimates_dict = pick_sep_mthd_and_est_soucres(args=args,est_mthd_params=est_mthd_params,track=track,source_targets=sources_targets,front_end=front_end_for_back,nb_chanels=args.eval_mthd_params["nb_chan"])

        #estimates_dict = oracle_mthd(sources_names_list=sources_targets,track=track,front_end=front_end_for_back,nb_chan=args.eval_mthd_params["nb_chan"])
        
        #-------------------------------------------------------------------------


        #EVALUATION---------------------------------------------------------------------------------------------------------------------------------------
        metrics_per_track.append( pick_eval_mthd_and_eval_track(args=args,eval_mthd_params=args.eval_mthd_params,musdb_track=mus[k],targets=sources_targets,estimates_dict=estimates_dict) )
        #-------------------------------------------------------------------------

    
        k+=1
        
        if k==nb_tracks:
            break


    #AGGRAGATE METRICS FOR EACH TRACK OVER FRAMES-----------------------------------------------------------------------------------------
    if args.eval_mthd_params["eval_mthd"] == "BSS_eval_mus_track":
        pass
    else:

        aggregation_method_lookup = {
            "median":np.median,
            "mean":np.mean
        }        
        aggregation_method_func = aggregation_method_lookup[args.eval_mthd_params["aggregation_method"]]

        track_scores_agg_over_frames = []
    


        for k in range(nb_tracks):

            tmp_agg_scores = aggregation_method_func(metrics_per_track[k],0)

            #track_scores_df_tmp = pd.DataFrame(data= tmp_agg_scores , index=targets , columns = ["SDR","ISR","SIR","SAR"]  ) 
            
            track_scores_agg_over_frames.append( tmp_agg_scores )  


    #AGGRAGATE METRICS OVER TRACKS-----------------------------------------------------------------------------------------
    if args.eval_mthd_params["eval_mthd"] == "BSS_eval_mus_track":
        results = museval.EvalStore( frames_agg=args.eval_mthd_params["aggregation_method"], tracks_agg=args.eval_mthd_params["aggregation_method"] )
        for k in range(nb_tracks):
            results.add_track(metrics_per_track[k])

    else:

        metrics_agg_over_frames_tracks = aggregation_method_func(np.array(track_scores_agg_over_frames),0) 

        if args.eval_mthd_params["eval_mthd"] == "BSSeval_custom":
            metrics_agg_over_frames_tracks = pd.DataFrame(data=metrics_agg_over_frames_tracks , index=sources_targets , columns = ["SDR","SIR","SAR"]  )  

        elif args.eval_mthd_params["eval_mthd"] == "mir_eval":
            metrics_agg_over_frames_tracks = metrics_agg_over_frames_tracks[:,:metrics_agg_over_frames_tracks.shape[1]-1]
            metrics_agg_over_frames_tracks = pd.DataFrame(data=metrics_agg_over_frames_tracks , index=sources_targets , columns = ["SDR","ISR","SIR","SAR"]  )            
            
        else:
            metrics_agg_over_frames_tracks = pd.DataFrame(data=metrics_agg_over_frames_tracks , index=sources_targets , columns = ["SDR","ISR","SIR","SAR"]  )            

        results = metrics_agg_over_frames_tracks
    




    #PRINTING--------------------------------------------------------------------------------------------------
    t2 = cputime()
    print(vars(args))
    print(results)
    print("Calculation time ("+args.est_mthd_params["est_mthd_name"]+  "+ EVALUATION): %.3fs"%(t2-t1) )


    #CREATING JSON LOG-------------------------------------------------------------------------------------------------
    json_log = vars(args)
    # json_log["metrics"] = {
    #         "metrics_agg_over_frames_tracks":metrics_agg_over_frames_tracks.to_json(),
    #         "metrics_per_track_agg_over_frames":{
    #             "track1":{
                    
    #             }
    #         }
    #     }

    #json_log["metrics_agg_over_frames_tracks"] = results.to_json()
    json_log["calc_time"] = {
        "calc_time_for_"+args.est_mthd_params["est_mthd_name"]+"+EVALUATION procedure" : t2-t1 ,
        "calc_time_for_"+args.est_mthd_params["est_mthd_name"]+"_for_one_song_and_"+"for_all_the_sources"  :  None,
        "calc_time_for_evaluation_for_one_song_and_"+"for_all_the_sources" :  None,
    }

    # #VISUALIZE THE METRICS BY CREATING BOX_PLOTS-----------------------------------------------------------------------------------------------------------------------------
    # methods = museval.aggregate.MethodStore()
    # methods.add_evalstore(results, name='Oracle_'+args.front_end_params["front_end_name"])
    # print(methods.df)


    # create_box_plot(methods,args.add_sisec)





    

    #SAVING-------------------------------------------------------------------------------
    import pickle

    #saving the results variable
    with open(args.front_end_params["front_end_name"]+'_'+args.eval_mthd_params["eval_mthd"]+args.est_mthd_params["est_mthd_name"]+'.pkl', 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)


    #savving the json produced
    with open('JSON_LOG_FOR'+'_'+args.front_end_params["front_end_name"]+'_'+args.eval_mthd_params["eval_mthd"]+args.est_mthd_params["est_mthd_name"]+'.json', 'wb', 'w') as outfile:
        json.dump(json_log, outfile)

    #read results
    # with open(front_end_str+'_IRM_results'+'.pkl', 'rb') as inp:
    #     results2 = pickle.load(inp)


    a = 0




