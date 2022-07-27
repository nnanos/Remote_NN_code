
import seaborn as sns
import matplotlib.pyplot as plt







def create_box_plot(methods,add_sisec_flag):



    #Add SiSEC2018 results from other participants--------------------------------
    # Open URL
    if add_sisec_flag=="True":
        methods.add_sisec18()


    #Display scores--------------------------------------------------------------------
    # important!
    df = methods.df.groupby(
        ['method', 'track', 'target', 'metric']
    ).median().reset_index()

    # Get sorting keys (sorted by median of SDR:vocals)
    df_sort_by = df[
        (df.metric == "SDR") &
        (df.target == "vocals")
    ]

    # sort methods by score
    methods_by_sdr = df_sort_by.score.groupby(
        df_sort_by.method
    ).median().sort_values().index.tolist()
        
    g = sns.FacetGrid(
        df, row="target", col="metric",
        row_order=['vocals', 'drums', 'bass', 'other'],
        height=5, sharex=False, aspect=0.8
    )
    g = (g.map(
        sns.boxplot,
        "score", 
        "method",
        orient='h',
        order=methods_by_sdr[::-1],
        showfliers=False,
        notch=True
    ).add_legend())    

    #g.fig.tight_layout()
    #plt.subplots_adjust(hspace=0.2, wspace=0.1)
    g.fig.savefig(
        "boxplot.pdf",
        bbox_inches='tight',
    )

    

