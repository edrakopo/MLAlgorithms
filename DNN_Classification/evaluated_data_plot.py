import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Make2DHist(df,xvariable,yvariable,labels,ranges):
    plt.hist2d(df[xvariable],df[yvariable], bins=(ranges['xbins'],ranges['ybins']),
              range=[ranges['xrange'],ranges['yrange']],
              cmap = plt.cm.turbo)
    plt.colorbar()
    plt.title(labels['title'])
    plt.xlabel(labels['xlabel'])
    plt.ylabel(labels['ylabel'])
    

def Plots(df, Type, Range, Misc, number):
   #--- CB to clusterPE:
    labels = {'title': f'{Type}: Charge balance parameters in time window \nfor CB range {Range} {Misc} ','xlabel': 'Cluster PE', 'ylabel': 'Charge balance'}
    ranges = {'xbins': 200, 'ybins':200, 'xrange':[0,200],'yrange':[0,1]}
    Make2DHist(df,'3','5',labels,ranges)
    plt.savefig(f'NewPlots/{Type}_{Range}_CB_PE_{Misc}.png')
    plt.show()

    #--- CB to cluster Time:   
    labels = {'title': f'{Type}: Charge balance parameters in time window \nfor CB range {Range} {Misc}', 'xlabel': 'Cluster time (ns)', 'ylabel': 'Charge balance'}
    ranges = {'xbins': 200, 'ybins':200, 'xrange':[0,20000],'yrange':[0,1]}
    Make2DHist(df,'2','5',labels,ranges)
    plt.savefig(f'NewPlots/{Type}_{Range}_CB_time_{Misc}.png')
    plt.show()
    
    #---  PMTids plots:
    
    rest, IDs, y = np.split(df,[10,3000],axis=1)
    print(IDs)
    print(type(IDs))
    ids = IDs.to_numpy()
    print(ids)
    print(type(ids))
    df_IDs = ids[ids != 0]
    print(df_IDs)
    print(type(df_IDs))
    #df_IDs = df.iloc[:, 10:3000]
    plt.hist(df_IDs, bins = number, density=True)
    plt.xlim(320, 470)
    plt.title(f'{Type}: PMT IDs {Misc}')
    plt.xlabel('PMT IDs')
    plt.savefig(f'NewPlots/{Type}_PMTid_{Misc}.png')
    plt.show()
    '''
    #-------- 1d histograms
    
    #--- cluster PE:
    plt.hist(df['3'],bins=200,range=(0,200), histtype='step', density=True)
    plt.title(f'1D Hist for clusterPE for {Type} \nin CB range {Range} {Misc}')
    plt.xlabel('Cluster PE')
    plt.savefig(f'NewPlots/{Type}_{Range}_pe_1D.png')
    plt.show()
    
    #--- cluster time:
    plt.hist(df['2'],bins=200,range=(0,20000), histtype='step', density=True)
    plt.title(f'1D Hist for clusterTime for {Type} \nin CB range {Range} {Misc}')
    plt.xlabel('Cluster Time')
    plt.savefig(f'NewPlots/{Type}_{Range}_time_1D.png')
    plt.show()
    '''

if __name__=='__main__':
    
    #data = pd.read_csv("predicted_data.csv")
    data = pd.read_csv("predicted_data_evaluate.csv")
    print(data.head())
    Sdf = data.loc[data['TrueY']==1]    #select events with trueY as signal
    Bdf = data.loc[data['TrueY']==0]    #select events with trueY as background
    
    TSig = Sdf.loc[(data['Predicted']==1)]  #select signal events predicted as signal
    FSig = Bdf.loc[(data['Predicted']==1)]  #select signal events predicted as background
    TBkgd = Bdf.loc[(data['Predicted']==0)] #select bacground events predicted as background
    FBkgd = Sdf.loc[(data['Predicted']==0)] #select bacground events predicted as signal
    
    
    
    '''
    #------select each event once----
    events = rafale['1']
    def get_unique_numbers(numbers):

        list_of_unique_numbers = []
        unique_numbers = set(numbers)
        for number in unique_numbers:
            list_of_unique_numbers.append(number)

        return list_of_unique_numbers
    
    unique_events = get_unique_numbers(events)
    #print(len(unique_events))
    unique_events = list(unique_events)
    #sabre = rafale.loc[rafale['1'] == unique_events].reset_index(drop=True)
    for i in rafale:
        for j in unique_events:
            if rafale.iloc[i] == unique_events[j]:
                sabre.append(rafale.iloc[i])
    print(sabre.shape)
    
    #----hard cuts to remove afterpulses from events-----
    
    Noapsignal1 = TSig.loc[(TSig['2']<6000)].reset_index(drop=True)
    Noapsignal2 = TSig.loc[(TSig['2']>10000)].reset_index(drop=True)
    NoapsignalT = pd.concat((Noapsignal1,Noapsignal2))
    
    
    Noapsignal1 = FSig.loc[(FSig['2']<6000)].reset_index(drop=True)
    Noapsignal2 = FSig.loc[(FSig['2']>10000)].reset_index(drop=True)
    NoapsignalF = pd.concat((Noapsignal1,Noapsignal2))
    
    
    noapbkgd1 = TBkgd.loc[(TBkgd['2']<6000)].reset_index(drop=True)
    noapbkgd2 = TBkgd.loc[(TBkgd['2']>10000)].reset_index(drop=True)
    noapbkgdT = pd.concat((noapbkgd1,noapbkgd2))


    noapbkgd1 = FBkgd.loc[(FBkgd['2']<6000)].reset_index(drop=True)
    noapbkgd2 = FBkgd.loc[(FBkgd['2']>10000)].reset_index(drop=True)
    noapbkgdF = pd.concat((noapbkgd1,noapbkgd2))
    
    gulag = noapbkgdF.loc[noapbkgdF['2']<1200].reset_index(drop=True) #cuts to test the low cb -low clustertime false backgroud(former signal)---cosmics THEY ARE
    gulag = gulag.loc[gulag['5']<0.15].reset_index(drop=True)
    
    spitfire = NoapsignalT.loc[NoapsignalT['5']>=0.6].reset_index(drop=True) #cuts to test the high cb low clustertime of signal, probably afterpulses
    spitfire = spitfire.loc[spitfire['2']<1200].reset_index(drop=True)
    
    mirage = noapbkgdT.loc[noapbkgdT['5']>=0.6].reset_index(drop=True)
    mirage = mirage.loc[mirage['2']<1200].reset_index(drop=True)
    '''
    '''
    NoapsignalT = NoapsignalT.loc[(NoapsignalT['5']>=0.4) & (NoapsignalT['5']<=0.6)]
    NoapsignalF = NoapsignalF.loc[(NoapsignalF['5']>=0.4) & (NoapsignalF['5']<=0.6)]
    noapbkgdT = noapbkgdT.loc[(noapbkgdT['5']>=0.4) & (noapbkgdT['5']<=0.6)]
    noapbkgdF = noapbkgdF.loc[(noapbkgdF['5']>=0.4) & (noapbkgdF['5']<=0.6)]
    '''
    
    '''
    Plots(rafale, 'True background PMT450','any', 'rafale cut', 1500)
    Plots(mirage, 'True Background Mirage', '>0.6', 'Mirage Cut', 1500)
    Plots(spitfire, 'True Signal Spitfire', '>0.6', 'Spitfire Cut',1500)    
    Plots(NoapsignalT, 'True Signal', '0.4-0.6', 'NO AFTERPULSES',1500)
    '''
    
    Plots(TSig, 'True Signal', 'all range', 'good loss function',1500) #variables are: (dataframe), (what type of data), (cb range), (random info to pass into the plot headline), (number of bins for the pmtid plots)
    Plots(TBkgd, 'True Background', 'all range', 'good loss function',1500)
    Plots(FBkgd, 'False Background', 'all range', 'good loss function',1500)
    Plots(FSig, 'False Signal', 'all range', 'good loss function',1500)
    

 

