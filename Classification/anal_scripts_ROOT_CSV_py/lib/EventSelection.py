import numpy as np
import copy
import lib.ROOTProcessor as rp
import pandas as pd

def EstimateLivetime(filelist):
    '''
    Estimate live time using the smallest and 
    largest time stamps in each separate file.  One or two 
    events are being set to an unphysically small or large number though,
    have to investigate.
    '''
    total_time = 0
    mybranches = ['eventTimeTank']
    for f1 in filelist:
        f1Processor = rp.ROOTProcessor(treename="phaseIITriggerTree")
        f1Processor.addROOTFile(f1,branches_to_get=mybranches)
        f1data = f1Processor.getProcessedData()
        f1data_pd = pd.DataFrame(f1data)
        early_time = np.min(f1data_pd.loc[(f1data_pd["eventTimeTank"]>1E6)].values)/1E9
        late_time = np.max(f1data_pd.loc[(f1data_pd["eventTimeTank"]<2.0E18)].values)/1E9
        print("EARLY_TIME: " + str(early_time))
        print("LATE_TIME: " + str(late_time))
        print("LATE - EARLY TIME: " + str(late_time - early_time))
        total_time+=(late_time-early_time)
    return total_time



#Some methods for returning a dataframe cleaned based on event selection

def SingleSiPMPulses(df):
    '''
    Return a dataframe with events that only have one SiPM pulse in each SiPM
    '''
    newdf = df.loc[(df["SiPM1NPulses"]==1) & (df["SiPM2NPulses"]==1)]
    return newdf.reset_index(drop=True)

def SingleSiPMPulsesDeltaT(df,TimeThreshold):
    '''
    Return a dataframe with events that only have one SiPM pulse in each SiPM,
    where the pulse peaks are within the TimeThreshold difference.
    '''
    DirtyTriggers = []
    for j in df.index.values:  #disgusting...
        TwoPulses = False
        if df["SiPM1NPulses"][j]!=1 or df["SiPM2NPulses"][j]!=1:
            DirtyTriggers.append(df["eventTimeTank"][j])
        else:
            if abs(df["SiPMhitT"][j][0] - df["SiPMhitT"][j][1]) > TimeThreshold:
                DirtyTriggers.append(df["eventTimeTank"][j])
    CleanIndices = []
    for j in df.index.values:  #disgusting...
        if df["eventTimeTank"][j] not in DirtyTriggers:
            CleanIndices.append(j)
    CleanIndices = np.array(CleanIndices)
    newdf = df.loc[CleanIndices]
    newdf.reset_index(drop=True)
    return newdf

def SiPMCorrelatedPrompts(df,SiPMTimeThreshold,CTimeThreshold,promptWindow):
    '''
    Return a dataframe with clusters that occur within CTimeThreshold ns of 
    two SiPM pulses (occurring within SiPMTimeThreshold) at any time earlier than 
    the defined promptWindow time.
    '''
    DirtyTriggers = []
    for j in df.index.values:  #disgusting...
        TwoPulses = False
        if df["SiPM1NPulses"][j]!=1 or df["SiPM2NPulses"][j]!=1:
            DirtyTriggers.append(j)
        elif abs(df["SiPMhitT"][j][0] - df["SiPMhitT"][j][1]) > SiPMTimeThreshold:
            DirtyTriggers.append(j)
        else:
            SiPMMeanTime = (df["SiPMhitT"][j][0] + df["SiPMhitT"][j][1])/2.
            if abs(df['clusterTime'][j]-SiPMMeanTime)> CTimeThreshold or df['clusterTime'][j]>promptWindow:
                DirtyTriggers.append(j)
    newdf = df.loc[~df.index.isin(DirtyTriggers)].reset_index(drop=True)
    return newdf

def SiPMCorrelatedPrompts_WholeFile(df_cluster,df_trig):
    '''
    Return a trigger DataFrame which shares eventTimeTank values with any of the 
    df_cluster entries.
    '''
    #Get Tank Trigger times that have a prompt cluster in the first two microseconds
    ClusterEventTimes = set(df_cluster["eventTimeTank"].values)
    CleanIndices = []
    for j in df_trig.index.values:  #disgusting...
        if df_trig["eventTimeTank"][j] in ClusterEventTimes:
            CleanIndices.append(j)
    CleanIndices = np.array(CleanIndices)
    return df_trig.loc[CleanIndices].reset_index(drop=True)


def HasVetoHit_TankClusters(df_cluster,df_trig):
    '''
    Return a filtered trigger DataFrame which has all triggers that either
    have no cluster at all, or only have clusters in the time greater than
    clusterTimeCut.
    '''
    #Get Tank Trigger times that have a prompt cluster in the first two microseconds
    VetoEvents = []
    for j in df_trig.index.values:  #disgusting...
        if df_trig["vetoHit"][j] == 1:
            VetoEvents.append(df_trig["eventTimeTank"][j])
    #Get indices for trigger entries that don't have an event time in DirtyPromptEvents
    VetoTankClusters = []
    for j in df_cluster.index.values:  #disgusting...
        if df_cluster["eventTimeTank"][j] in VetoEvents:
            VetoTankClusters.append(j)
    VetoTankClusters = np.array(VetoTankClusters)
    return df_cluster.loc[VetoTankClusters].reset_index(drop=True)

def NoPromptClusters_WholeFile(df_cluster,df_trig,clusterTimeCut):
    '''
    Return a filtered trigger DataFrame which has all triggers that either
    have no cluster at all, or only have clusters in the time greater than
    clusterTimeCut.
    '''
    #Get Tank Trigger times that have a prompt cluster in the first two microseconds
    DirtyPromptEvents = []
    for j in df_cluster.index.values:  #disgusting...
        if df_cluster["clusterTime"][j] < clusterTimeCut:
            DirtyPromptEvents.append(df_cluster["eventTimeTank"][j])
    #Get indices for trigger entries that don't have an event time in DirtyPromptEvents
    CleanIndices = []
    for j in df_trig.index.values:  #disgusting...
        if df_trig["eventTimeTank"][j] not in DirtyPromptEvents:
            CleanIndices.append(j)
    CleanIndices = np.array(CleanIndices)
    return df_trig.loc[CleanIndices].reset_index(drop=True)

def NoBurst_WholeFile(df_cluster,df_trig,timeWindowCut,clusterPECut):
    '''
    Return a filtered trigger DataFrame which has all triggers that have no
    cluster with a PE greater than clusterPECut input for times later than timeWindowCut.'''
    #Get Tank Trigger times that have a prompt cluster in the first two microseconds
    DirtyEvents = []
    for j in df_cluster.index.values:  #disgusting...
        if df_cluster["clusterPE"][j] > clusterPECut and df_cluster["clusterTime"][j] >timeWindowCut:
            DirtyEvents.append(df_cluster["eventTimeTank"][j])
    #Get indices for trigger entries that don't have an event time in DirtyPromptEvents
    CleanIndices = []
    for j in df_trig.index.values:  #disgusting...
        if df_trig["eventTimeTank"][j] not in DirtyEvents:
            CleanIndices.append(j)
    CleanIndices = np.array(CleanIndices)
    return df_trig.loc[CleanIndices].reset_index(drop=True)

def NoPromptClusters(df_cluster,clusterTimeCut):
    '''
    Return clusters from events that have no prompt cluster with a time earlier
    than the TimeCut variable.
    '''
    DirtyPromptEvents = []
    for j in df_cluster.index.values:  #disgusting...
        if df_cluster["clusterTime"][j] < clusterTimeCut:
            DirtyPromptEvents.append(df_cluster["eventTimeTank"][j])
    CleanIndices = []
    for j in df_cluster.index.values:  #disgusting...
        if df_cluster["eventTimeTank"][j] not in DirtyPromptEvents:
            CleanIndices.append(j)
    CleanIndices = np.array(CleanIndices)
    return df_cluster.loc[CleanIndices].reset_index(drop=True)

def NoBurstClusters(df_cluster,clusterTimeCut,maxPE):
    '''
    Return clusters from events that have no PE cluster with a time later than
    than the TimeCut variable AND PE greater than the PE variable..
    '''
    DirtyEvents = []
    for j in df_cluster.index.values:  #disgusting...
        if df_cluster["clusterTime"][j] > clusterTimeCut and \
        df_cluster["clusterPE"][j] > maxPE:
            DirtyEvents.append(df_cluster["eventTimeTank"][j])
    CleanIndices = []
    for j in df_cluster.index.values:  #disgusting...
        if df_cluster["eventTimeTank"][j] not in DirtyEvents:
            CleanIndices.append(j)
    CleanIndices = np.array(CleanIndices)
    return df_cluster.loc[CleanIndices].reset_index(drop=True)

def FilterByEventNumber(df,eventnums):
    ReturnIndices = []
    for j in df.index.values:  #disgusting...
        if df["eventTimeTank"][j] in eventnums:
            ReturnIndices.append(j)
    return df.loc[np.array(ReturnIndices)].reset_index(drop=True)

def FilterByEventTime(df,eventnums):
    ReturnIndices = []
    for j in df.index.values:  #disgusting...
        if df["eventTimeTank"][j] in eventnums:
            ReturnIndices.append(j)
    return df.loc[np.array(ReturnIndices)].reset_index(drop=True)

def ValidPromptClusterEvents(df,clusterTimeCut):
    '''
    Given a dataframe and prompt cut, return all clusters associated
    with an event that only has one SiPM pulse per pulse and no clusters
    at less than the clusterTimeCut variable.
    '''
    NoPromptClusterDict = {}
    OnePulses = np.where((df.SiPM1NPulses==1) & (df.SiPM2NPulses==1))[0]
    DirtyPromptEvents = []
    for j in df.index.values:  #disgusting...
        if df["clusterTime"][j] < clusterTimeCut:
            DirtyPromptEvents.append(df["eventTimeTank"][j])
    CleanIndices = []
    for j in df.index.values:  #disgusting...
        if df["eventTimeTank"][j] not in DirtyPromptEvents:
            CleanIndices.append(j)
    CleanIndices = np.array(CleanIndices)
    Cleans = np.intersect1d(OnePulses,CleanIndices)
    df_CleanPrompt = df.loc[Cleans]
    return df_CleanPrompt.reset_index(drop=True)


########## BEAM-RELATED EVENT SELECTION FUNCTIONS ###########

def MaxPEClusters(df):
    '''
    Prune down the data frame to clusters that only have the largest photoelectron
    value.
    '''
    CurrentEventNum = None
    HighestPE_indices = []
    HighestPE = 0
    for j in df.index.values:  #disgusting...
        if CurrentEventNum is None:
            CurrentEventNum = df["eventTimeTank"][j]
            HighestPE = df["clusterPE"][j]
            HighestPE_index = j
        if CurrentEventNum != df["eventTimeTank"][j]:
            HighestPE_indices.append(HighestPE_index)
            HighestPE = df["clusterPE"][j]
            HighestPE_index = j
            CurrentEventNum = df["eventTimeTank"][j]
        else:
            if df["clusterPE"][j] > HighestPE:
                HighestPE = df["clusterPE"][j]
                HighestPE_index = j
    HighestPE_indices = np.array(HighestPE_indices)
    return df.loc[HighestPE_indices].reset_index(drop=True)

def MaxHitClusters(df):
    '''
    Prune down the data frame to clusters that only have the largest clusterHits 
    value.
    '''
    CurrentEventNum = None
    HighestNhit_indices = []
    HighestNhit = 0
    for j in df.index.values:  #disgusting...
        if CurrentEventNum is None:
            CurrentEventNum = df["eventTimeTank"][j]
            HighestNhit = df["clusterHits"][j]
            HighestNhit_index = j
        if CurrentEventNum != df["eventTimeTank"][j]:
            HighestNhit_indices.append(HighestNhit_index)
            HighestNhit = df["clusterHits"][j]
            HighestNhit_index = j
            CurrentEventNum = df["eventTimeTank"][j]
        else:
            if df["clusterHits"][j] > HighestNhit:
                HighestNhit_index = j
    HighestNhit_indices = np.array(HighestNhit_indices)
    return df.loc[HighestNhit_indices].reset_index(drop=True)

def MatchingEventTimes(df1,df2):
    '''
    Take two dataframes and return an array of clusterTimes that 
    come from clusters with matching eventTimeTanks.
    '''
    PMTIndices = []
    MRDIndices = []
    for j in df1.index.values:
        eventTime = df1["eventTimeTank"][j]
        Match = np.where(df2["eventTimeTank"].values == eventTime)[0]
        if len(Match) > 0:
            PMTIndices.append(j)
            MRDIndices.append(Match[0])
        if len(Match) > 1:
            print("OH SHIT, WHY ARE THERE MULTIPLE CLUSTERS AT THIS TIME NOW???")
            print("EVENT TIME TANK IS " + str(eventTime))
    return np.array(PMTIndices), np.array(MRDIndices)

def SingleTrackSelection(df1,trackRadius,trackAngle,trackDepth):
    '''
    Return a dataFrame with MRD clusters that have one track, and have an entry point
    within the trackRadius, angle less than track angle (radians), and a track 
    depth than the trackDepth (cm).
    '''
    df1_singleTracks = df1.loc[df1["numClusterTracks"]==1].reset_index(drop=True)
    valid_singleTrackInds = []
    for j in df1_singleTracks.index.values:
        radius = df1_singleTracks["MRDEntryPointRadius"][j][0]
        angle = df1_singleTracks["MRDTrackAngle"][j][0]
        depth = df1_singleTracks["MRDPenetrationDepth"][j][0]
        if radius < trackRadius and angle < trackAngle and depth > trackDepth:
            valid_singleTrackInds.append(j)
    return np.array(valid_singleTrackInds)

def SingleTrackEnergies(df1):
    '''
    Return a dataFrame with MRD clusters that have one track, and have an entry point
    within the trackRadius, angle less than track angle (radians), and a track 
    depth than the trackDepth (cm).
    '''
    Energies = []
    for j in df1.index.values:
        Energies.append(df1['MRDEnergyLoss'][j][0])
    return np.array(Energies)
