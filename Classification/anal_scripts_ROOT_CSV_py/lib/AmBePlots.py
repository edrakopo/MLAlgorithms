import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from . import EventSelection as es

import pandas as pd

sns.set_context('poster')
sns.set(font_scale=2.5)
sns.set_style("whitegrid")
sns.axes_style("darkgrid")
#xkcd_colors = ['teal','dark teal','light purple','purple','adobe','red']
#xkcd_colors = ['blue' for x in range(11)]
#xkcd_colors = xkcd_colors + ['red' for x in range(11)]
xkcd_colors = ['light purple','purple','teal','dark teal','adobe','red']
#xkcd_colors = ['dark teal','dark orange']
#xkcd_colors = ['adobe','dark orange']
#xkcd_colors = ['light blue','blue','pink','red','teal','dark teal','light purple','purple','adobe','dark orange']
sns.set_palette(sns.xkcd_palette(xkcd_colors))


def NiceBins(theax, bin_left,bin_right,value,color,llabel):
    #xkcd_colors = [color for x in range(len(value)*2)]
    #sns.set_palette(sns.xkcd_palette(xkcd_colors))
    for j,val in enumerate(value):
        if j == len(value)-1:
            theax.plot([bin_left[j],bin_right[j]],[val,val],linewidth=6,linestyle='-',label=llabel)
            theax.plot([bin_right[j],bin_right[j]],[val,0],linewidth=6,linestyle='-')
            break
        elif j == 0:
            #ax.plot([0,bin_left[j]],[0,0],linewidth=6,linestyle='-')
            theax.plot([bin_left[j],bin_left[j]],[0,val],linewidth=6,linestyle='-')
            theax.plot([bin_left[j],bin_right[j]],[val,val],linewidth=6,linestyle='-')
            theax.plot([bin_right[j],bin_right[j]],[val,value[j+1]],linewidth=6,linestyle='-')
        else:
            theax.plot([bin_left[j],bin_right[j]],[val,val],linewidth=6,linestyle='-')
            theax.plot([bin_right[j],bin_right[j]],[val,value[j+1]],linewidth=6,linestyle='-')
    return theax

def SiPMClusterDifferences(df,SiPMTimeThreshold):
    '''
    For events with a single SiPM pulse in each SiPM, get the difference in the cluster time 
    and each SiPM's peak time.  Return two arrays, one with the SiPM1 - cluster time difference and
    the second with SiPM2 - cluster time difference.
    '''
    S1Delta = []
    S2Delta = []
    for j in df.index.values:  #disgusting...
        TwoPulses = False
        if df["SiPM1NPulses"][j]!=1 or df["SiPM2NPulses"][j]!=1:
            continue
        elif abs(df["SiPMhitT"][j][0] - df["SiPMhitT"][j][1]) > SiPMTimeThreshold:
           continue 
        clusterTime = df["clusterTime"][j]
        print("SIPMNum: " + str(df["SiPMNum"][j]))
        SiPM1_hit = np.where(np.array(df["SiPMNum"][j])==1.)[0]
        SiPM2_hit = np.where(np.array(df["SiPMNum"][j])==2.)[0]
        print("SiPM1_hit: " + str(SiPM1_hit))
        S1_hitTime = np.array(df["SiPMhitT"][j])[SiPM1_hit][0]
        S2_hitTime = np.array(df["SiPMhitT"][j])[SiPM2_hit][0]
        S1Delta.append(S1_hitTime - clusterTime)
        S2Delta.append(S2_hitTime - clusterTime)
    return np.array(S1Delta),np.array(S2Delta)


def MakeClusterMultiplicityPlot(df,df_trig):
    allEvents = df_trig['eventTimeTank'].values
    ClusterMultiplicities = []
    CurrentEventNum = None
    Evs_withCluster = []
    ClusterMultiplicity = 0
    for j in df.index.values:  #disgusting...
        if CurrentEventNum is None:
            CurrentEventNum = df["eventTimeTank"][j]
            Evs_withCluster.append(df["eventTimeTank"][j])
        if CurrentEventNum != df["eventTimeTank"][j]:
            ClusterMultiplicities.append(ClusterMultiplicity)
            ClusterMultiplicity=0
            Evs_withCluster.append(df["eventTimeTank"][j])
            CurrentEventNum = df["eventTimeTank"][j]
        ClusterMultiplicity +=1
    Evs_withCluster = np.array(Evs_withCluster)
    print("NUM EVTS WITH CLUSTER: " + str(len(Evs_withCluster)))
    allEvents = np.array(allEvents)
    print("NUM EVENTS IN TRIG CLEAN: " + str(len(allEvents)))
    zero_clusters = np.setdiff1d(allEvents,Evs_withCluster)
    zeros = np.zeros(len(zero_clusters))
    MultiplicityData = np.concatenate((ClusterMultiplicities,zeros))
    return MultiplicityData 

def SiPMVariableSum(df,variable, labels, ranges):
    '''
    Return a dataframe with events that only have one SiPM pulse in each SiPM,
    where the pulse peaks are within the TimeThreshold difference.
    '''
    TotalVariable = []
    for j in df.index.values:  #disgusting...
        TotalVariable.append(np.sum(df[variable][j]))
    variableval = np.array(TotalVariable)
    if 'llabel' in labels.keys():
        plt.hist(variableval,bins=ranges['bins'],range=ranges['range'],alpha=0.5,histtype='stepfilled',linewidth=6)
        plt.hist(variableval,bins=ranges['bins'],range=ranges['range'],alpha=0.75,histtype='step',label = labels['llabel'],linewidth=6)
    else:
        plt.hist(variableval,bins=ranges['bins'],range=ranges['range'],alpha=0.5,histtype='stepfilled',linewidth=6)
        plt.hist(variableval,bins=ranges['bins'],range=ranges['range'],alpha=0.75,histtype='step',linewidth=6)
    plt.xlabel(labels["xlabel"])
    plt.ylabel(labels["ylabel"])
    plt.title(labels["title"])



def MakeSiPMVariableDistribution(df, variable, sipm_num, labels, ranges,SingleSiPMPulses):
    '''
    Plot the SiPM variable distributions SiPMhitQ, SiPMhitT, or SiPMhitAmplitude.  
    If SingleSiPMPulses is True, only plot the amplitudes for events where there was one
    of each SiPM pulse in the event.
    '''
    variableval = []
    numbers = []
    variableval = np.hstack(df[variable])
    numbers = np.hstack(df.SiPMNum)
    variableval = variableval[np.where(numbers==sipm_num)[0]]
    plt.hist(variableval,bins=ranges['bins'],range=ranges['range'],alpha=0.5,histtype='stepfilled',linewidth=6)
    plt.hist(variableval,bins=ranges['bins'],range=ranges['range'],label=labels['llabel'],alpha=0.75,histtype='step',linewidth=6)
    plt.xlabel(labels["xlabel"])
    appendage = ""
    if SingleSiPMPulses:
        appendage = "\n (Only one SiPM1 and SiPM2 pulse in an acquisition)"
    plt.title(labels["title"]+"%s"%(appendage))

def MakePMTVariableDistribution(df, variable, labels, ranges, SingleSiPMPulses):
    '''
    Plot the Tank PMT variable distributions, including hitX, hitY, hitZ, and hitT, and hitQ.  
    If SingleSiPMPulses is True, only plot entries for clusters where there was one
    of each SiPM pulse in the event.
    '''
    variableval = []
    numbers = []
    variableval = np.hstack(df[variable])
    plt.hist(variableval,bins=ranges['bins'],range=ranges['range'],label=labels['llabel'],alpha=0.8)
    plt.xlabel(labels["xlabel"])
    appendage = ""
    if SingleSiPMPulses:
        appendage = "\n (Only one SiPM1 and SiPM2 pulse in an acquisition)"
    plt.title(labels["title"]+"%s"%(appendage))

def MakeHexJointPlot(df,xvariable,yvariable,labels,ranges):
    g = sns.jointplot(x=df[xvariable],y=df[yvariable],
            kind="hex",xlim=ranges['xrange'],
            ylim=ranges['yrange'],
            joint_kws=dict(gridsize=ranges['bins']),
            stat_func=None).set_axis_labels(labels['xlabel'],labels['ylabel'])
    plt.subplots_adjust(left=0.2,right=0.8,
            top=0.90,bottom=0.2)
    cbar_ax = g.fig.add_axes([0.85,0.2,0.05,0.62])
    plt.colorbar(cax=cbar_ax)
    g.fig.suptitle(labels['title'])
    plt.show()

def MakeHexJointPlot(df,xvariable,yvariable,labels,ranges):
    g = sns.jointplot(x=df[xvariable],y=df[yvariable],
            kind="hex",xlim=ranges['xrange'],
            ylim=ranges['yrange'],
            joint_kws=dict(gridsize=ranges['bins']),
            stat_func=None).set_axis_labels(labels['xlabel'],labels['ylabel'])
    plt.subplots_adjust(left=0.2,right=0.8,
            top=0.90,bottom=0.2)
    cbar_ax = g.fig.add_axes([0.85,0.2,0.05,0.62])
    plt.colorbar(cax=cbar_ax)
    g.fig.suptitle(labels['title'])
    plt.show()

def Make2DHist(df,xvariable,yvariable,labels,ranges):
    plt.hist2d(df[xvariable],df[yvariable], bins=(ranges['xbins'],ranges['ybins']),
            range=[ranges['xrange'],ranges['yrange']],
            cmap = plt.cm.inferno)
    plt.colorbar()
    plt.title(labels['title'])
    plt.xlabel(labels['xlabel'])
    plt.ylabel(labels['ylabel'])

def Make2DHist_PEVsQ(df,labels,ranges):
    clusterPEs = []
    SiPMQ = []
    for j in df.index.values:
        if df['clusterTime'][j]>ranges['promptTime']:
            continue
        if df['clusterPE'][j]>ranges['xrange'][1]:
            continue
        clusterPEs.append(df['clusterPE'][j])
        SiPMQ.append(np.sum(df['SiPMhitQ'][j]))
    plt.hist2d(np.array(clusterPEs),np.array(SiPMQ), bins=(ranges['xbins'],ranges['ybins']),
            range=[ranges['xrange'],ranges['yrange']],
            cmap = plt.cm.inferno)
    plt.colorbar()
    plt.title(labels['title'])
    plt.xlabel(labels['xlabel'])
    plt.ylabel(labels['ylabel'])

def MakeKDEPlot(df,xvariable,yvariable,labels,ranges):
    sns.kdeplot(df[xvariable],df[yvariable],shade=True,shade_lowest=False, Label=labels['llabel'],
            cmap=labels['color'],alpha=0.7)
    plt.xlabel(labels['xlabel'])
    plt.ylabel(labels['ylabel'])
    plt.title(labels['title'])
    #def MakeHeatMap(df,xvariable,yvariable,labels):
#    g = sns.jointplot(xvariable,yvariable,data=df,kind="hex",xlim=ranges['xrange'],
#            ylim=ranges['yrange'],
#            joint_kws=dict(gridsize=ranges['bins']),
#            stat_func=None).set_axis_labels(labels['xlabel'],labels['ylabel'])
#    plt.subplots_adjust(left=0.2,right=0.8,
#            top=0.90,bottom=0.2)
#    cbar_ax = g.fig.add_axes([0.85,0.2,0.05,0.62])
#    plt.colorbar(cax=cbar_ax)
#    g.fig.suptitle(labels['title'])
#    plt.show()
