import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing

#------- Read .csv files -------

var_info = pd.read_csv("data/AddEvInfo_single_Original.csv", header = None)
predictions = pd.read_csv("predictions/MLP_predictions_single_noDistsVarSkewKurt.csv", header = None)

nhits = var_info.iloc[:,1]
charge = var_info.iloc[:,2]
time = var_info.iloc[:,3]
bary = var_info.iloc[:,4]
rms = var_info.iloc[:,5]
distVert = var_info.iloc[:,6]
distHor = var_info.iloc[:,7]
energy = var_info.iloc[:,8]

true = predictions.iloc[:,1]
pred = predictions.iloc[:,2]

print("nhits: ",nhits)
print("charge: ",charge)
print("time: ",time)
print("bary: ",bary)
print("rms: ",rms)
print("distVert: ",distVert)
print("distHor: ",distHor)
print("energy: ",energy)
print("true: ",true)
print("pred: ",pred)

#------- Select correct and wrong prediction data sets -------

nhits_correct = list()
charge_correct = list()
time_correct = list()
bary_correct = list()
rms_correct = list()
distVert_correct = list()
distHor_correct = list()
energy_correct = list()


nhits_incorrect = list()
charge_incorrect = list()
time_incorrect = list()
bary_incorrect = list()
rms_incorrect = list()
distVert_incorrect = list()
distHor_incorrect = list()
energy_incorrect = list()

for i in range(len(true)):
    #print(true[i],pred[i])
    if true[i]==pred[i]:
        nhits_correct.append(nhits[i])
        charge_correct.append(charge[i])
        time_correct.append(time[i])
        bary_correct.append(bary[i])
        rms_correct.append(rms[i])
        distVert_correct.append(distVert[i])
        distHor_correct.append(distHor[i])
        energy_correct.append(energy[i])
    else:
        nhits_incorrect.append(nhits[i])
        charge_incorrect.append(charge[i])
        time_incorrect.append(time[i])
        bary_incorrect.append(bary[i])
        rms_incorrect.append(rms[i])
        distVert_incorrect.append(distVert[i])
        distHor_incorrect.append(distHor[i])
        energy_incorrect.append(energy[i])        
    
#print(nhits_correct)
def getRatio(bin1,bin2):
    # Sanity check
    if len(bin1) != len(bin2):
        print("Cannot make ratio!")
    bins = []
    for b1,b2 in zip(bin1,bin2):
        if b1==0 and b2==0:
            bins.append(1.)
        elif b2==0:
            bins.append(0.)
        else:	
            bins.append(float(b1)/float(b2))
    # The ratio can of course be expanded with eg. error 
    return bins

#---------------------------------------------
#---histograms with ratio incorrect/correct---
#---------------------------------------------

bins_nhits = np.arange(10,140,14)
_bins, _edges = np.histogram(nhits_correct,bins_nhits)
_bins2, _edges2 = np.histogram(nhits_incorrect,bins_nhits)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.plot(edges,bins,'o-',color="blue",lw=2,label="correct")
ax.plot(edges2,bins2,'o-',color="orange",lw=2,label="incorrect")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
ax.plot(edges,rat)
ax.set_ylim(0,1)
ax.set_xlabel("n hit PMTs")
ax.set_ylabel("Incorrect classification")
plt.savefig("MisclassifiedRatio_hitPMTs.pdf")
plt.close("all")


bins_charge = np.arange(0,12500,625)
_bins, _edges = np.histogram(charge_correct,bins_charge)
_bins2, _edges2 = np.histogram(charge_incorrect,bins_charge)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.plot(edges,bins,'o-',color="blue",lw=2,label="correct")
ax.plot(edges2,bins2,'o-',color="orange",lw=2,label="incorrect")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
ax.plot(edges,rat)
ax.set_ylim(0,1)
ax.set_xlabel("charge [p.e.]")
ax.set_ylabel("Incorrect classification")
plt.savefig("MisclassifiedRatio_charge.pdf")
plt.close("all")

bins_time = np.arange(0,1500,75)
_bins, _edges = np.histogram(time_correct,bins_time)
_bins2, _edges2 = np.histogram(time_incorrect,bins_time)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.plot(edges,bins,'o-',color="blue",lw=2,label="correct")
ax.plot(edges2,bins2,'o-',color="orange",lw=2,label="incorrect")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
ax.plot(edges,rat)
ax.set_ylim(0,1)
ax.set_xlabel("average hit time [ns]")
ax.set_ylabel("Incorrect classification")
plt.savefig("MisclassifiedRatio_time.pdf")
plt.close("all")

bins_time_fine = np.arange(0,400,25)
_bins, _edges = np.histogram(time_correct,bins_time_fine)
_bins2, _edges2 = np.histogram(time_incorrect,bins_time_fine)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.plot(edges,bins,'o-',color="blue",lw=2,label="correct")
ax.plot(edges2,bins2,'o-',color="orange",lw=2,label="incorrect")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
ax.plot(edges,rat)
ax.set_ylim(0,1)
ax.set_xlabel("average hit time [ns]")
ax.set_ylabel("Incorrect classification")
plt.savefig("MisclassifiedRatio_time_0_400ns.pdf")
plt.close("all")

bins_bary= np.arange(0,3.1415,0.31415)
_bins, _edges = np.histogram(bary_correct,bins_bary)
_bins2, _edges2 = np.histogram(bary_incorrect,bins_bary)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.plot(edges,bins,'o-',color="blue",lw=2,label="correct")
ax.plot(edges2,bins2,'o-',color="orange",lw=2,label="incorrect")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
ax.plot(edges,rat)
ax.set_ylim(0,1)
ax.set_xlabel("angular barycenter [rad]")
ax.set_ylabel("Incorrect classification")
plt.savefig("MisclassifiedRatio_bary.pdf")
plt.close("all")


bins_rms = np.arange(0.5,2.5,0.2)
_bins, _edges = np.histogram(rms_correct,bins_rms)
_bins2, _edges2 = np.histogram(rms_incorrect,bins_rms)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.plot(edges,bins,'o-',color="blue",lw=2,label="correct")
ax.plot(edges2,bins2,'o-',color="orange",lw=2,label="incorrect")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
ax.plot(edges,rat)
ax.set_ylim(0,1)
ax.set_xlabel("angular RMS [rad]")
ax.set_ylabel("Incorrect classification")
plt.savefig("MisclassifiedRatio_rms.pdf")
plt.close("all")

bins_dist = np.arange(0.,1.,0.01)
_bins, _edges = np.histogram(distVert_correct,bins_dist)
_bins2, _edges2 = np.histogram(distVert_incorrect,bins_dist)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.plot(edges,bins,'o-',color="blue",lw=2,label="correct")
ax.plot(edges2,bins2,'o-',color="orange",lw=2,label="incorrect")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
ax.plot(edges,rat)
ax.set_ylim(0,1)
ax.set_xlabel("distVert [relative]")
ax.set_ylabel("Incorrect classification")
plt.savefig("MisclassifiedRatio_distVert.pdf")
plt.close("all")

_bins, _edges = np.histogram(distHor_correct,bins_dist)
_bins2, _edges2 = np.histogram(distHor_incorrect,bins_dist)
bins=_bins
edges = _edges[:len(_edges)-1]
bins2 = _bins2
edges2 = _edges2[:len(_edges2)-1]
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# Plot as colored o's, connected
ax.plot(edges,bins,'o-',color="blue",lw=2,label="correct")
ax.plot(edges2,bins2,'o-',color="orange",lw=2,label="incorrect")
ax.set_ylabel('#')
leg = ax.legend()
ax = fig.add_subplot(2,1,2)
rat = getRatio(bins2,bins)
ax.plot(edges,rat)
ax.set_ylim(0,1)
ax.set_xlabel("distHor [relative]")
ax.set_ylabel("Incorrect classification")
plt.savefig("MisclassifiedRatio_distHor.pdf")
plt.close("all")

#---------------------------------------------
#-histograms without ratio incorrect/correct--
#---------------------------------------------

plt.hist(nhits_correct,bins_nhits,label='correct')
plt.hist(nhits_incorrect,bins_nhits,label='incorrect')
plt.xlabel('hit PMTs')
plt.ylabel('#')
plt.legend(loc='upper left')
#plt.show()
plt.savefig("Misclassified_hitPMTs.pdf")
plt.close("all")

bins_charge = np.arange(0,12500,625)
plt.hist(charge_correct,bins_charge, label='correct')
plt.hist(charge_incorrect,bins_charge, label='incorrect')
plt.xlabel('charge [p.e.]')
plt.ylabel('#')
plt.legend(loc='upper right')
#plt.show()
plt.savefig("Misclassified_charge.pdf")
plt.close("all")

bins_time = np.arange(0,1500,75)
plt.hist(time_correct,bins_time,label='correct')
plt.hist(time_incorrect,bins_time,label='incorrect')
plt.xlabel('average hit time [ns]')
plt.ylabel('#')
plt.legend(loc='upper right')
#plt.show()
plt.savefig("Misclassified_averageTime.pdf")
plt.close("all")

bins_bary = np.arange(0,3.1415,0.31415)
plt.hist(bary_correct,bins_bary,label='correct')
plt.hist(bary_incorrect,bins_bary,label='incorrect')
plt.xlabel('angular barycenter [rad]')
plt.ylabel('#')
plt.legend(loc='upper left')
#plt.show()
plt.savefig("Misclassified_Barycenter.pdf")
plt.close("all")

bins_rms = np.arange(0.5,2.5,0.2)
plt.hist(rms_correct,bins_rms,label='correct')
plt.hist(rms_incorrect,bins_rms,label='incorrect')
plt.xlabel('angular RMS [rad]')
plt.ylabel('#')
plt.legend(loc='upper right')
#plt.show()
plt.savefig("Misclassified_RMS.pdf")
plt.close("all")

bins_dist = np.arange(0.,1.,0.01)
plt.hist(distVert_correct,bins_dist,label='correct')
plt.hist(distVert_incorrect,bins_dist,label='incorrect')
plt.xlabel('vertical distance [relative]')
plt.ylabel('#')
plt.legend(loc='upper left')
#plt.show()
plt.savefig("Misclassified_distVert.pdf")
plt.close("all")


plt.hist(distHor_correct,bins_dist,label='correct')
plt.hist(distHor_incorrect,bins_dist,label='incorrect')
plt.xlabel('horizontal distance [relative]')
plt.ylabel('#')
plt.legend(loc='upper left')
#plt.show()
plt.savefig("Misclassified_distHor.pdf")
plt.close("all")


plt.hist(energy_correct,label='correct')
plt.hist(energy_incorrect,label='incorrect')
plt.xlabel('energy [MeV]')
plt.ylabel('#')
plt.legend(loc='upper left')
#plt.show()
plt.savefig("Misclassified_energy.pdf")
plt.close("all")

