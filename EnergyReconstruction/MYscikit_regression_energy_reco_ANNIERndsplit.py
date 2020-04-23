import numpy as np
import matplotlib.pyplot as plt
#plt.rc('font', family='serif', size=80)
#import pylab
#pylab.rcParams['figure.figsize'] = 16, 8
plt.rc('font', family='Times', size=16)
import pylab
pylab.rcParams['figure.figsize'] = 8, 6

from sklearn import linear_model, ensemble
import sys
sys.argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
from root_numpy import root2array, tree2array, fill_hist

import random
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

seed = 10000 #150
np.random.seed(seed)

#rfile = ROOT.TFile('vars_forEreco_NEW0_1_2QESB.root')#vars_forEreco_NEW0_1_QES.root')
#rfile = ROOT.TFile('vars_forEreco_NEW0_1_5QESC.root')
#rfile = ROOT.TFile('OUTreco25_0_5NEW.root')
#rfile = ROOT.TFile('OUTreco25_0_5NEW_PMTonly.root')
#rfile = ROOT.TFile('vars_forEreco26_NEW_0_8.root')
#intree = rfile.Get('nu_eneNEW')
rfile = ROOT.TFile('TreeforEnergyRecoB.root')
intree = rfile.Get('tuple')
#rfile = ROOT.TFile('vars_forEreco26_NEW_0_8MRD.root')
#intree = rfile.Get('nu_eneNEW')

#E_threshold = 1000 #100
E_threshold = 2.
E_low=0
E_high=2000
div=100
bins = int((E_high-E_low)/div)
print('bins: ', bins)
#arr_hi_E0 = tree2array(intree,selection='trueKE<'+str(E_threshold)) 
arr_hi_E0 = tree2array(intree,selection='neutrinoE<'+str(E_threshold))

#arr2_hi_E0 = arr_hi_E0[['lambda_max_2','TrueTrackLengthInMrd','diffDirAbs2','z1b','z2b','z3b','z4b','z5b','z6b','recoDWallR2','recoDWallZ2','totalLAPPDs','totalPMTs']]#,'raw_hits2']]
#arr2_hi_E0 = arr_hi_E0[['newrecolength','TrueTrackLengthInMrd','diffDirAbs2','z1b','z2b','z3b','z4b','z5b','z6b','recoDWallR2','recoDWallZ2','totalLAPPDs','totalPMTs']]
arr2_hi_E0 = arr_hi_E0[['newrecolength','TrueTrackLengthInMrd','diffDirAbs2','recoDWallR2','recoDWallZ2','totalLAPPDs','totalPMTs','vtxX','vtxY','vtxZ']]
#arr2_hi_E0 = arr_hi_E0[['newrecolength','diffDirAbs2','recoDWallR2','recoDWallZ2','totalLAPPDs','totalPMTs']]
#arr2_hi_E0 = arr_hi_E0[['newrecolength','TrueTrackLengthInMrd','diffDirAbs2']]
arr3_hi_E0 = 1000.*arr_hi_E0['trueKE']
#arr3_hi_E0 = 1000.*arr_hi_E0['neutrinoE']
newrecolength1=600.*arr_hi_E0['newrecolength']
TrueTrackLengthInMrd1=200.*arr_hi_E0['TrueTrackLengthInMrd']

rfile2 = ROOT.TFile('TreeforMCInfoB.root')
intree2 = rfile2.Get('tuple')
#rfile2 = ROOT.TFile('vars_forEreco26_NEW_0_8MRD.root')
#intree2 = rfile2.Get('nu_eneNEW')
#arr_hi = tree2array(intree2,selection='trueKE<'+str(E_threshold))
arr_hi = tree2array(intree2,selection='neutrinoE<'+str(E_threshold))
#print("vars: ", arr2_hi_E0)
#print("trueKE: ", arr3_hi_E0)
vtxX0=arr_hi['vtxX']
vtxY0=arr_hi['vtxY']
vtxZ0=arr_hi['vtxZ']
dirX0=arr_hi['dirX']
dirY0=arr_hi['dirY']
dirZ0=arr_hi['dirZ']
truedirX0=arr_hi['truedirX']
truedirY0=arr_hi['truedirY']
truedirZ0=arr_hi['truedirZ']
truevtxX0=arr_hi['truevtxX']
truevtxY0=arr_hi['truevtxY']
#truevtxZ0=arr_hi['truevtxZ']
trueMomentumTransfer0=arr_hi['TrueMomentumTransfer']
trueMuonAngle0=arr_hi['TrueMuonAngle']

rnd_indices = np.random.rand(len(arr_hi_E0)) < 0.50
arr_hi_E0B = arr_hi_E0[rnd_indices]
arr2_hi_E = arr2_hi_E0[rnd_indices]
arr2_hi_E_n = arr2_hi_E.view(arr2_hi_E.dtype[0]).reshape(arr2_hi_E.shape + (-1,))
#arr3_hi_E = 1000.*arr_hi_E0['trueKE']
#var_hi_E = arr_hi_E0[['TotlTrueTrackLentgh']]
arr3_hi_E = arr3_hi_E0[rnd_indices]

test_data_reduced_hi_E = arr2_hi_E0[~rnd_indices]
test_data_reduced_hi_E_n = test_data_reduced_hi_E.view(test_data_reduced_hi_E.dtype[0]).reshape(test_data_reduced_hi_E.shape + (-1,))
test_data_trueKE_hi_E = arr3_hi_E0[~rnd_indices]
test_data_hi_Eneu0 = 1000.*arr_hi_E0['neutrinoE']
test_data_neutrinoE_hi_E  = test_data_hi_Eneu0[~rnd_indices]
newrecolength2=newrecolength1[~rnd_indices]
TrueTrackLengthInMrd2=TrueTrackLengthInMrd1[~rnd_indices]
vtxX=vtxX0[~rnd_indices]
vtxY=vtxY0[~rnd_indices]
vtxZ=vtxZ0[~rnd_indices]
dirX=dirX0[~rnd_indices]
dirY=dirY0[~rnd_indices]
dirZ=dirZ0[~rnd_indices]
truedirX=truedirX0[~rnd_indices]
truedirY=truedirY0[~rnd_indices]
truedirZ=truedirZ0[~rnd_indices]
truevtxX=truevtxX0[~rnd_indices]
truevtxY=truevtxY0[~rnd_indices]
#truevtxZ=truevtxZ0[~rnd_indices]
trueMomentumTransfer=trueMomentumTransfer0[~rnd_indices]
trueMuonAngle=trueMuonAngle0[~rnd_indices]

#print('trueKE train:',arr3_hi_E0,' eval: ',test_data_trueKE_hi_E)
print('events for training: ',len(arr3_hi_E),' events for eval: ',len(test_data_trueKE_hi_E))
print('train: ',arr3_hi_E[0],',',arr3_hi_E[1],',',arr3_hi_E[2])
print('eval: ',test_data_trueKE_hi_E[0],',',test_data_trueKE_hi_E[1],',',test_data_trueKE_hi_E[2])

outputFile = ROOT.TFile.Open("NEWOUTRecoMuonMRD.root", "RECREATE")
#outputFile = ROOT.TFile.Open("PrevOUTRecoMuon.root", "RECREATE")
#outputFile = ROOT.TFile.Open("OUTreconeutrinoENEW2rndsplit.root", "RECREATE")
#outputFile = ROOT.TFile.Open("OUTreconeutrinoENEW2rndsplit_PMTsonly.root", "RECREATE")
#outputFile = ROOT.TFile.Open("OUTrecomuonENEW2rndsplit.root", "RECREATE")
#outputFile = ROOT.TFile.Open("OUTrecomuonENEW2rndsplit_PMTsonly.root", "RECREATE")
#print('test_data_reduced_hi_E_n ',test_data_reduced_hi_E_n)
#outputFile = ROOT.TFile.Open("OUTrecomuonE2.root", "RECREATE")

# In[16]: ########## BDTG ###########
#params = {'n_estimators': 1000, 'max_depth': 20,
n_estimators=1000
params = {'n_estimators':n_estimators, 'max_depth': 50,
          'learning_rate': 0.01, 'loss': 'lad'}#'lad'} 
#net_hi_E = ensemble.GradientBoostingRegressor(**params)
#net_hi_E.fit(arr2_hi_E_n,arr3_hi_E)
##net_hi_E.fit(arr2_hi_E0_n,arr3_hi_E0)
#net_hi_E

#adding train/test sample with random selection and check deviations:
arr_hi_E_Rn = shuffle(arr_hi_E0B, random_state=13)
#arr2_hi_E_Rn = arr_hi_E_Rn[['lambda_max_2','TrueTrackLengthInMrd','diffDirAbs2','z1b','z2b','z3b','z4b','z5b','z6b','recoDWallR2','recoDWallZ2','totalLAPPDs','totalPMTs']]#'raw_hits2']]
#arr2_hi_E_Rn = arr_hi_E_Rn[['newrecolength','TrueTrackLengthInMrd','diffDirAbs2','z1b','z2b','z3b','z4b','z5b','z6b','recoDWallR2','recoDWallZ2','totalLAPPDs','totalPMTs']]
arr2_hi_E_Rn = arr_hi_E_Rn[['newrecolength','TrueTrackLengthInMrd','diffDirAbs2','recoDWallR2','recoDWallZ2','totalLAPPDs','totalPMTs','vtxX','vtxY','vtxZ']]
#arr2_hi_E_Rn = arr_hi_E_Rn[['newrecolength','diffDirAbs2','recoDWallR2','recoDWallZ2','totalLAPPDs','totalPMTs']]
#arr2_hi_E_Rn = arr_hi_E_Rn[['newrecolength','TrueTrackLengthInMrd','diffDirAbs2']]
arr2_hi_E_Rn_n = arr2_hi_E_Rn.view(arr2_hi_E_Rn.dtype[0]).reshape(arr2_hi_E_Rn.shape + (-1,))
arr3_hi_E_Rn = 1000.*arr_hi_E_Rn['trueKE']
#arr3_hi_E_Rn = 1000.*arr_hi_E_Rn['neutrinoE']

offset = int(arr2_hi_E_Rn_n.shape[0] * 0.7) #Y.shape[0] returns the number of rows in Y 
arr2_hi_E_train, arr3_hi_E_train = arr2_hi_E_Rn_n[:offset], arr3_hi_E_Rn[:offset]  # train sample
arr2_hi_E_test, arr3_hi_E_test   = arr2_hi_E_Rn_n[offset:], arr3_hi_E_Rn[offset:]  # test sample

print("train shape: ",arr2_hi_E_train.shape," label: ",arr3_hi_E_train.shape)

net_hi_E = ensemble.GradientBoostingRegressor(**params)
net_hi_E.fit(arr2_hi_E_train, arr3_hi_E_train)
##net_hi_E.fit(arr2_hi_E_n,arr3_hi_E)
net_hi_E

#for i in range(0,10):
#print('arr2_hi_E_train: ',arr2_hi_E_train,' arr2_hi_E_test: ',arr2_hi_E_test)
#print('arr3_hi_E_test: ',arr3_hi_E_test)
#print('test_data_reduced_hi_E_n ',test_data_reduced_hi_E_n)

mse = mean_squared_error(arr3_hi_E_test, net_hi_E.predict(arr2_hi_E_test))
print("MSE: %.4f" % mse)
print("events at training & test samples: ", len(arr2_hi_E_Rn))
print("events at train sample: ", len(arr2_hi_E_train))
print("events at test sample: ", len(arr2_hi_E_test))

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(net_hi_E.staged_predict(arr2_hi_E_test)):
    test_score[i] = net_hi_E.loss_(arr3_hi_E_test, y_pred)

#plt.figure(figsize=(8, 5))
#plt.subplot(1, 2, 1)
#plt.title('Deviance')
fig,ax=plt.subplots(ncols=1, sharey=True)
ax.plot(np.arange(params['n_estimators']) + 1, net_hi_E.train_score_, 'b-',
         label='Training Set Deviance')
ax.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
ax.set_ylim(0.,500.)
ax.set_xlim(0.,n_estimators)
ax.legend(loc='upper right')
ax.set_ylabel('Least Absolute Deviations [MeV]')
ax.set_xlabel('Number of Estimators')
ax.yaxis.set_label_coords(-0.1, 0.6)
ax.xaxis.set_label_coords(0.85, -0.08)
plt.savefig("plotsMRD/deviation_train_test.png")

print("events for energy reco: ", len(test_data_reduced_hi_E_n))
test_data_recoKE_hi_E_BDTG = net_hi_E.predict(test_data_reduced_hi_E_n)

#print(' test_data_trueKE_hi_E: ', test_data_trueKE_hi_E, ' test_data_recoKE_hi_E_BDTG: ',test_data_recoKE_hi_E_BDTG)

#write everything in tree:
#outputTuple = ROOT.TNtuple("tuple", "tuple", "trueKE:recoKE:recoKE_BDTG:recoKE_BDTG2:recoKE_BDTG_par0:recoKE_BDTG_par1:recoKE_BDTG_par2:recoKE_BDTG_par3:recoKE_BDTG_par4:recoKE_BDTG_par5")
#outputTuple = ROOT.TNtuple("tuple", "tuple", "trueKE:neutrinoE:recoKE:recoKE_BDTG:recoKE_BDTG01:recoKE_BDTG_par02:recoDwall:recoToWall")
outputTuple = ROOT.TNtuple("tuple", "tuple", "trueKE:neutrinoE:recoKE:vtxX:vtxY:vtxZ:dirX:dirY:dirZ:TrueMomentumTransfer:TrueMuonAngle:truedirX:truedirY:truedirZ:DE_E")#truevtxX:truevtxY:truevtxZ")

#y = [0 for j in range (0,len(test_data_trueKE_hi_E))]
v=0
z=0
x=0
for i in range(0,len(test_data_trueKE_hi_E)):
    if test_data_trueKE_hi_E[i]<400:
       v=v+1
    if test_data_trueKE_hi_E[i]>=400 and test_data_trueKE_hi_E[i]<900:
       x=x+1
    if test_data_trueKE_hi_E[i]>=900:
       z=z+1
print('v: ', v, ' z: ', z)
y = [0 for j in range (0,v)]
y2 = [0 for j in range (0,z)]
y3=[0 for j in range (0,x)]

v0=0
z0=0
x0=0
Y=[0 for j in range (0,len(test_data_trueKE_hi_E))]
for i in range(len(test_data_trueKE_hi_E)):
      if TrueTrackLengthInMrd2[i]<1:
         print("TrueTrackLengthInMrd2: ", TrueTrackLengthInMrd2[i])
      Y[i] = 100.*(test_data_trueKE_hi_E[i]-test_data_recoKE_hi_E_BDTG[i])/(1.*test_data_trueKE_hi_E[i])
      outputTuple.Fill(test_data_trueKE_hi_E[i], test_data_neutrinoE_hi_E[i], test_data_recoKE_hi_E_BDTG[i],vtxX[i],vtxY[i],vtxZ[i],dirX[i],dirY[i],dirZ[i],trueMomentumTransfer[i],trueMuonAngle[i],truedirX[i],truedirY[i],truedirZ[i],Y[i])#truevtxX[i],truevtxY[i],truevtxZ[i])
      if test_data_trueKE_hi_E[i]<400.: 
         y[v0] = 100.*(test_data_trueKE_hi_E[i]-test_data_recoKE_hi_E_BDTG[i])/(1.*test_data_trueKE_hi_E[i])
         v0=v0+1
      if test_data_trueKE_hi_E[i]>=400 and test_data_trueKE_hi_E[i]<900:
         y3[x0] = 100.*(test_data_trueKE_hi_E[i]-test_data_recoKE_hi_E_BDTG[i])/(1.*test_data_trueKE_hi_E[i])
         x0=x0+1     
      if test_data_trueKE_hi_E[i]>=900.:
         y2[z0] = 100.*(test_data_trueKE_hi_E[i]-test_data_recoKE_hi_E_BDTG[i])/(1.*test_data_trueKE_hi_E[i])
         z0=z0+1
      #print("trueKE: ", test_data_trueKE_hi_E[i], " reco: ", test_data_recoKE_hi_E_BDTG[i])
#outputTuple.Write()
#outputFile.Close()
#######################
f, (ax2) = plt.subplots(1, 1)
ax2.scatter(test_data_trueKE_hi_E , test_data_recoKE_hi_E_BDTG)
ax2.set_xlabel("trueKE [MeV]")
ax2.set_ylabel("recoKE [MeV]")
xy_line = (0, 2000)
lims =[0,0,E_high,E_high]
ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
#ax2.plot(xy_line, 'b--')
plt.savefig("plotsMRD/Etrue_Erec.png")
ax2.set_ylim(0.,1500)
ax2.set_xlim(0.,1500)
plt.savefig("plotsMRD/Etrue_ErecZOOM.png")

canvas = ROOT.TCanvas()
canvas.cd(1)
th2f = ROOT.TH2F("Etrue_ErecZOOMCol", " ;E_{#mu} [MeV];E_{reco} [MeV]", 5*bins, E_low, E_high, 5*bins, E_low, E_high)
for i in range(len(test_data_trueKE_hi_E)):
    th2f.Fill(test_data_trueKE_hi_E[i], test_data_recoKE_hi_E_BDTG[i])
th2f.Draw("ColZ")
th2f.SetStats(0)
canvas.Draw()
canvas.SaveAs("plotsMRD/Etrue_ErecZOOMCol.png")

canvas.cd(1)
th2f = ROOT.TH2F("Enu_ErecZOOMCol", " ;E_{#nu} [MeV];E_{reco} [MeV]", 5*bins, E_low, E_high, 5*bins, E_low, E_high)
for i in range(len(test_data_trueKE_hi_E)):
    th2f.Fill(test_data_neutrinoE_hi_E[i], test_data_recoKE_hi_E_BDTG[i])
th2f.Draw("ColZ")
th2f.SetStats(0)
canvas.Draw()
canvas.SaveAs("plotsMRD/Enu_ErecZOOMCol.png")

canvas.cd(1)
th2f = ROOT.TH2F("Etrue_totLengthCol", " ;E_{#mu} [MeV];Total Track Length [cm]", 5*bins, E_low, E_high, 80, 0., 800.)
for i in range(len(test_data_trueKE_hi_E)):
    th2f.Fill(test_data_trueKE_hi_E[i], (newrecolength2[i]+TrueTrackLengthInMrd2[i]))
th2f.Draw("ColZ")
th2f.SetStats(0)
canvas.Draw()
canvas.SaveAs("plotsMRD/Etrue_totLengthCol.png")

canvas.cd(1)
th2f = ROOT.TH2F("Etrue_MRDLength", " ;E_{#mu} [MeV];Total Track Length [cm]", 5*bins, E_low, E_high, 40, 0., 200.)
for i in range(len(test_data_trueKE_hi_E)):
    th2f.Fill(test_data_trueKE_hi_E[i], (TrueTrackLengthInMrd2[i]))
th2f.Draw("ColZ")
th2f.SetStats(0)
canvas.Draw()
canvas.SaveAs("plotsMRD/Etrue_MRDLength.png")

nbins=np.arange(-100,100,2)
fig,ax0=plt.subplots(ncols=1, sharey=True)#, figsize=(8, 6))
cmap = sns.light_palette('b',as_cmap=True)
f=ax0.hist(Y, nbins, histtype='step', fill=True, color='gold',alpha=0.75)
ax0.set_xlim(-100.,100.)
ax0.set_xlabel('$\Delta E/E$ [%]')
ax0.set_ylabel('Number of Entries')
ax0.xaxis.set_label_coords(0.95, -0.08)
ax0.yaxis.set_label_coords(-0.1, 0.71)
title = "mean = %.2f, std = %.2f " % (np.array(Y).mean(), np.array(Y).std())
plt.title(title)
plt.savefig("plotsMRD/DE_E.png")

fig,ax=plt.subplots(ncols=1, sharey=True)#, figsize=(8, 6))
cmap = sns.light_palette('b',as_cmap=True)
f1=ax.hist(y, nbins, histtype='step', fill=False, edgecolor='blue', hatch="\\\\",alpha=0.75)
ax.set_xlim(-100.,100.)
ax.set_xlabel('$\Delta E/E$ [%]')
ax.set_ylabel('Number of Entries')
ax.xaxis.set_label_coords(0.95, -0.08)
ax.yaxis.set_label_coords(-0.1, 0.71)
#plt.savefig("plots/DE_E1.png")
f2=ax.hist(y2, nbins, histtype='step', fill=False, edgecolor='red', hatch="/",alpha=0.75)
f2=ax.hist(y3, nbins, histtype='step', fill=False, edgecolor='green', hatch="/",alpha=0.75)
b_line = mpatches.Patch(fill=True, edgecolor='blue', hatch="\\\\", label='E<400 MeV')
r_line = mpatches.Patch(fill=False, edgecolor='green', hatch="/", label='400 MeV<E<900 MeV')
bl_line = mpatches.Patch(fill=False, edgecolor='red', hatch="/", label='E>900 MeV')
plt.legend(handles=[b_line, r_line, bl_line])
plt.savefig("plotsMRD/DE_E2.png")

f, (ax1, ax2) = plt.subplots(1, 2)
ax2.scatter(test_data_trueKE_hi_E,(test_data_recoKE_hi_E_BDTG-test_data_trueKE_hi_E), c='r')
ax1.set_xlabel("trueKE [MeV]")
ax1.set_ylabel("recoKE - trueKE [MeV]")
ax2.set_xlabel("trueKE [MeV]")
ax2.set_ylabel("recoKE - trueKE [MeV]")
#matrix_hi_E = np.dstack(test_data_trueKE_hi_E , (test_data_recoKE_hi_E_BDTG-test_data_trueKE_hi_E) )

f, (ax1, ax2) = plt.subplots(1, 2)
ax2.scatter(test_data_trueKE_hi_E,(np.abs(test_data_recoKE_hi_E_BDTG-test_data_trueKE_hi_E)/test_data_trueKE_hi_E),c='r')
ax1.set_xlabel("trueKE [MeV]")
ax1.set_ylabel("DeltaE/E")
ax1.set_ylim(0,2)
ax2.set_xlabel("trueKE [MeV]")
ax2.set_ylabel("DeltaE/E")
ax2.set_ylim(0,2)
#twod_GBR_abs_hi_E = np.dstack((test_data_trueKE_hi_E, 100.*(test_data_recoKE_hi_E_BDTG-test_data_trueKE_hi_E)/test_data_trueKE_hi_E))
twod_GBR_abs_hi_E = np.dstack((test_data_trueKE_hi_E,Y))

hist_GBR_abs_hi_E = ROOT.TH2D('name_GBR_abs_hi_E', 'title', bins, E_low, E_high, 200, -100, 100)
fill_hist(hist_GBR_abs_hi_E, twod_GBR_abs_hi_E[0])
canvas = ROOT.TCanvas()
canvas.Divide(1,1)
canvas.cd(1)
hist_GBR_abs_hi_E.Draw("ColZ")
hist_GBR_abs_hi_E.GetXaxis().SetTitle('E_{MC,muon} [MeV]')
hist_GBR_abs_hi_E.GetYaxis().SetTitle('#Delta E/E [%]')
hist_GBR_abs_hi_E.Write()
canvas.Draw()
canvas.SaveAs("plotsMRD/BDTGscikitDE_E.png")

x = [0 for j in range (0,200)]
y = [0 for j in range (0,200)]
list = [0 for j in range (0,200)]
def median1(h1):
    #compute the median for 1-d histogram h1 
    nbins2 = h1.GetXaxis().GetNbins()
    #print("---nbins: ",nbins)
    for i in range(0,nbins2):
        #print("i, ", i, "nbins: ",nbins)
        x[i] = h1.GetBinCenter(i)
        y[i] = h1.GetBinContent(i) 
        #print("i: ",i," x[i]: ",x[i]," y[i]: ", y[i])       
    median= ROOT.TMath.Median(nbins2,x)
    ## In C++:
    #Double_t *x1 = new Double_t[200]; 
    #Double_t *y2 = new Double_t[200];
    #for (int i = 0; i < 200; i++) { x1[i] = a7->GetBinCenter(i); y2[i] = a7->GetBinContent(i); } 
    #double MedianOfHisto = TMath::Median(200, x1, y2);
    #cout<<MedianOfHisto <<endl;
    print(" Median= ", median)
        #list[i]=median
    return median

def median2(h2):
    #compute and print the median for each slice along X of h
    nbins = h2.GetXaxis().GetNbins()
    for i in range(0,nbins):
        h1=h2.ProjectionY("",i,i+1)
        median = median1(h1)
        mean = h1.GetMean()
        print("Median of Slice: ",i," Median= ", median," Mean= ",mean)
#       delete h1
#median2(hist_GBR_abs_hi_E)

#TProfile *EnerResoBDTG_pfx= EnerResoBDTG->ProfileX("EnerResoBDTG_pfx", 0., 3000., "s");
#profile_GBR_abs_hi_E = hist_GBR_abs_hi_E.ProfileX("EnerResoBDTG_pfx", 1, -1, "s")
profile_GBR_abs_hi_E = hist_GBR_abs_hi_E.ProfileX()
profile_GBR_abs_hi_E.SetLineColor(ROOT.kBlue+2)
profile_GBR_abs_hi_E.SetMarkerColor(ROOT.kBlue+2)
profile_GBR_abs_hi_E.SetLineWidth(1)
canvas_prof = ROOT.TCanvas()
canvas_prof.cd(1)
style = ROOT.TStyle()
style.cd()
style.SetOptStat(0)
#style.SetLabelSize(1.2)
profile_GBR_abs_hi_E.SetTitle("")
profile_GBR_abs_hi_E.Draw("ColZ")
profile_GBR_abs_hi_E.GetYaxis().SetRangeUser(-100.,100.)
profile_GBR_abs_hi_E.GetXaxis().SetTitle('E_{MC,muon} [MeV]')
profile_GBR_abs_hi_E.GetYaxis().SetTitle('#Delta E/E [%]')
canvas_prof.Draw()
canvas_prof.SaveAs("plotsMRD/BDTGscikitDE_E_profX.png")

hist_trueKE_hi_E = ROOT.TH1D('trueKE', 'title', 5*bins, E_low, E_high)
hist_recoKE_hi_E = ROOT.TH1D('recoKE', 'title', 5*bins, E_low, E_high)
hist_trueNeu_hi_E= ROOT.TH1D('trueNeu', 'title', 5*bins, E_low, E_high)
hist_trueKE_hi_E.SetLineColor(ROOT.kBlack)
hist_recoKE_hi_E.SetLineColor(ROOT.kBlue+2)
hist_trueNeu_hi_E.SetLineColor(ROOT.kRed+2)
hist_trueKE_hi_E.SetLineWidth(2)
hist_recoKE_hi_E.SetLineWidth(2)
hist_trueNeu_hi_E.SetLineWidth(2)
fill_hist(hist_trueKE_hi_E, test_data_trueKE_hi_E)
fill_hist(hist_recoKE_hi_E, test_data_recoKE_hi_E_BDTG)
fill_hist(hist_trueNeu_hi_E,test_data_neutrinoE_hi_E)
c2 = ROOT.TCanvas()
hist_recoKE_hi_E.Draw()
hist_trueKE_hi_E.Draw("same")
#hist_trueNeu_hi_E.Draw("same")
hist_trueKE_hi_E.GetXaxis().SetTitle('true or reco KE [MeV]')
hist_trueKE_hi_E.GetYaxis().SetTitle('Events')
c2.Draw()
c2.SaveAs("plotsMRD/BDTG_MCEmu.png")
c2.SetLogy()
c2.Draw()
c2.SaveAs("plotsMRD/BDTG_logMCEmu.png")


#net.feature_importances_

outputTuple.Write()
outputFile.Close()


