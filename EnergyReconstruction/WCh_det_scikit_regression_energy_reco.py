import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Times', size=20)
import pylab
pylab.rcParams['figure.figsize'] = 8, 4
from sklearn import linear_model, ensemble
import sys
sys.argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
from root_numpy import root2array, tree2array, fill_hist

import random
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

rfile = ROOT.TFile('nu_numu_1000_1039_CCQE_12in_energy_studies_for_training.root')
intree = rfile.Get('nu_eneNEW')

E_threshold = 0 
arr_hi_E = tree2array(intree,selection='trueKE>0')#+str(E_threshold))

#input variables
arr2_hi_E = arr_hi_E[['total_hits2','total_ring_PEs2','recoDWallR2','recoDWallZ2','lambda_max_2']]
arr2_hi_E_n = arr2_hi_E.view(arr2_hi_E.dtype[0]).reshape(arr2_hi_E.shape + (-1,))
#Mc muon energy
arr3_hi_E = arr_hi_E['trueKE']

chain = ROOT.TChain('nu_eneNEW')
for i in range(1040,1099):
     chain.Add('nu_numu_'+str(i)+'_CCQE_12in_energy_studies.root')
test_data_hi_E = tree2array(chain, selection='trueKE')
test_data_hi_Eneu = tree2array(chain, selection='neutrinoE')

recoDwall = tree2array(chain, selection='recoDWall_2')
recoToWall = tree2array(chain, selection='recoToWall_2')

test_data_reduced_hi_E = test_data_hi_E[['total_hits2','total_ring_PEs2','recoDWallR2','recoDWallZ2','lambda_max_2']]
test_data_reduced_hi_E_n = test_data_reduced_hi_E.view(test_data_reduced_hi_E.dtype[0]).reshape(test_data_reduced_hi_E.shape + (-1,))
test_data_trueKE_hi_E = test_data_hi_E['trueKE']
test_data_neutrinoE_hi_E = test_data_hi_Eneu['neutrinoE']

recoDwall_data = recoDwall['recoDWall_2']
recoToWall_data = recoToWall['recoToWall_2']

outputFile = ROOT.TFile.Open("WChscikit_reco.root", "RECREATE")

########## BDTG ###########
params = {'n_estimators': 600, 'max_depth': 10, 'random_state': 100, 
          'learning_rate': 0.01, 'loss': 'lad'} 
net_hi_E = ensemble.GradientBoostingRegressor(**params)
net_hi_E.fit(arr2_hi_E_n,arr3_hi_E)
net_hi_E

plt.figure(figsize=(10, 6))

test_data_recoKE_hi_E_BDTG = net_hi_E.predict(test_data_reduced_hi_E_n)

#write everything in tree:
outputTuple = ROOT.TNtuple("tuple", "tuple", "trueKE:neutrinoE:recoKE_BDTG:recoDwall:recoToWall")

for i in range(len(test_data_trueKE_hi_E)):
     outputTuple.Fill(test_data_trueKE_hi_E[i], test_data_neutrinoE_hi_E[i], test_data_recoKE_hi_E_BDTG[i], recoDwall_data[i], recoToWall_data[i])

outputTuple.Write()
outputFile.Close()
#######################

twod_GBR_abs_hi_E = np.dstack((test_data_trueKE_hi_E, 100.*(test_data_trueKE_hi_E-net_hi_E.predict(test_data_reduced_hi_E_n))/test_data_trueKE_hi_E))

hist_GBR_abs_hi_E = ROOT.TH2D('name_GBR_abs_hi_E', 'title', 50, E_threshold, 2000, 200, -100, 100)
fill_hist(hist_GBR_abs_hi_E, twod_GBR_abs_hi_E[0])
canvas = ROOT.TCanvas()
canvas.Divide(2,1)
canvas.cd(1)
canvas.cd(2)
hist_GBR_abs_hi_E.Draw()
hist_GBR_abs_hi_E.GetXaxis().SetTitle('true KE [MeV]')
hist_GBR_abs_hi_E.GetYaxis().SetTitle('abs(#Delta E)/E')
canvas.Draw()
canvas.SaveAs("newplots/BDTGscikitDE_E.png")

profile_GBR_abs_hi_E = hist_GBR_abs_hi_E.ProfileX()
profile_GBR_abs_hi_E.SetLineColor(ROOT.kBlue+2)
profile_GBR_abs_hi_E.SetMarkerColor(ROOT.kBlue+2)
profile_GBR_abs_hi_E.SetLineWidth(1)
canvas_prof = ROOT.TCanvas()
canvas_prof.Divide(2,1)
canvas_prof.cd(1)
canvas_prof.cd(2)
profile_GBR_abs_hi_E.Draw("ColZ")
profile_GBR_abs_hi_E.GetXaxis().SetTitle('true KE [MeV]')
profile_GBR_abs_hi_E.GetYaxis().SetTitle('abs(#Delta E)/E')
canvas_prof.Draw()
canvas_prof.SaveAs("newplots/BDTGscikitDE_E_profX.png")

hist_trueKE = ROOT.TH1D('trueKE', 'title', 100, 0, 2000)
hist_recoKE_hi_E = ROOT.TH1D('recoKE_GBR', 'title', 100, E_threshold, 2000)
hist_trueKE.SetLineColor(ROOT.kBlack)
hist_recoKE_hi_E.SetLineColor(ROOT.kBlue+2)
hist_trueKE.SetLineWidth(2)
hist_recoKE_hi_E.SetLineWidth(2)
fill_hist(hist_trueKE, test_data_trueKE_hi_E)
fill_hist(hist_recoKE_hi_E, net_hi_E.predict(test_data_reduced_hi_E_n))
c2 = ROOT.TCanvas()
hist_trueKE.Draw()
hist_recoKE_hi_E.Draw("same")
hist_trueKE.GetXaxis().SetTitle('true or reco KE [MeV]')
hist_trueKE.GetYaxis().SetTitle('Events')
c2.SetLogy()
c2.Draw()
c2.SaveAs("newplots/BDTG_MCEmu.png")


hist_trueKE_zoom = ROOT.TH1D('trueKE_zoom', 'title', 50, 0, 2000)
hist_recoKE_zoom = ROOT.TH1D('recoKE_zoom', 'title', 50, 0, 2000)
hist_recoKE_GBR_zoom = ROOT.TH1D('recoKE_GBR_zoom', 'title', 50, 0, 2000)
hist_trueKE_zoom.SetLineColor(ROOT.kBlack)
hist_recoKE_zoom.SetLineColor(ROOT.kRed)
hist_recoKE_GBR_zoom.SetLineColor(ROOT.kBlue+2)
hist_trueKE_zoom.SetLineWidth(2)
hist_recoKE_GBR_zoom.SetLineWidth(2)
hist_trueKE_zoom.Draw()
hist_recoKE_GBR_zoom.Draw("same")
hist_trueKE_zoom.GetXaxis().SetTitle('true or reco KE [MeV]')
hist_trueKE_zoom.GetYaxis().SetTitle('Events')
ROOT.gPad.SetLogy()
ROOT.gPad.Draw()












