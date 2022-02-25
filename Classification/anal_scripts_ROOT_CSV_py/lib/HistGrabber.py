import ROOT
import numpy as np

def GetHist(filepath,histname):
    '''
    Open up a root file that contains histograms, and returns the 
    histogram bin lefts and bin contents.
    '''
    ROOTFile = ROOT.TFile.Open(filepath)
    thehist =  ROOTFile.Get(histname)
    bin_lefts, evts =[], [] #pandas wants ntuples
    for i in xrange(int(thehist.GetNbinsX()+1)):
        #if i==0:
        #    continue
        bin_lefts.append(float(thehist.GetBinLowEdge(i)))
        evts.append(thehist.GetBinContent(i))
        print(str(bin_lefts[i]) + "," + str(evts[i]))
    bin_lefts = np.array(bin_lefts)
    evts = np.array(evts)
    return evts,bin_lefts

if __name__ == "__main__":
    MCHISTS = "../Data/MCProfiles/Analyzer_AmBe_Housing_Center-0-0-0_10k.root"
    DELTATHIST = "h_Pair_DeltaT"
    PEHIST = "h_HitsDelayed"
    GetHist(MCHISTS,DELTATHIST)
