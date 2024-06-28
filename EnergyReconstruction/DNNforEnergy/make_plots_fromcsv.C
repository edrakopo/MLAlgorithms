#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"
#include "TF1.h"
#include "TMath.h"
#include "TLine.h"

void make_plots_fromcsv()
{
  
  TTree *t = new TTree("t", "my tree"); 
  t->ReadFile("Etrue_Ereco_till800MeV.csv", "i:TrueEnergy:DNNEnergy");
  //t->Draw("TrueEnergy : DNNEnergy");

  Float_t TrueEnergy, DNNEnergy; 
  t->SetBranchAddress("TrueEnergy", &TrueEnergy);
  t->SetBranchAddress("DNNEnergy", &DNNEnergy);

  //TH2D *Etrue_Ereco =new TH2D("Etrue_Ereco "," ;MC Muon Energy; Predicted Energy", 100, 0., 2000., 100, 0.,2000.);
  TH2D *Etrue_Ereco =new TH2D("Etrue_Ereco_till800MeV"," ;MC Muon Energy; Predicted Energy", 100, 0., 1000., 100, 0.,1000.);

  for (Long64_t ievt=0; ievt<t->GetEntries();ievt++) {
      t->GetEntry(ievt);
      if (ievt%1000 == 0) {
          std::cout << "--- ... Processing event: "<<ievt<<std::endl;
      }
      if(TrueEnergy>800.){ std::cout << "WHAT?? "<<"i= "<<ievt<<", "<<TrueEnergy<<std::endl;}

      Etrue_Ereco->Fill(TrueEnergy,DNNEnergy); 
 
  }
  new TCanvas();
  TLine * line = new TLine(0.,0.,2000.,2000.);
  line->SetLineColor(2);
  Etrue_Ereco->Draw("ColZ"); 
  line->Draw("same");

  t->Clear();  
}
