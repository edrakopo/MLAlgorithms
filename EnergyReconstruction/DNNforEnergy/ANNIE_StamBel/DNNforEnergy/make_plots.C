#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"
#include "TF1.h"
#include "TMath.h"   
#if not defined(__CINT__) || defined(__MAKECINT__)
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#endif

double median1(TH1D* h1){
  double x[200]={0}; double y[200]={0}; double median=0;
 //compute the median for 1-d histogram h1
 int nbins2 = h1->GetXaxis()->GetNbins();
 //cout<<"nbins2: "<<nbins2<<endl;
 for(int i=0; i<nbins2; i++){
    x[i] = h1->GetBinCenter(i+1);
    y[i] = h1->GetBinContent(i+1);
    //cout<<"x: "<<x[i]<<" y: "<<y[i]<<endl;
  }
  median = TMath::Median(nbins2, x, y);
  return median;
}
/*  
void median2(TH2D* h2){
   //compute and print the median for each slice along X of h
   int nbins = h2->GetXaxis()->GetNbins();
   for(int i=0; i<nbins; i++){
      h1=h2->ProjectionY("",i,i+1);
      median = median1(h1);
      mean = h1->GetMean();
      cout<<"i: "<<i<<" median: "<<median<<" mean: "<<mean<<endl;
      delete h1;
    }
}
*/
  
void  make_plots()
{
  TFile* file = new TFile("output_length.root"); 

 //new TCanvas();
  TTree *regTree = (TTree*)file->Get("tuple");
  Float_t test_y,y_predicted,recoVtxFOM,deltaVtxR,deltaAngle;
  regTree->SetBranchAddress("test_y", &test_y);
  regTree->SetBranchAddress("y_predicted", &y_predicted);
  regTree->SetBranchAddress("recoVtxFOM", &recoVtxFOM);
  regTree->SetBranchAddress("deltaVtxR", &deltaVtxR);
  regTree->SetBranchAddress("deltaAngle", &deltaAngle);
 
  TH2D *DL_var1 =new TH2D("DL_var1"," ;DL;recoVtxFOM", 200, 0., 2000., 100, 0.,100.);
  TH2D *DL_var2 =new TH2D("DL_var2"," ;DL;deltaVtxR", 200, 0., 2000., 300, 0.,300.);
  TH2D *DL_var3 =new TH2D("DL_var3"," ;DL;deltaAngle", 200, 0., 2000., 100, 0.,100.);

  for (Long64_t ievt=0; ievt<regTree->GetEntries();ievt++) {
     if (ievt%1000 == 0) {
        std::cout << "--- ... Processing event: "<<ievt<<std::endl;
     }
     regTree->GetEntry(ievt);

    DL_var1->Fill(y_predicted-test_y, recoVtxFOM);
    DL_var2->Fill(y_predicted-test_y, deltaVtxR);
    DL_var3->Fill(y_predicted-test_y, deltaAngle);
    
  }//end of entries
  //new TCanvas();
  new TCanvas();
  DL_var1->Draw("ColZ");

  new TCanvas();
  DL_var2->Draw("ColZ");  

  new TCanvas();
  DL_var3->Draw("ColZ");

}
