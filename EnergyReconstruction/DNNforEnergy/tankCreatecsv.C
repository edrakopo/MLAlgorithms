#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"
#include "TF1.h"
#include "TMath.h"
#include <cmath>
#include <iostream>
#include <algorithm>

//---------------------------------------------------------------------------------------------------------//
// Find Distance between Muon Vertex and Photon Production Point (lambda)
//---------------------------------------------------------------------------------------------------------//
double find_lambda(double xmu_rec,double ymu_rec,double zmu_rec,double xrecDir,double yrecDir,double zrecDir,double x_pmtpos,double y_pmtpos,double z_pmtpos,double theta_cher){
     double lambda1 = 0.0;    double lambda2 = 0.0;    double length = 0.0 ;
     double xmupos_t1 = 0.0;  double ymupos_t1 = 0.0;  double zmupos_t1 = 0.0;
     double xmupos_t2 = 0.0;  double ymupos_t2 = 0.0;  double zmupos_t2 = 0.0;
     double xmupos_t = 0.0;   double ymupos_t = 0.0;   double zmupos_t = 0.0;

     double theta_muDir_track = 0.0;
     double theta_muDir_track1 = 0.0;  double theta_muDir_track2 = 0.0;
     double cos_thetacher = cos(theta_cher*TMath::DegToRad());
     double xmupos_tmin = 0.0; double ymupos_tmin = 0.0; double zmupos_tmin = 0.0;
     double xmupos_tmax = 0.0; double ymupos_tmax = 0.0; double zmupos_tmax = 0.0;
     double lambda_min = 10000000;  double lambda_max = -99999999.9;  double lambda = 0.0;

     double alpha = (xrecDir*xrecDir + yrecDir*yrecDir + zrecDir*zrecDir) * ( (xrecDir*xrecDir + yrecDir*yrecDir + zrecDir*zrecDir) - (cos_thetacher*cos_thetacher) );
     double beta = ( (-2)*(xrecDir*(x_pmtpos - xmu_rec) + yrecDir*(y_pmtpos - ymu_rec) + zrecDir*(z_pmtpos - zmu_rec) )*((xrecDir*xrecDir + yrecDir*yrecDir + zrecDir*zrecDir) - (cos_thetacher*cos_thetacher)) );
     double gamma = ( ( (xrecDir*(x_pmtpos - xmu_rec) + yrecDir*(y_pmtpos - ymu_rec) + zrecDir*(z_pmtpos - zmu_rec))*(xrecDir*(x_pmtpos - xmu_rec) + yrecDir*(y_pmtpos - ymu_rec) + zrecDir*(z_pmtpos - zmu_rec)) ) - (((x_pmtpos - xmu_rec)*(x_pmtpos - xmu_rec) + (y_pmtpos - ymu_rec)*(y_pmtpos - ymu_rec) + (z_pmtpos - zmu_rec)*(z_pmtpos - zmu_rec))*(cos_thetacher*cos_thetacher)) );


     double discriminant = ( (beta*beta) - (4*alpha*gamma) );

     lambda1 = ( (-beta + sqrt(discriminant))/(2*alpha));
     lambda2 = ( (-beta - sqrt(discriminant))/(2*alpha));

     xmupos_t1 = xmu_rec + xrecDir*lambda1;  xmupos_t2 = xmu_rec + xrecDir*lambda2;
     ymupos_t1 = ymu_rec + yrecDir*lambda1;  ymupos_t2 = ymu_rec + yrecDir*lambda2;
     zmupos_t1 = zmu_rec + zrecDir*lambda1;  zmupos_t2 = zmu_rec + zrecDir*lambda2;

     double tr1 = sqrt((x_pmtpos - xmupos_t1)*(x_pmtpos - xmupos_t1) + (y_pmtpos - ymupos_t1)*(y_pmtpos - ymupos_t1) + (z_pmtpos - zmupos_t1)*(z_pmtpos - zmupos_t1));
     double tr2 = sqrt((x_pmtpos - xmupos_t2)*(x_pmtpos - xmupos_t2) + (y_pmtpos - ymupos_t2)*(y_pmtpos - ymupos_t2) + (z_pmtpos - zmupos_t2)*(z_pmtpos - zmupos_t2));
     theta_muDir_track1 = (acos( (xrecDir*(x_pmtpos - xmupos_t1) + yrecDir*(y_pmtpos - ymupos_t1) + zrecDir*(z_pmtpos - zmupos_t1))/(tr1) )*TMath::RadToDeg());
     theta_muDir_track2 = (acos( (xrecDir*(x_pmtpos - xmupos_t2) + yrecDir*(y_pmtpos - ymupos_t2) + zrecDir*(z_pmtpos - zmupos_t2))/(tr2) )*TMath::RadToDeg());
     //---------------------------- choose lambda!! ---------------------------------
     if( theta_muDir_track1 < theta_muDir_track2 ){
       lambda = lambda1;
       xmupos_t = xmupos_t1;
       ymupos_t = ymupos_t1;
       zmupos_t = zmupos_t1;
       theta_muDir_track=theta_muDir_track1;
     }else if( theta_muDir_track2 < theta_muDir_track1 ){
       lambda = lambda2;
       xmupos_t = xmupos_t2;
       ymupos_t = ymupos_t2;
       zmupos_t = zmupos_t2;
       theta_muDir_track=theta_muDir_track2;
     }

     return lambda;
}

//---------------------------------------------------------------------------------------------------------//
// Split the water tank in small boxes (20cm) and search for vertex in each through PMT (tank)
//---------------------------------------------------------------------------------------------------------//


double find_gridPoint(double truevtxx_tank,double truevtxy_tank,double truevtxz_tank, int step){ //step= to split tank

double Gridpoint = -10;

  
  //gridpoint is the box no. that we find each vtx
 //int  gridpoint[15][20][15], z, r, c; //first elemennt of 3d array represents "how many 2D arrays" we want
//int  gridpoint[30][40][30], z, r, c; //10cm
int  gridpoint[10][10][10], z, r, c; //volume inside 100x100cm
 
 // i want to make truevtxx_tank is an array. truevtxx_tank[i]

 //cout<<"truevtx1 "<<truevtxx_tank<<endl;
 //cout<<"truevty1 "<<truevtxy_tank<<endl;
 //cout<<"truevtz1 "<<truevtxz_tank<<endl; 
 
 
//this is for 20cm
/*
//make grid
int i = 1;
for(int z=0; z<15 ; z++){ 
 	//cout<<"z "<<z<<endl;
 	


	for(int r=0; r<20 ; r++){ //we need 20 rows
 	//cout<<"r "<<r<<endl;
     
 		for(int c=0; c<15 ; c++){ //we need 15 columns
 		//cout<<"c "<<c<<endl;
 		
 			gridpoint[z][r][c] = i;
 			i++;
 			
 		}
 	}
 }	
 */
 
 /*
 
 //make grid for 10cm
int i = 1;
for(int z=0; z<30 ; z++){ 
 	//cout<<"z "<<z<<endl;
 	


	for(int r=0; r<40 ; r++){ //we need 40 rows
 	//cout<<"r "<<r<<endl;
     
 		for(int c=0; c<30 ; c++){ //we need 30 columns
 		//cout<<"c "<<c<<endl;
 		
 			gridpoint[z][r][c] = i;
 			i++;
 			
 		}
 	}
 }	
 
 //find grid number for root file values
 for(int l=0; l<291; l=l+step) { //l is z
	//cout<<"l1 "<<l<<endl;
	if((-150+l)<truevtxz_tank && truevtxz_tank<=(-140+l)) { 
		//cout<<"l2 "<<l<<endl;
    	
 		for(int j=0; j<391; j=j+step) { //j represents rows. It grows every 10 cm from -200 to 200.
	      		//cout<<"j1 "<<j<<endl;
  			if((-200+j)<truevtxy_tank && truevtxy_tank<=(-190+j)) {
  				//cout<<"j2 "<<j<<endl;
  			
  				for(int k=0; k<291; k=k+step){
  					//cout<<"k1 "<<k<<endl;
  					if((-150+k)<truevtxx_tank  && truevtxx_tank<=(-140+k)){ //k represents columns. It grows evry 10cm from -150 to 150.
    				 		//cout<<"k2 "<<k<<endl;
    				 		
    				 		//cout<<"--- "<<(1.*l/step)<<","<<(1.*j/step)<<","<<(1.*k/step)<<endl;
                                                Gridpoint = gridpoint[int(1.*l/step)][int(1.*j/step)][int(1.*k/step)];
    				 		//cout<<"Gridpoint= "<<Gridpoint<<endl;
    				 	

 						break;
    		 		  		 		 }
    		 		 	//else cout<<"nopec"<<endl;
    		 		}
    		 	break;}
    		 	//else cout<<"noper"<<endl;    		 	
    		 }
break;}
//else{ cout<<"nopez" <<endl;}
}

*/

//Make grid for 100x100 cm 

 //make grid for 10cm
int i = 1;
for(int z=0; z<10 ; z++){ 
 	//cout<<"z "<<z<<endl;
 	
	for(int r=0; r<10 ; r++){ //we need 40 rows
 	//cout<<"r "<<r<<endl;
     
 		for(int c=0; c<10 ; c++){ //we need 30 columns
 		//cout<<"c "<<c<<endl;
 		
 			gridpoint[z][r][c] = i;
 			i++;
 			
 		}
 	}
 }
 
 //find grid number for root file values inside the volume of tank in a 100x100 cm
 for(int l=0; l<191; l=l+step) { //l is z
	//cout<<"l1 "<<l<<endl;
	if((-50+l)<truevtxz_tank && truevtxz_tank<=(-40+l)) { 
		//cout<<"l2 "<<l<<endl;
    	
 		for(int j=0; j<191; j=j+step) { //j represents rows. It grows every 10 cm from -200 to 200.
	      		//cout<<"j1 "<<j<<endl;
  			if((-50+j)<truevtxy_tank && truevtxy_tank<=(-40+j)) {
  				//cout<<"j2 "<<j<<endl;
  			
  				for(int k=0; k<191; k=k+step){
  					//cout<<"k1 "<<k<<endl;
  					if((-50+k)<truevtxx_tank  && truevtxx_tank<=(-40+k)){ //k represents columns. It grows evry 10cm from -150 to 150.
    				 		//cout<<"k2 "<<k<<endl;
    				 		
    				 		//cout<<"--- "<<(1.*l/step)<<","<<(1.*j/step)<<","<<(1.*k/step)<<endl;
                                                Gridpoint = gridpoint[int(1.*l/step)][int(1.*j/step)][int(1.*k/step)];
    				 		//cout<<"Gridpoint= "<<Gridpoint<<endl;
    				 	

 						break;
    		 		  		 		 }
    		 		 	//else cout<<"nopec"<<endl;
    		 		}
    		 	break;}
    		 	//else cout<<"noper"<<endl;    		 	
    		 }
break;}
//else{ cout<<"nopez" <<endl;}
}

	
    return Gridpoint;
}

void tankCreatecsv()
{

   int ievt=0; float trueNeuE=0; float trueE=0;
   float TrueTrackLengthInWater2=0; float TrueTrackLengthInMrd2=0; float recoDWallR2=0; float recoDWallZ2=0;
   float diffDirAbs2=0; 
   float dirX2=0; float dirY2=0; float dirZ2=0;
   float vtxX2=0; float vtxY2=0; float vtxZ2=0;
   double C_VAC=29.9792458; //speed of light in vacuum [cm/ns]
   
   ofstream csvfile;
    //csvfile.open ("tankPMT_forEnergy.csv");
    csvfile.open ("tankPMT_forEnergy_till800MeV.csv");
   
   //ofstream csvfile2;
   //csvfile2.open ("tank_MRD_insidevolume.csv");
   int maxhits0=1100; 
   //int maxhits0=300; 
      //--- write to file: ---//
      //if(first==1 && deny_access==0){
      //    deny_access=1;
      
      
          for (int i=0; i<maxhits0;++i){
             stringstream strs;
             strs << i;
             string temp_str = strs.str();
             string X0_name= "l_";
             X0_name.append(temp_str);
             const char * x0name = X0_name.c_str();
             csvfile<<x0name<<",";
          }
          
          //x
         for (int xx=0; xx<maxhits0;++xx){ 
             stringstream strs4;
             strs4 << xx;
             string temp_str4 = strs4.str();
             string X_name= "X_";
             X_name.append(temp_str4);
             const char * xxname = X_name.c_str();
             csvfile<<xxname<<",";
             //csvfile2<<xxname<<",";
          }
          //y
          for (int yy=0; yy<maxhits0;++yy){ 
             stringstream strs4;
             strs4 << yy;
             string temp_str4 = strs4.str();
             string Y_name= "Y_";
             Y_name.append(temp_str4);
             const char * yname = Y_name.c_str();
             csvfile<<yname<<",";
             //csvfile2<<yname<<",";
          }
          //z
          for (int zz=0; zz<maxhits0;++zz){
             stringstream strs4;
             strs4 << zz;
             string temp_str4 = strs4.str();
             string Z_name= "Z_";
             Z_name.append(temp_str4);
             const char * zname = Z_name.c_str();
             csvfile<<zname<<",";
             //csvfile2<<zname<<",";
          }

          
          for (int ii=0; ii<maxhits0;++ii){
             stringstream strs4;
             strs4 << ii;
             string temp_str4 = strs4.str();
             string T_name= "T_";
             T_name.append(temp_str4);
             const char * tname = T_name.c_str();
             csvfile<<tname<<",";
             //csvfile2<<tname<<",";
          }
          
          /*
          for (int ii=0; ii<maxhits0;++ii){
             stringstream strs4;
             strs4 << ii;
             string temp_str4 = strs4.str();
             string Q_name= "Q_";
             Q_name.append(temp_str4);
             const char * qname = Q_name.c_str();
             csvfile<<qname<<",";
          }
	*/
      
        
        //csvfile<<"lambda_max"<<","; //first estimation of track length(using photons projection on track)
        //csvfile<<"totalPMTs"<<",";
        //csvfile<<"totalLAPPDs"<<",";
        
        csvfile<<"lambda_max"<<",";
        csvfile<<"TrueTrackLengthInWater"<<",";
        /*csvfile<<"neutrinoE"<<",";   
        csvfile<<"trueKE"<<",";*/
        csvfile<<"diffDirAbs"<<",";
        //csvfile<<"TrueTrackLengthInMrd"<<",";
        csvfile<<"recoDWallR"<<",";
        csvfile<<"recoDWallZ"<<",";
        /*
        csvfile<<"dirX"<<",";
        csvfile<<"dirY"<<",";
        csvfile<<"dirZ"<<",";
        csvfile<<"recovtxX"<<",";
        csvfile<<"recovtxY"<<",";
        csvfile<<"recovtxZ"<<",";
        */
       
        /*
        csvfile<<"truedirX"<<",";
        csvfile<<"truedirY"<<",";
        csvfile<<"truedirZ"<<",";
        
        csvfile<<"truevtxX"<<",";
        csvfile<<"truevtxY"<<",";
        csvfile<<"truevtxZ"<<",";
        csvfile<<"Gridpoint"<<","; */
        csvfile<<"nhits"<<",";
        csvfile<<"recoVtxFOM"<<",";
        csvfile<<"TrueTrackLengthInMrd"<<",";
        csvfile<<"trueKE";
        //csvfile<<"GridpointReco";
        /*
        for (int ii=0; ii<maxhits0;++ii){
             stringstream strs4;
             strs4 << ii;
             string temp_str4 = strs4.str();
             string pmt_name= "gridPMT_";
             pmt_name.append(temp_str4);
             const char * pmtname = pmt_name.c_str();
             csvfile<<pmtname<<",";
          }
          
          */
        //csvfile<<"GridpointPMT"<<",";
        //csvfile<<"recoVtxFOM";
        //csvfile<<"recoStatus"<<",";  //comment
        //csvfile<<"deltaVtxR"<<",";   //comment
        //csvfile<<"deltaAngle";       //comment
        csvfile<<'\n';
       /*
        csvfile2<<"recovtxX"<<",";
        csvfile2<<"recovtxY"<<",";
        csvfile2<<"recovtxZ"<<",";
        csvfile2<<"truevtxX"<<",";
        csvfile2<<"truevtxY"<<",";
        csvfile2<<"truevtxZ"<<",";
        csvfile2<<"Gridpoint"<<",";
        csvfile2<<"GridpointReco"<<",";
        
	csvfile2<<'\n';
*/
   char fname[1000]; int count1=0, count2=0, count3=0;
   //sprintf(fname,"PhaseIIReco_SuccessfulReco_04202019.root");//,i);
   //sprintf(fname,"PMTLAPPDReco_743Runs_05202019.root");//,i);
   //sprintf(fname,"data/PMTLAPPDReco_All_06082019.root");
   //sprintf(fname,"/home/liliadrak/ANNIE/ntuples_Ereco/vtxreco-beamlikegridall-cut.root");
   //sprintf(fname,"/home/evi/Desktop/ANNIE-THESIS/beamlikemu/Data/vtxreco-beamlikemrd.root");
   sprintf(fname, "/Users/edrakopo/Desktop/working@NCSRDemokritos/ANNIE/SupervisedTheses/EviSakkou_VertexRecoDNN/Evi_scripts/vtxreco-beamlikemrd.root");
   TFile *input=new TFile(fname,"READONLY");
   cout<<"input file: "<<fname<<endl;
   //TFile f("recovtxfom.root","new");
   TTree *regTree= (TTree*) input->Get("phaseIITriggerTree");
   //----------- read the tree from file:
   //deny_access=1;
   
  
   Int_t run, event, nhits, trigger, recoStatus;
   double recoVtxFOM, deltaVtxR, deltaAngle;
   double truevtxX,truevtxY,truevtxZ,truedirX,truedirY,truedirZ,vtxX,vtxY,vtxZ,dirX,dirY,dirZ, TrueTrackLengthInMrd,TrueTrackLengthInWater,TrueNeutrinoEnergy,trueEnergy,TrueMomentumTransfer,TrueMuonAngle;
   std::string *TrueInteractionType = 0;
   std::vector<double> *digitX=0; std::vector<double> *digitY=0;  std::vector<double> *digitZ=0;
   std::vector<double> *digitT=0; std::vector<int>  *digitType=0; std::vector<double> *digitQ=0;
   //std::vector<double> *seedX=0; std::vector<double> *seedY=0; std::vector<double> *seedZ=0;
   float trueMuonEnergy=0.;
   int Pi0Count,PiPlusCount,PiMinusCount, Gridpoint, GridpointPMT, GridpointReco;

   regTree->SetBranchAddress("runNumber", &run);
   regTree->SetBranchAddress("eventNumber", &event);
   regTree->SetBranchAddress("trueMuonEnergy", &trueEnergy);
   //regTree->SetBranchAddress("TrueNeutrinoEnergy", &TrueNeutrinoEnergy);
   //regTree->SetBranchAddress("trigger", &trigger);
   regTree->SetBranchAddress("nhits", &nhits);
   regTree->SetBranchAddress("recoVtxX", &vtxX);
   regTree->SetBranchAddress("recoVtxY", &vtxY);
   regTree->SetBranchAddress("recoVtxZ", &vtxZ);
   regTree->SetBranchAddress("recoDirX", &dirX);
   regTree->SetBranchAddress("recoDirY", &dirY);
   regTree->SetBranchAddress("recoDirZ", &dirZ);
   regTree->SetBranchAddress("trueVtxX", &truevtxX);
   regTree->SetBranchAddress("trueVtxY", &truevtxY);
   regTree->SetBranchAddress("trueVtxZ", &truevtxZ);
   //regTree->SetBranchAddress("seedVtxX", &seedX);
   //regTree->SetBranchAddress("seedVtxY", &seedY);
   //regTree->SetBranchAddress("seedVtxZ", &seedZ);
   regTree->SetBranchAddress("trueDirX", &truedirX);
   regTree->SetBranchAddress("trueDirY", &truedirY);
   regTree->SetBranchAddress("trueDirZ", &truedirZ);
   regTree->SetBranchAddress("hitQ", &digitQ);
   regTree->SetBranchAddress("hitT", &digitT);
   regTree->SetBranchAddress("hitX", &digitX);
   regTree->SetBranchAddress("hitY", &digitY);
   regTree->SetBranchAddress("hitZ", &digitZ);
   regTree->SetBranchAddress("hitType", &digitType);
   regTree->SetBranchAddress("recoStatus", &recoStatus);
   regTree->SetBranchAddress("recoVtxFOM", &recoVtxFOM);
   //regTree->SetBranchAddress("TrueInteractionType", &TrueInteractionType);
   regTree->SetBranchAddress("trueTrackLengthInMRD", &TrueTrackLengthInMrd);
   regTree->SetBranchAddress("trueTrackLengthInWater", &TrueTrackLengthInWater);
   //regTree->SetBranchAddress("TrueMomentumTransfer", &TrueMomentumTransfer);
   //regTree->SetBranchAddress("TrueMuonAngle", &TrueMuonAngle);
   regTree->SetBranchAddress("deltaVtxR", &deltaVtxR);
   regTree->SetBranchAddress("deltaAngle", &deltaAngle);
   regTree->SetBranchAddress("Pi0Count", &Pi0Count);
   regTree->SetBranchAddress("PiPlusCount", &PiPlusCount);
   regTree->SetBranchAddress("PiMinusCount", &PiMinusCount);
   
   //TH1F h1("recoVtxFOM","DR_recoVtxFOM",300,50,163000);
  
   cout<<"regTree->GetEntries(): "<<regTree->GetEntries()<<endl;
  // double DR;
//for (Long64_t ievt=0; ievt<10; ievt++) {
 for(Long64_t ievt=0; ievt<regTree->GetEntries(); ievt++) {
   regTree->GetEntry(ievt);
 	//if(recoVtxFOM<=0){

   //double** coordinates = find_Coords(1);
   //cout<<"find_Coords(1): "<<find_Coords(1)<<endl;
  // DR=TMath::Sqrt((vtxX-truevtxX)*(vtxX-truevtxX)+(vtxY-truevtxY)*(vtxY-truevtxY)+(vtxZ-truevtxZ)*(vtxZ-truevtxZ));
 //  h1.Fill(DR); 
  // }
   double lambda_min = 10000000;  double lambda_max = -99999999.9; double lambda = 0;
   int totalPMTs=0; int totalLAPPDs=0; recoDWallR2=0; recoDWallZ2=0; diffDirAbs2=0;
   double lambda_vec[1100]={0.}; double digitt[1100]={0.}; double digitq[1100]={0.}; double digitx[1100]={0.}; double digity[1100]={0.}; double digitz[1100]={0.}; double gridpmt[1100]={0.}; 
   double dr[1100]={0.};
    
   if(ievt%100==0){
     cout << "ievt:" << ievt <<endl; //printing the events number to check if all the events are being run
    }
   //if(recoStatus == 0){ count1++;
   
   //if(recoVtxFOM>0){ count1++;
	//   cout<<"count1: "<<count1<<endl;
     //if(nhits>10 && trueEnergy<1500.){count3++;
      if(nhits>10 && trueEnergy<800.){count3++;
     //if((*TrueInteractionType == "QES - Weak[CC]") && TrueTrackLengthInMrd>0.){
      //cout<<"TrueTrackLengthInMrd: "<<TrueTrackLengthInMrd<<endl;
      
      //if(TrueTrackLengthInMrd>0){count2++;// && Pi0Count==0 && PiPlusCount==0 && PiMinusCount==0){
      
      	if(-50<truevtxX && truevtxX<=50){ 
      		if(-50<truevtxY && truevtxY<=50){ 
      			if(-50<truevtxZ && truevtxZ<=50){ 
				/*cout<<"ievt: "<<ievt<<" trueEnergy: "<<trueEnergy<<" with nhits: "<<nhits<<endl;
				//Gridpoint=find_gridPoint(truevtxX,truevtxY, truevtxZ, 20);
				//Gridpoint=find_gridPoint(truevtxX,truevtxY, truevtxZ, 10);
				//GridpointReco =find_gridPoint(vtxX,vtxY,vtxZ, 10);
				
				//cout<< "gridPoint:"<< Gridpoint<<endl;
				//cout<< "gridPointRECO:"<< GridpointReco<<endl;
				cout<< "\n";
				//cout<< "tank:"<< find_griPoint((truevtxX,truevtxY, truevtxZ, 20)<<endl;
				cout<< "truevtxX:"<< truevtxX<<endl;
				cout<< "truevtxY:"<< truevtxY<<endl;
				cout<< "truevtxZ:"<< truevtxZ<<endl;
				cout<< "\n";*/
				//cout<< "recovtxX:"<< vtxX<<endl;
				//cout<< "recovtxY:"<< vtxY<<endl;
				//cout<< "recovtxZ:"<< vtxZ<<endl;
				//cout<< "\n";
				
				
				//calculate diff dir with (0,0,1)
				//double diffDirAbs0 = TMath::ACos(dirZ)*TMath::RadToDeg();
                                double diffDirAbs0 = TMath::ACos(truedirZ)*TMath::RadToDeg();
				//cout<<"diffDirAbs0: "<<diffDirAbs0<<endl;
				diffDirAbs2=diffDirAbs0/90.;
				//double recoVtxR2 = vtxX*vtxX + vtxZ*vtxZ;//vtxY*vtxY;
				//double recoDWallR = 152.4-TMath::Sqrt(recoVtxR2);
				//double recoDWallZ = 198-TMath::Abs(vtxY);
                                //calculate wall distances with true veretx info: 
    				double recoVtxR2 = truevtxX*truevtxX + truevtxZ*truevtxZ;//vtxY*vtxY;				
 				double recoDWallR = 152.4-TMath::Sqrt(recoVtxR2);
				double recoDWallZ = 198-TMath::Abs(truevtxY);

				for(int k=0; k<nhits; k++){
				  //std::cout<<"k: "<<k<<", "<<digitT->at(k)<<" | "<<digitType->at(k)<<std::endl;
				  digitt[k]=digitT->at(k);
				  digitx[k]=digitX->at(k);
				  digity[k]=digitY->at(k);
				  digitz[k]=digitZ->at(k);
				  digitq[k]=digitQ->at(k);
			       
			      
				  GridpointPMT=find_gridPoint(digitx[k], digity[k], digitz[k], 10);
				  //cout<<"gridpointPMT "<<GridpointPMT<<endl;
				  
				  gridpmt[k]=GridpointPMT;
				  //cout<<"gridpmt2 "<<gridpmt[k]<<endl;
				  
				  //if( (digitType->at(k)) == "PMT8inch"){ totalPMTs++; }
				  //if( (digitType->at(k)) == "lappd_v0"){ totalLAPPDs++; }
				  //if( (digitType->at(k)) == 0){ totalPMTs++; }
				  //if( (digitType->at(k)) == 1){ totalLAPPDs++; }

				   //------ Find rack Length as the distance between the reconstructed vertex last photon emission point ----/
				   //lambda = find_lambda(vtxX,vtxY,vtxZ,dirX,dirY,dirZ,digitX->at(k),digitY->at(k),digitZ->at(k),42.);
                                  //calculate lambda with true vertex info:
                                  lambda = find_lambda(truevtxX, truevtxY, truevtxZ, truedirX, truedirY, truedirZ, digitX->at(k),digitY->at(k),digitZ->at(k),42.);
				  if( lambda <= lambda_min ){
				      lambda_min = lambda;
				  }
				  if( lambda >= lambda_max ){
				      lambda_max = lambda;
				  }
				  
				  lambda_vec[k]=lambda;
				 //m_data->Stores["ANNIEEvent"]->Set("WaterRecoTrackLength",lambda_max);
				}
				//std::cout<<"the track length in the water tank (1st approx) is: "<<lambda_max<<std::endl;
				//cout<<"TrueTrackLengthInWater: "<<TrueTrackLengthInWater<<" TrueTrackLengthInMrd: "<<TrueTrackLengthInMrd<<endl;

			       //---------------------------------
			       //---- add values to tree for energy reconstruction - the values added to the BDT need to be normalised:
			       ievt=ievt; //currententry;
			       trueNeuE=1.*TrueNeutrinoEnergy;
			       trueE=1.*trueEnergy;
			       dirX2=dirX; dirY2=dirY; dirZ2=dirZ;
			       vtxX2=vtxX; vtxY2=vtxY; vtxZ2=vtxZ;
			       recoDWallR2      = recoDWallR/152.4;
			       recoDWallZ2      = recoDWallZ/198.;
			       TrueTrackLengthInMrd2 = TrueTrackLengthInMrd/200.;
			       //--- 
			       TrueTrackLengthInWater2 = TrueTrackLengthInWater*100.;//converting from m to cm //TrueTrackLengthInWater/500.;

				//----- write to .csv file - including variables for track length & energy reconstruction:
				for(int i=0; i<maxhits0;++i){
				   csvfile<<lambda_vec[i]<<",";
				}
			      

				for(int i=0; i<maxhits0;++i){
				   csvfile<<digitx[i]<<",";
				   //csvfile2<<digitx[i]<<",";
				}
				for(int i=0; i<maxhits0;++i){
				   csvfile<<digity[i]<<",";
				   //csvfile2<<digity[i]<<",";
				}
				for(int i=0; i<maxhits0;++i){
				   csvfile<<digitz[i]<<",";
				   //csvfile2<<digitz[i]<<",";
				}
				for(int i=0; i<maxhits0;++i){
				   csvfile<<digitt[i]<<",";
				   //csvfile2<<digitt[i]<<",";
				}
				/*
				for(int i=0; i<maxhits0;++i){
				   csvfile<<digitq[i]<<",";
				   
				}
				*/		
				//csvfile<<lambda_max<<",";
				//csvfile<<totalPMTs<<",";
				//csvfile<<totalLAPPDs<<",";
				
				csvfile<<lambda_max<<",";
				csvfile<<TrueTrackLengthInWater2<<",";
				//csvfile<<trueNeuE<<",";
				//csvfile<<trueE<<",";
				csvfile<<diffDirAbs2<<",";
				//csvfile<<TrueTrackLengthInMrd2<<",";
				csvfile<<recoDWallR2<<",";
				csvfile<<recoDWallZ2<<",";
                                /*
				csvfile<<dirX2<<",";
				csvfile<<dirY2<<",";
				csvfile<<dirZ2<<",";
					
				csvfile<<vtxX2<<",";
				csvfile<<vtxY2<<",";
				csvfile<<vtxZ2<<",";
				
				csvfile<<truedirX<<",";
				csvfile<<truedirY<<",";
				csvfile<<truedirZ<<",";
				
				csvfile<<truevtxX<<",";
				csvfile<<truevtxY<<",";
				csvfile<<truevtxZ<<",";
				*/
				//csvfile<<Gridpoint<<",";
		                csvfile<<nhits<<",";
                		csvfile<<recoVtxFOM<<",";
		                csvfile<<TrueTrackLengthInMrd<<",";
		                csvfile<<trueE;
				//csvfile<<GridpointReco;
				/*
				for(int i=0; i<maxhits0;++i){
				   csvfile<<gridpmt[i]<<",";
				   //cout<<"gridpmt3 "<<gridpmt[i]<<endl;
				}
				*/
				
				//csvfile<<GridpointPMT;
				//csvfile<<recoVtxFOM;
				//csvfile<<recoStatus<<",";
				//csvfile<<deltaVtxR<<",";
				//csvfile<<deltaAngle;
				csvfile<<'\n';
                                /*
				csvfile2<<vtxX2<<",";
				csvfile2<<vtxY2<<",";
				csvfile2<<vtxZ2<<",";
				csvfile2<<truevtxX<<",";
				csvfile2<<truevtxY<<",";
				csvfile2<<truevtxZ<<",";
				csvfile2<<Gridpoint<<",";
                csvfile2<<GridpointReco<<",";
				csvfile2<<'\n';
				*/
	}
        //------------------------
	//cout<<vtxX2<<truevtxX<<endl;
   	//cout<< "gridPointttttttt:"<< Gridpoint<<endl;
        }
      }
     // fom}
    //}
   }
   }
    //h1.Write();
    //f.Close();
   input->Close();
cout <<"Num of events with nhits>10="<<count3<<endl;
//cout <<"Num of events with TrueTrackLengtninMRD>0="<<count2<<endl;
//cout <<"Num of events with recovtxfom<=0:"<<count1<<endl;
}







