#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TString.h"
#include "TTree.h"
#include "TBranch.h"
#include <iostream>
using namespace std;

void writeHisto(){

  cout << "Starting analysis" << endl;

  TFile* out=new TFile("ourana.root","RECREATE");

  TH1F *hNcharged[5];
  TH2F *h_Ncharged_vs_Pcharged[5];

  for(unsigned int i=0; i<5; i=i+1){
    TString title="Ncharged"; title+=i;
    hNcharged[i]=new TH1F(title,title,40,0,40);
    TString title2="h_Ncharged_vs_Pcharged"; title2+=i;
    h_Ncharged_vs_Pcharged[i]=new TH2F(title2,title2,50,0,50,100,0,400);
  }
  
  TFile* files[5];
  files[0] = new TFile("../daten/mc/ee.root");
  files[1] = new TFile("../daten/mc/mm.root");
  files[2] = new TFile("../daten/mc/tt.root");
  files[3] = new TFile("../daten/mc/qq.root");
  files[4] = new TFile("../daten/daten/daten_4.root");
  
  Float_t         run;
  Float_t         event;
  Float_t         Ncharged;
  Float_t         Pcharged;
  Float_t         E_ecal;
  Float_t         E_hcal;
  Float_t         E_lep;
  Float_t         cos_thru;
  Float_t         cos_thet;
  for(unsigned ifile=0; ifile<4;ifile++){
    files[ifile]->cd();
    TTree *h33 = (TTree*)files[ifile]->Get("h3");
    TBranch *b_event = h33->GetBranch("event");
    b_event->SetAddress(&event);
    TBranch *b_run  = h33->GetBranch("run");
    b_run->SetAddress(&run);
    TBranch *b_Ncharged  = h33->GetBranch("Ncharged");
    b_Ncharged->SetAddress(&Ncharged);
    TBranch *b_Pcharged  = h33->GetBranch("Pcharged");
    b_Pcharged->SetAddress(&Pcharged);
    TBranch *b_E_ecal  = h33->GetBranch("E_ecal");
    b_E_ecal->SetAddress(&E_ecal);
    TBranch *b_E_hcal  = h33->GetBranch("E_hcal");
    b_E_hcal->SetAddress(&E_hcal);
    TBranch *b_E_lep  = h33->GetBranch("E_lep");
    b_E_lep->SetAddress(&E_lep);
    TBranch *b_cos_thru  = h33->GetBranch("cos_thru");
    b_cos_thru->SetAddress(&cos_thru);
    TBranch *b_cos_thet  = h33->GetBranch("cos_thet");
    b_cos_thet->SetAddress(&cos_thet);
    int nevents = h33->GetEntries();

    //nevents=20;
    for(int ievent=0; ievent<nevents; ievent=ievent+1){
      if(ievent % 1000==0) cout << "Event " << ievent << endl;
      h33->GetEvent(ievent);
      
      //Fill the histograms here
	
      hNcharged[ifile]->Fill(Ncharged);
      h_Ncharged_vs_Pcharged[ifile]->Fill(Ncharged, Pcharged);
    }
  }

 
  for(unsigned int i=0; i<5; i++)files[i]->Close();
  out->Write();
  out->Close();
}

