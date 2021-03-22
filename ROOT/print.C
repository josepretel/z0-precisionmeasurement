#include "TFile.h"
#include "TROOT.h"
#include "TH1F.h"
#include "TString.h"
#include "TLegend.h"
#include <iostream>
using namespace std;

void print(){
  gROOT->Reset();
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0);//if you want to get rid of the statistics box
 


  TFile* histos=new TFile("ourana.root");
  TH1F *Ncharged[5];


  int colors[5]={1,2,3,4,5};//for ee, mm, tt, qq, data
  TString numbers[5] = {"0","1","2","3","4"};
  for(unsigned int i=0; i<4; i++){
    Ncharged[i] = (TH1F*)histos->Get(TString("Ncharged")+numbers[i]);
    Ncharged[i]->SetTitle("Ncharged");
    Ncharged[i]->GetXaxis()->SetTitle("Ncharged");
    Ncharged[i]->GetYaxis()->SetTitle("a.u.");
    Ncharged[i]->SetLineColor(colors[i]);
    Ncharged[i]->Scale(1./Ncharged[i]->GetEntries());
    
    if(i==0)Ncharged[i]->Draw("");
    else Ncharged[i]->Draw("same");
  }

  TString titles[5]={"ee","mm","tt","qq","data"}
  TLegend *leg = new TLegend(0.7,0.7,0.9,0.9);
  leg->SetHeader("Legend"); 
  leg->SetFillColor(0);
  for(unsigned int i=0; i<4; i++) leg->AddEntry(Ncharged[i],titles[i],"l");
  leg->Draw("same");
 
  //histos->Close();
}

