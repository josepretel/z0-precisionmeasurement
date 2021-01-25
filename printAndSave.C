#include "TFile.h"
#include "TROOT.h"
#include "TH1F.h"
#include "TString.h"
#include "TLegend.h"
#include <iostream>
using namespace std;

void printAndSave(){
  gROOT->Reset();
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0);//if you want to get rid of the statistics box
 
  TFile* histos=new TFile("ourana.root");
  TH1F *Ncharged[5];

  TCanvas * NChargedCanvas = new TCanvas("Ncharged","Ncharged",1200,800); // you can set the size here

  int colors[5]={1,2,3,4,5};//for ee, mm, tt, qq, data
  for(unsigned int i=0; i<4; i++){
    Ncharged[i] = (TH1F*)histos->Get(TString("Ncharged")+i);
    Ncharged[i]->SetTitle("Number of Charged Tracks");
    Ncharged[i]->GetXaxis()->SetTitle("Number of Charged Tracks n_{charged}^{tracks}, not #pi");//just an example of LaTeX style
    Ncharged[i]->GetYaxis()->SetTitle("a.u.");// arbitrary units
    Ncharged[i]->SetLineColor(colors[i]);
    Ncharged[i]->Scale(1./Ncharged[i]->GetEntries());
    
    if(i==0)Ncharged[i]->Draw("");
    else Ncharged[i]->Draw("same");// Draw without overwriting the existing histogramms
  }

  TString titles[5]={"ee","mm","tt","qq","data"}
  TLegend *leg = new TLegend(0.65,0.65,0.85,0.85);
  //  leg->SetHeader("Legend"); 
  leg->SetFillColor(0);
  //  leg->SetLineColor(0);
  for(unsigned int i=0; i<4; i++) leg->AddEntry(Ncharged[i],titles[i],"l");
  leg->Draw();
  // Print to file:
  NChargedCanvas->Print("NChargedCanvas.root","root");
  NChargedCanvas->Print("NChargedCanvas.pdf","pdf Portrait");
  NChargedCanvas->Print("NChargedCanvas.png","png");
  //clean up ;-)
  delete leg;
  delete NChargedCanvas;
  delete histos;
}

