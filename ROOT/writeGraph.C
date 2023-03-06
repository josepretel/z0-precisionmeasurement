#include "TFile.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include <iostream>
using namespace std;

void writeGraph(){
  TFile* out=new TFile("pro.root","RECREATE");
  TCanvas *C1 = new TCanvas("Canvas","Canvas",600,400);
  TGraphErrors* gr= new TGraphErrors(3);
  float x[3]={1.,2.,3.};
  float y[3]={0.9,2.,3.1};
  float yerror[3]={0.1,0.1,0.1};
  for( int ipoint=0; ipoint<3; ipoint++){
    gr->SetPoint(ipoint, x[ipoint], y[ipoint]);
    gr->SetPointError(ipoint,0,yerror[ipoint]);
  }

  gr->Draw("A*");
  gr->SetMarkerStyle(20);
  out->cd();
  gr->SetName("Graph");
  gr->Write();
  C1->Print("Graph.ps");
  C1->Close();
}
