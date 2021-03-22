#include "TFile.h"
#include "TROOT.h"
#include "TGraph.h"
#include "TF1.h"
#include "TStyle.h"
#include "TGraphErrors.h"
#include <iostream>
using namespace std;

void fit(){
  gROOT->Reset();
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0000);
  TFile* histos = new TFile("pro.root");
  TGraphErrors *gr = (TGraphErrors*)histos->Get("Graph");

  gr->SetLineWidth(2);
  gr->SetLineColor(1);

  TF1* grfit = new TF1("grfit","[0]+[1]*x",0.9,3.1);
  grfit->SetParameters(0.5,1.6);
  gr->Draw("A*");
  gr->SetMarkerStyle(20);
  gr->Fit("grfit","","",0.9,3.1);
  cout << "Param0=" << grfit->GetParameter(0)<<endl;
  cout << "Param1=" << grfit->GetParameter(1)<<endl;

  gStyle->SetOptFit(111);


}
