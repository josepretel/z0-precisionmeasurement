#include "TMatrixD.h"
#include <iostream>
using namespace std;

void useMatrix(){
  //Filling a matrix

  TMatrixD matrix(4,4);
  matrix(0,0)=0.934;
  matrix(0,1)=0.02;
  matrix(0,2)=0.01;
  matrix(0,3)=0;
  matrix(1,0)=0.01;
  matrix(1,1)=0.946;
  matrix(1,2)=0.01;
  matrix(1,3)=0.01;
  matrix(2,0)=0.01;
  matrix(2,1)=0.01;
  matrix(2,2)=0.965;
  matrix(2,3)=0.01;
  matrix(3,0)=0.01;
  matrix(3,1)=0.01;
  matrix(3,2)=0.01;
  matrix(3,3)=0.999;
  cout<<"Matrix" << endl;
  matrix.Print();

  //Inverting the matrix 
  TMatrixD Inverse(4,4);
  Inverse = matrix.Invert();
  cout<<"Inverse"<< endl;
  Inverse.Print();

  double mean[4][4];
  mean[0][0]=0.934;
  mean[0][1]=0.02;
  mean[0][2]=0.01;
  mean[0][3]=0;
  mean[1][0]=0.01;
  mean[1][1]=0.946;
  mean[1][2]=0.01;
  mean[1][3]=0.01;
  mean[2][0]=0.01;
  mean[2][1]=0.01;
  mean[2][2]=0.965;
  mean[2][3]=0.01;
  mean[3][0]=0.01;
  mean[3][1]=0.01;
  mean[3][2]=0.01;
  mean[3][3]=0.999;

  double err[4][4];
  err[0][0]=0.001;
  err[0][1]=0.0001;
  err[0][2]=0.0001;
  err[0][3]=0.0001;
  err[1][0]=0.0001;
  err[1][1]=0.001;
  err[1][2]=0.0001;
  err[1][3]=0.0001;
  err[2][0]=0.0001;
  err[2][1]=0.0001;
  err[2][2]=0.001;
  err[2][3]=0.0001;
  err[3][0]=0.0001;
  err[3][1]=0.0001;
  err[3][2]=0.0001;
  err[3][3]=0.001;

  TRandom3 *r = new TRandom3();
  int ntoy = 1000;

  TH1F* histo[4][4];
  histo[0][0] =  new TH1F("h00","h00",200,1,1.20);
  histo[0][1] =  new TH1F("h01","h01",200,-0.03,0.02);
  histo[0][2] =  new TH1F("h02","h02",200,-0.02,0.02);
  histo[0][3] =  new TH1F("h03","h03",200,-0.02,0.02);
  histo[1][0] =  new TH1F("h10","h10",200,-0.02,0.02);
  histo[1][1] =  new TH1F("h11","h11",200,1,1.20);
  histo[1][2] =  new TH1F("h12","h12",200,-0.02,0.02);
  histo[1][3] =  new TH1F("h13","h13",200,-0.02,0.02);
  histo[2][0] =  new TH1F("h20","h20",200,-0.02,0.02);
  histo[2][1] =  new TH1F("h21","h21",200,-0.02,0.02);
  histo[2][2] =  new TH1F("h22","h22",200,1,1.2);
  histo[2][3] =  new TH1F("h23","h23",200,-0.02,0.02);
  histo[3][0] =  new TH1F("h30","h30",200,-0.02,0.02);
  histo[3][1] =  new TH1F("h31","h31",200,-0.02,0.02);
  histo[3][2] =  new TH1F("h32","h32",200,-0.02,0.02);
  histo[3][3] =  new TH1F("h33","h33",200,0.85,1.05);
  
  for(int i=0;i<ntoy;i++){
    TMatrixD matrix_toy(4,4);
    for(int j=0;j<4;j++){
      for(int k=0;k<4;k++){
	matrix_toy(j,k)=r->Gaus(mean[j][k],err[j][k]);
      }
    }

    TMatrixD Inverse_toy(4,4);
    Inverse_toy = matrix_toy.Invert();

    for(int j=0;j<4;j++){
      for(int k=0;k<4;k++){
	histo[j][k]->Fill(Inverse_toy(j,k));
      }
    }
  }



  double inverse_err[4][4];

  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      TString name = "c";
      name+=i;
      name+=j;
      TCanvas *c = new TCanvas(name,name,800,700);
      c->cd();
      histo[i][j]->Draw();
      histo[i][j]->Fit("gaus");
      TF1 *fit = histo[i][j]->GetFunction("gaus");
      inverse_err[i][j] = fit->GetParameter(2);
    }
  }
  
  cout<<"errors"<<endl;
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      cout<<"["<<i<<"]["<<j<<"]  "<<inverse_err[i][j]<<endl;      cout<<"errors"<<endl;
    }
  }
  
}
