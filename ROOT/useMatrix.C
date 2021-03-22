#include "TMatrixD.h"
#include <iostream>
using namespace std;

void useMatrix(){
  //Filling a matrix

  TMatrixD matrix(2,2);
  matrix(0,0)=0;
  matrix(0,1)=1;
  matrix(1,0)=1;
  matrix(1,1)=2;
  cout<<"Matrix" << endl;
  matrix.Print();

  //Inverting the matrix 
  TMatrixD Inverse(2,2);
  Inverse = matrix.Invert();
  cout<<"Inverse"<< endl;
  Inverse.Print();

  //Having a vector
  TMatrixD vector(2,1);
  vector(0,0)=1.;
  vector(1,0)=2.;
  
  //Multiply the both

  TMatrixD result(2,1);
  result=Inverse * vector;
  cout<<"Multiplied result:"<< endl;
  result.Print();
}
