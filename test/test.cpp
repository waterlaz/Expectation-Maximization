#include <iostream>
#include <list>
#include "Gaussian.hpp"
#include "GaussianGenerator.hpp"
#include "EM.hpp"

using namespace Eigen;

int main(){
    Matrix2d C;
    C<<2.0, 0.0,
       0.0, 0.0001;
    Gaussian<double, 2> gaussian( C.inverse(), Vector2d(0.0, 0.0) );
    std::cout<<"covar^-1\n"<<gaussian.invCovariance<<"\n";
    std::cout<<"factor\n"<<gaussian.invCovariance.constantFactor<<"\n";
    Generator g(gaussian);
    for(int i=0; i<10; i++){
        std::cout<<g()<<"\n\n";
    }
}



