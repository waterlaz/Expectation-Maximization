#include <iostream>
#include <list>
#include "Gaussian.hpp"
#include "GaussianGenerator.hpp"
#include "EM.hpp"

using namespace Eigen;

int main(){
    Matrix2d C;
    C<<2.0, 0.0,
       0.0, 0.01;
    Gaussian<double, 2> gaussian( C.inverse(), Vector2d(0.0, 0.0) );
    KnownDistribution kd(gaussian);
    std::cout<<"covar^-1\n"<<gaussian.invCovariance<<"\n";
    std::cout<<"factor\n"<<gaussian.invCovariance.constantFactor<<"\n";
    Generator g(gaussian);

    for(int i=0; i<10; i++){
        Vector2d x = g();
        std::cout<<"p = "<<kd(x)<<"\n";
        std::cout<<x<<"\n\n";
    }

    double s = 0.0;
    double d = 0.1;
    for(double x = -100.0; x <= 100.0; x+=d){
        for(double y = -100.0; y <= 100.0; y+=d){
            s += kd(Vector2d(x, y))*d*d;
        }
    }
    std::cout<<"1 ~= "<<s<<"\n";
}



