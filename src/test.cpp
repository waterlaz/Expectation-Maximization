#include <iostream>
#include <fstream>
#include "GMM.hpp"
#include "GaussianGenerator.hpp"


int main(){
    /*
    IndependentGaussian g(5);
    for(int i=0; i<g.n; i++){
        g.mean[i] = i;
        g.deviation[i] = 1.0+i;
    }
    */
    //std::vector<Eigen::VectorXd>    
    std::cout<<"Hello, world\n";
    
    Eigen::VectorXd mean(2);
    mean<<1.0, 2.0;
    Eigen::MatrixXd covar(2, 2);
    int m = 1000;
    covar<<2.0, 0.0,
           0.0, 1;
    GaussianGenerator gg(covar, mean);
    std::vector<Eigen::VectorXd> xs(m);
    for(auto&& x:xs){
        x = gg();
    }
    std::vector<double> a(m, 1.0);
    
    IndependentGaussian<double, Eigen::Dynamic> ig(2);

    ig.likelihoodEstimate(a, xs);
    std::cout<<"mean\n"<<ig.mean<<"\n\n";
    std::cout<<"deviation\n"<<ig.deviation<<"\n\n";
}



