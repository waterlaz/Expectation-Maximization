#include <iostream>
#include <fstream>
#include "Gaussian.hpp"
#include "GaussianGenerator.hpp"
#include "EM.hpp"


int main(){
    MixtureModel<IndependentGaussian<double, 2> > gmm(2);
    
    gmm[0] = IndependentGaussian<double, 2>( Eigen::Vector2d(2.0, 1.0), Eigen::Vector2d(7.0, 9.0) );
    gmm[1] = IndependentGaussian<double, 2>( Eigen::Vector2d(1.0, 2.0), Eigen::Vector2d(-5.0, -5.0) );
    gmm.prior[0] = 0.3;
    gmm.prior[1] = 0.7;
    
    Generator gg(gmm);

    int m = 1000;
    std::vector<Eigen::Vector2d> xs(m);
    for(auto&& x:xs){
        x = gg();
    }
    
    std::ofstream file;
    file.open("dump");
    for(auto&& x:xs){
        file<<x.transpose()<<"\n";
    }
    
    MixtureModel<IndependentGaussian<double, 2> > gmm2(2);
    
    EM em(gmm2);
    em.init(xs);
    for(int i=0; i<100; i++){
        em.iterate(xs);
    }
    
    std::cout<<gmm2.prior[0]<<"\n";
    std::cout<<"Deviation: \n"<<gmm2[0].deviation<<"\n\n";
    std::cout<<"Mean: \n"<<gmm2[0].mean<<"\n\n\n\n";
    
    std::cout<<gmm2.prior[1]<<"\n";
    std::cout<<"Deviation: \n"<<gmm2[1].deviation<<"\n\n";
    std::cout<<"Mean: \n"<<gmm2[1].mean<<"\n\n\n\n";
    
    /*

    std::vector<double> a(m, 1.0);
    
    IndependentGaussian<double, 2> ig;

    ig.likelihoodEstimate(a, xs);
    std::cout<<"mean\n"<<ig.mean<<"\n\n";
    std::cout<<"deviation\n"<<ig.deviation<<"\n\n";
    */
}



