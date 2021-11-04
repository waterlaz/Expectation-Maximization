#include <iostream>
#include <fstream>
#include <list>
#include "Gaussian.hpp"
#include "GaussianGenerator.hpp"
#include "EM.hpp"

using namespace Eigen;

int main(){
    MixtureModel<IndependentGaussian<double, 2> > gmm(2);
    
    gmm[0] = IndependentGaussian<double, 2>( Eigen::Vector2d(2.0, 1.0), Eigen::Vector2d(7.0, 9.0) );
    gmm[1] = IndependentGaussian<double, 2>( Eigen::Vector2d(1.0, 2.0), Eigen::Vector2d(-5.0, -5.0) );
    gmm.prior[0] = 0.3;
    gmm.prior[1] = 0.7;
    
    Generator gg(gmm);

    int m = 1000;
    //std::vector<Eigen::Vector2d> xs(m);
    std::list<Eigen::Vector2d> xs(m);
    for(auto&& x:xs){
        x = gg();
    }
    
    std::ofstream file;
    file.open("dump");
    for(auto&& x:xs){
        file<<x.transpose()<<"\n";
    }
    
    MixtureModel<IndependentGaussian<double, 2> > gmm1(2);
    EM em1(gmm1);
    em1.init(xs);
    
    for(int i=0; i<100; i++){
        em1.iterate(xs);
    }
    
    std::cout<<"Independent Gaussian:\n";
    std::cout<<gmm1.prior[0]<<"\n";
    std::cout<<"Deviation: \n"<<gmm1[0].deviation<<"\n\n";
    std::cout<<"Mean: \n"<<gmm1[0].mean<<"\n\n\n\n";
    
    std::cout<<gmm1.prior[1]<<"\n";
    std::cout<<"Deviation: \n"<<gmm1[1].deviation<<"\n\n";
    std::cout<<"Mean: \n"<<gmm1[1].mean<<"\n\n\n\n";
    

    MixtureModel<Gaussian<double, 2> > gmm2(2);
    EM em2(gmm2);
    em2.init(xs);
    
    for(int i=0; i<100; i++){
        em2.iterate(xs);
    }
    
    std::cout<<"General Gaussian:\n";
    std::cout<<gmm2.prior[0]<<"\n";
    std::cout<<"Caovariance: \n"<<gmm2[0].getCovariance()<<"\n\n";
    std::cout<<"Mean: \n"<<gmm2[0].getMean()<<"\n\n\n\n";
    
    std::cout<<gmm2.prior[1]<<"\n";
    std::cout<<"Covariance: \n"<<gmm2[1].getCovariance()<<"\n\n";
    std::cout<<"Mean: \n"<<gmm2[1].getMean()<<"\n\n\n\n";
    

    
}



