#include <iostream>
#include <fstream>
#include <list>
#include "Gaussian.hpp"
#include "GaussianGenerator.hpp"
#include "EM.hpp"

using namespace Eigen;

int main(){
    // a mixture of two 2-dimensional gaussian distributions
    MixtureModel<IndependentGaussian<double, 2> > gaussianMixture(2);
    
    // fill in the parameters of gaussian distributions
    // expected value (7, 9) and deviation (2, 1)
    gaussianMixture[0] = IndependentGaussian<double, 2>( Eigen::Vector2d(2.0, 1.0), Eigen::Vector2d(7.0, 9.0) );
    // expected value (-5,-5) and deviation (1, 2)
    gaussianMixture[1] = IndependentGaussian<double, 2>( Eigen::Vector2d(1.0, 2.0), Eigen::Vector2d(-5.0, -5.0) );
    // prior class probabilities in the mixture
    gaussianMixture.prior[0] = 0.3;
    gaussianMixture.prior[1] = 0.7;
    
    // create a random variable generator from gaussianMixture distribution
    Generator gaussianGenerator(gaussianMixture);

    // generate 1000 samples from gaussianMixture distribution
    std::vector<Eigen::Vector2d> xs(1000);
    for(auto&& x:xs){
        x = gaussianGenerator();
    }
   
    // try to use the EM algorithm to learn the gaussian mixture from generated sample
    // general gaussian distribution case
    GeneralMixtureModel<Vector2d, double> learnedGaussianMixture(2);
    Gaussian<double, 2> gaussian1;
    Gaussian<double, 2> gaussian2;
    learnedGaussianMixture.set(0, gaussian1);
    learnedGaussianMixture.set(1, gaussian2);

    EM emGeneral(learnedGaussianMixture);
    
    
    
    //initialize distribution parameters according to learning sample
    emGeneral.init(xs);
    
    //perform 100 EM algorithm iterations
    for(int i=0; i<100; i++){
        emGeneral.iterate(xs);
    }
    
    
    std::cout<<"General Gaussian:\n";
    std::cout<<"Class1 prior probability:  "<<learnedGaussianMixture.prior[0]<<"\n";
    std::cout<<"Caovariance: \n"<<gaussian1.invCovariance.inverse()<<"\n\n";
    std::cout<<"Mean: \n"<<gaussian1.mean<<"\n\n\n";
    
    std::cout<<"Class2 prior probability:  "<<learnedGaussianMixture.prior[1]<<"\n";
    std::cout<<"Covariance: \n"<<gaussian2.invCovariance.inverse()<<"\n\n";
    std::cout<<"Mean: \n"<<gaussian2.mean<<"\n\n\n\n";
    

    
}



