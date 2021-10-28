#pragma once

#include <Eigen/Dense>
#include <cmath>

template<class P>
class Generator;


//A class to generate normal distribution values of type T
//with expected value 0.0 and variation 1.0
template<class T>
class NormalGenerator {
private: 
    int hasValue = 0;
    T value;
public:
    T operator()(){
        if(hasValue){
            hasValue = 0;
            return value;
        } else {
            T x, y;
            T s;
            do{
                x = 2.0 * (T) rand()/RAND_MAX - 1.0;
                y = 2.0 * (T) rand()/RAND_MAX - 1.0;
                s = x*x + y*y;
            } while(s>1.0);
            T tmp = sqrt(-2*log(s)/s);
            value = x*tmp;
            hasValue = 1;
            return y*tmp;
        }
    }
};



//A class to generate N-dimensional multivariate normal distributions of floating type Float
template<class Float, int N>
class GaussianGenerator {
private:
    typedef Eigen::Matrix<Float, N, 1> Vec;
    typedef Eigen::Matrix<Float, N, N> Mat;
    NormalGenerator<Float> ng;
    Vec mean;
    Mat transform;
public:
    GaussianGenerator(const Mat& covar, const Vec& _mean) :
        mean{_mean} 
    {
        Eigen::SelfAdjointEigenSolver<Mat> eigenSolver(covar);
        transform = eigenSolver.eigenvectors() 
                  * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }
    
    Vec operator()(){
        Vec normal(mean.size());
        return mean + transform * normal.unaryExpr([&](auto x){ return ng(); });
    }

};


template<class Float, int N>
class Generator<IndependentGaussian<Float, N> > {
private:
    typedef Eigen::Matrix<Float, N, 1> Vec;
    typedef Eigen::Matrix<Float, N, N> Mat;
    GaussianGenerator<Float, N> gg;
public:
    Generator(const IndependentGaussian<Float, N>& gaussian) :
        gg(gaussian.deviation.cwiseAbs2().asDiagonal(), gaussian.mean){
    }
    Generator(const Vec& deviation, const Vec& mean) :
        gg(deviation.cwiseAbs2().asDiagonal(), mean){
    }
    Vec operator()(){
        return gg();
    }
};


