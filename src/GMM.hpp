#pragma once

#include "EM.hpp"
#include <Eigen/Dense>
#include <cmath>

template<class Float, int N>
Eigen::Matrix<Float, N, 1> weightedSum(
    const std::vector<Float>& a,
    const std::vector<Eigen::Matrix<Float, N, 1> >& x)
{
    assert( a.size() == x.size() );
    int n = x[0].size();
    Eigen::Matrix<Float, N, 1> s = Eigen::Matrix<Float, N, 1>::Zero(n);
    Float sa = 0.0;
    for(int i=0; i<x.size(); i++){
        s += a[i]*x[i];
    }
    return s;
}


template<class Float, int N>
class IndependentGaussian
        : ProbabilityDistribution<Eigen::Matrix<Float, N, 1>, Float > {
public:
    typedef Eigen::Matrix<Float, N, 1> Vector;
    const int n;
    Vector mean;
    Vector deviation;
    IndependentGaussian(int _n) : 
        n{_n}
    {
        static_assert(N==Eigen::Dynamic, "The constructor is only usable with dynamic number of dimensions");
        init();
    }
    IndependentGaussian() : 
        n{N}
    {
        static_assert(N!=Eigen::Dynamic, "The constructor is only usable with fixed number of dimensions");
        init();
    }
    Float operator()(const Vector& x){
        assert(x.size() == 0);
        Float p = 1.0;
        Float s = 0.0;
        for(int j=0; j<n; j++){
            p *= one_two_pi/deviation(j);
            s += - (x[j]-mean[j])*(x[j]-mean[j])/2.0/deviation[j]/deviation[j];
        }
        return p*exp(s);
    }
    void likelihoodEstimate(const std::vector<Float>& a, 
                            const std::vector<Vector>& x)
    {
        Float sumA = std::reduce(a.begin(), a.end());
        if(sumA == 0.0){
            return;
        }
        mean = weightedSum(a, x)/sumA;
        
        Vector s2 = Vector::Zero(x[0].size());
        for(int i=0; i<x.size(); i++){
            s2 += a[i]*(x[i]-mean).unaryExpr([&](auto xi_m){ return xi_m*xi_m; });
        }

        deviation = (s2/sumA).cwiseSqrt();
    } 
private:
    static constexpr Float one_two_pi = 1.0/sqrt(2*M_PI);
    void init(){
        mean = Vector::Zero(n);
        deviation = Vector::Constant(n, 1.0);
    }
};
