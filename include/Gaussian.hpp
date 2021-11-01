#pragma once

#include "EM.hpp"
#include <Eigen/Dense>
#include <cmath>

template<class Float, class C>
requires ContainerOf<C, Eigen::Matrix<Float, C::value_type::RowsAtCompileTime, 1> >
Eigen::Matrix<Float, C::value_type::RowsAtCompileTime, 1> weightedSum(
    const std::vector<Float>& a,
    const C& x)
{
    typedef Eigen::Matrix<Float, C::value_type::RowsAtCompileTime, 1> Vec;
    assert( a.size() == x.size() );
    int n = x.begin()->size();
    return std::transform_reduce(a.begin(), a.end(), x.begin(), 
                                 (Vec)Vec::Zero(n));
}


template<class Float, int N>
class IndependentGaussian
        : public ProbabilityDistribution<Eigen::Matrix<Float, N, 1>, Float > {
public:
    typedef Eigen::Matrix<Float, N, 1> Vec;
    int n;
    Vec deviation;
    Vec mean;
    IndependentGaussian(int _n) : 
        n{_n}
    {
        static_assert(N==Eigen::Dynamic, "The constructor is only usable with dynamic number of dimensions");
        init();
    }
    IndependentGaussian(int _n, const Vec& _deviation, const Vec& _mean) : 
        n{_n},
        deviation{_deviation},
        mean{_mean}
    {
        static_assert(N==Eigen::Dynamic, "The constructor is only usable with dynamic number of dimensions");
    }
    IndependentGaussian() : 
        n{N}
    {
        static_assert(N!=Eigen::Dynamic, "The constructor is only usable with fixed number of dimensions");
        init();
    }
    IndependentGaussian(const Vec& _deviation, const Vec& _mean) : 
        n{N},
        deviation{_deviation},
        mean{_mean}
    {
        static_assert(N!=Eigen::Dynamic, "The constructor is only usable with fixed number of dimensions");
    }
    Float operator()(const Vec& x) const{
        assert(x.size() == n);
        Float p = 1.0;
        Float s = 0.0;
        for(int j=0; j<n; j++){
            p *= one_two_pi/deviation(j);
            s += - (x[j]-mean[j])*(x[j]-mean[j])/2.0/deviation[j]/deviation[j];
        }
        return p*exp(s);
    }
    template <Container C>
    requires ContainerOf<C, Vec>
    void likelihoodEstimate(const std::vector<Float>& a, 
                            const C& x)
    {
        Float sumA = std::reduce(a.begin(), a.end());
        if(sumA == 0.0){
            return;
        }
        mean = weightedSum(a, x)/sumA;
        
        Vec s2 = std::transform_reduce(
            a.begin(), a.end(), x.begin(),
            (Vec)Vec::Zero(n),
            std::plus<>(),
            [&](Float ai, const Vec& xi) -> Vec {
                return ai*(xi-mean).unaryExpr([&](auto xi_m){ return xi_m*xi_m; });
            });

        deviation = (s2/sumA).cwiseSqrt();
    } 
private:
    static constexpr Float one_two_pi = 1.0/sqrt(2*M_PI);
    void init(){
        mean = Vec::Zero(n);
        deviation = Vec::Constant(n, 1.0);
    }
};

