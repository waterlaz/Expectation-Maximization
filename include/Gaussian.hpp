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
private:
    typedef Eigen::Matrix<Float, N, 1> Vec;
public:
    size_t n;
    Vec deviation;
    Vec mean;
    
    IndependentGaussian(size_t _n = N) : 
        n{_n}
    {
        assert((N==Eigen::Dynamic || n==N) && "Wrong number of dimensions");
        init();
    }
    
    IndependentGaussian(const Vec& _deviation, const Vec& _mean) : 
        n{(size_t)_deviation.size()},
        deviation{_deviation},
        mean{_mean}
    {
        assert((N==Eigen::Dynamic || n==N) && "Wrong number of dimensions");
    }
    
    Float operator()(const Vec& x) const{
        assert((size_t)x.size() == n);
        Float p = 1.0;
        Float s = 0.0;
        for(size_t j=0; j<n; j++){
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


template<class Float, int N>
class Gaussian
        : public ProbabilityDistribution<Eigen::Matrix<Float, N, 1>, Float > {
private:
    typedef Eigen::Matrix<Float, N, 1> Vec;
    typedef Eigen::Matrix<Float, N, N> Mat;
    size_t n;
public:
    Vec mean;
    class : public Mat{
    public:
        Float constantFactor{ calculateConstantFactor() };
        void operator=(const Mat& _invCovariance){
            assert(_invCovariance.rows() == _invCovariance.cols() && "The inverse covariance matrix must be square");
            Mat::operator=(_invCovariance);
            constantFactor = calculateConstantFactor();
        }
    private:
        Float calculateConstantFactor(){
            return pow(one_two_pi, this->rows())*sqrt(this->determinant());
        }
    } invCovariance;

    Gaussian(size_t _n=N) : 
        n{_n},
        mean{ Vec::Zero(n) },
        invCovariance{ Vec::Constant(n, 1.0).asDiagonal() }
    {
        assert((N==Eigen::Dynamic || n==N) && "Wrong number of dimensions");
    }
    
    Gaussian(const Mat& _invCovariance, const Vec& _mean) : 
        n{(size_t)_mean.size()},
        mean{_mean},
        invCovariance{_invCovariance}
    {
        assert((N==Eigen::Dynamic || n==N) && "Wrong number of dimensions");
    }
    
    Float operator()(const Vec& x) const{
        assert((size_t)x.size() == n);
        return invCovariance.constantFactor * exp(-0.5*(x-mean).dot(invCovariance*(x-mean)));
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
        
        Mat Cov = std::transform_reduce(
            a.begin(), a.end(), x.begin(),
            (Mat)Mat::Zero(n, n),
            std::plus<>(),
            [&](Float ai, const Vec& xi) -> Mat {
                return ai*(xi-mean)*(xi-mean).transpose();
            });
        //FIXME: what if there is no inverse?
        invCovariance = (Cov/sumA).inverse();
    } 
private:
    static constexpr Float one_two_pi = 1.0/sqrt(2*M_PI);
};


