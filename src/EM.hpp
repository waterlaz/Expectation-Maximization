#pragma once

#include <vector>
#include <numeric>
#include <algorithm>

template <class X>
class ProbabilityDistribution {
public:
    virtual double operator()(const X& x) = 0;
    virtual void likelihoodEstimate(const std::vector<double>& a, const std::vector<X>& x) = 0;
};

template <class X>
class MixtureModel {
public:
    std::vector<ProbabilityDistribution<X> > px;
    std::vector<double> pk;
    double operator()(const X& x){
        double p = 0.0;
        for(unsigned int k=0; k<size(); k++){
            p += pk[k]*px[k](x);
        }
        return p;
    }
    double operator()(int k, const X& x){
        return pk[k]*px[k](x);
    }
    size_t size(){
        return pk.size();
    }
};



template <class X>
class EM {
private:
public:
    MixtureModel<X>& mixture;
    EM(MixtureModel<X>& _mixture) : mixture{_mixture}
    {
    }
    //perform one Expectation-Maximization step with learning sample xs
    void iterate(const std::vector<X>& xs){
        std::vector<std::vector<double> > a(
            mixture.size(), 
            std::vector<double>(xs.size()) );
        //Expectation:
        for(unsigned int i=0; i<xs.size(); i++){
            double sum_a = 0.0;
            for(unsigned int k=0; k<mixture.size(); k++){
                a[k][i] = mixture(k, xs[i]);
                sum_a += a[k][i];
            }
            if(sum_a != 0.0){
                for(unsigned int k=0; k<mixture.size(); k++){
                    a[k][i] /= sum_a;
                }
            }
        }
        //Maximization:
        for(unsigned int k=0; k<mixture.size(); k++){
            mixture.pk[k] = std::reduce(a[k].begin(), a[k].end()) / a[k].size();
            mixture.px[k].likelihoodEstimate(a[k], xs);
        }
    }
};

