#pragma once

#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>

template <class X, class Float>
class ProbabilityDistribution {
public:
    typedef Float float_type;
    typedef X sample_type;
    /*
    virtual Float operator()(const X& x) const = 0;
    virtual void likelihoodEstimate(const std::vector<Float>& a, const std::vector<X>& x) = 0;
    */
};

template <class C>
concept Container = 
    requires(C c) {
        c.begin();
        c.end();
        c.begin()++;
        c.size();
    };

template <class C, class T>
concept ContainerOf = 
    requires(C c) {
        { *c.begin() } -> std::convertible_to<T>;
        c.end();
        c.begin()++;
    };


template <class P>
concept HasLikelihoodEstimate =
    requires(P p, std::vector<typename P::float_type> a, std::vector<typename P::sample_type> x) {
        p.likelihoodEstimate(a, x);
    };


template <HasLikelihoodEstimate P>
class MixtureModel {
private:
    std::vector<P> variable;
public:
    typedef typename P::float_type float_type;
    typedef typename P::sample_type sample_type;
    typedef P variable_type;
    std::vector<float_type> prior;
    float_type operator()(const sample_type& x) const{
        float_type p = 0.0;
        for(size_t k=0; k<size(); k++){
            p += prior[k]*variable[k](x);
        }
        return p;
    }
    float_type operator()(int k, const sample_type& x) const{
        return prior[k]*variable[k](x);
    }
    size_t size() const{
        return variable.size();
    }
    const P& operator[](size_t k) const{
        return variable[k];
    }
    P& operator[](size_t k){
        return variable[k];
    }
    MixtureModel(int nClasses) : 
        variable(nClasses),
        prior(nClasses, 1.0/nClasses)
    {
    }
};


template <class Mixture>
class EM {
public:
    typedef typename Mixture::float_type float_type;
    typedef typename Mixture::sample_type sample_type;
private:
    void makeSumToOne(std::vector<std::vector<float_type> >& a){
        for(size_t i=0; i<a[0].size(); i++){
            float_type sum_a = 0.0;
            for(size_t k=0; k<mixture.size(); k++){
                sum_a += a[k][i];
            }
            if(sum_a != 0.0){
                for(size_t k=0; k<mixture.size(); k++){
                    a[k][i] /= sum_a;
                }
            } else {
                //Something went wrong! Make them all equal.
                for(size_t k=0; k<mixture.size(); k++){
                    a[k][i] /= 1.0/mixture.size();
                }
            }
        }
    }
public:
    Mixture& mixture;
    EM(Mixture& _mixture) : mixture{_mixture}
    {
    }

    template <Container C>
    requires ContainerOf<C, sample_type>
    void expectation(const C& xs,
                     std::vector<std::vector<float_type> >& a)
    {
        for(size_t k=0; k<mixture.size(); k++){
            std::transform(xs.begin(), xs.end(), a[k].begin(),
            [&] (sample_type x) -> float_type { 
                return mixture(k, x); 
            } );
        }
    }
    template <Container C>
    requires ContainerOf<C, sample_type>
    void maximization(const C& xs,
                      std::vector<std::vector<float_type> >& a)
    {
        for(size_t k=0; k<mixture.size(); k++){
            mixture.prior[k] = std::reduce(a[k].begin(), a[k].end()) / a[k].size();
            mixture[k].likelihoodEstimate(a[k], xs);
        }
    }

    //perform one Expectation-Maximization step with learning sample xs
    template <Container C>
    requires ContainerOf<C, sample_type>
    void iterate(const C& xs)
    {
        std::vector<std::vector<float_type> > a(
            mixture.size(), 
            std::vector<float_type>(xs.size()) );

        expectation(xs, a);
        makeSumToOne(a);
        maximization(xs, a);
    }
    
    template <Container C>
    requires ContainerOf<C, sample_type>
    void init(const C& xs){
        std::vector<std::vector<float_type> > a(
            mixture.size(), 
            std::vector<float_type>(xs.size()) );
        for(size_t k=0; k<mixture.size(); k++){
            for(size_t i=0; i<a[k].size(); i++){
                a[k][i] = (float_type) rand()/RAND_MAX;
            }
        }

        makeSumToOne(a);
        maximization(xs, a);
    }
};

