#pragma once

#include <Eigen/Dense>
#include <vector>
#include "EM.hpp"

template<class Mixture>
class Generator {
public:
    typedef typename Mixture::float_type float_type;
    typedef typename Mixture::sample_type sample_type;
    typedef typename Mixture::variable_type variable_type;
private:
    std::vector<float_type> prior;
    std::vector<Generator<variable_type> > variableGenerator;
public:
    Generator(const Mixture& mixture) :
        prior{mixture.prior}
    {
        for(size_t k=0; k<mixture.size(); k++){
            variableGenerator.emplace_back( Generator<variable_type>(mixture[k]) );
        }
    }
    sample_type operator()(){
        float_type p = (float_type) rand()/RAND_MAX;
        float_type sum = 0.0;
        for(size_t k=0; k<prior.size(); k++){
            sum += prior[k];
            if( sum>=p ){
                return variableGenerator[k]();
            }
        }
        return variableGenerator.back()();
    }
};

