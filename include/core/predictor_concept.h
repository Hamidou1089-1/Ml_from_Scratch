//
// Created by hamid on 19/08/2025.
//

#pragma once


#include "estimator_concept.h"


template <typename T>
concept PredictorConcept = EstimatorConcept<T> && requires(T predictor, const Eigen::MatrixXd& X) {
    {predictor.predict(X)} -> std::convertible_to<Eigen::VectorXd>;

};