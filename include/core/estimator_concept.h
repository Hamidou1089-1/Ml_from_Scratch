//
// Created by hamid on 19/08/2025.
//

#pragma once

#include <iostream>
#include <concepts>
#include <Eigen/Dense>

template <typename T>
concept EstimatorConcept = requires(T estimator, const Eigen::MatrixXd& X, const Eigen::VectorXd& y_true) {
    {estimator.fit(X, y_true)} -> std::convertible_to<void>;
};

