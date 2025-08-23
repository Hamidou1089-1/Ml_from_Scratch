//
// Created by hamid on 19/08/2025.
//

#pragma once

#include <iostream>
#include <concepts>
#include <Eigen/Dense>


template <typename T>
concept OptimizerConcept = requires(T optimizer, const Eigen::VectorXd& weights,const Eigen::VectorXd& gradient) {
    {optimizer.step(weights, gradient)} -> std::convertible_to<Eigen::VectorXd>;
};

