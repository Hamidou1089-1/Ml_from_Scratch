//
// Created by hamid on 18/08/2025.
//

#include "gradient_descent.h"


GradientDescent::GradientDescent(double learning_rate) : learning_rate_( learning_rate) {}


Eigen::VectorXd GradientDescent::step(const Eigen::VectorXd& weights, const Eigen::VectorXd& gradient) {
    return weights - learning_rate_*gradient;
}