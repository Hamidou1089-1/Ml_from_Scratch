//
// Created by hamid on 18/08/2025.
//

#pragma once

#include <eigen3/Eigen/Dense>

using namespace Eigen::MatrixXd;
using namespace Eigen::VectorXd;


class Optimizer {


public:
    virtual VectorXd step(const VectorXd& current_weights, const VectorXd& gradient) = 0;
};


