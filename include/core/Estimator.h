//
// Created by hamid on 18/08/2025.
//

#pragma once

#include <eigen3/Eigen/Dense>

using namespace Eigen::MatrixXd;
using namespace Eigen::VectorXd;

class Estimator {

public:
    virtual void fit(const MatrixXd& X, const VectorXd& y) = 0;
};