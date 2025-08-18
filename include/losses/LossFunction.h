//
// Created by hamid on 18/08/2025.
//

#pragma once
#include <eigen3/Eigen/Dense>

using namespace Eigen::MatrixXd;
using namespace Eigen::VectorXd;


class LossFunction {

public:
    virtual double loss(
        const VectorXd& y_true,
        const VectorXd& y_pred,
        const VectorXd& weights
    ) const = 0;

    virtual VectorXd gradient(
        const VectorXd& y_true,
        const VectorXd& y_pred,
        const MatrixXd& X,
        const VectorXd& weights
    ) const = 0;
};