//
// Created by hamid on 18/08/2025.
//

#pragma once

#include <Eigen/Dense>

class MeanSquaredError{


public:

    double compute_loss(
        const Eigen::VectorXd& y_true,
        const Eigen::VectorXd& y_pred,
        const Eigen::VectorXd& current_weight
    );

    Eigen::VectorXd compute_gradient(
        const Eigen::VectorXd& y_true,
        const Eigen::VectorXd& y_pred,
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& current_weight
    );
};



