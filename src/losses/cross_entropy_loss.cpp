//
// Created by hamid on 18/08/2025.
//

#include "cross_entropy_loss.h"


#include <iostream>

double CrossEntropyLoss::compute_loss(
    const Eigen::VectorXd& y_true, 
    const Eigen::VectorXd& y_pred, 
    const Eigen::VectorXd& current_weight
) {
    const double eps = 1e-15;
    Eigen::VectorXd y_pred_clipped = y_pred.cwiseMax(eps).cwiseMin(1.0 - eps);

    Eigen::VectorXd log_y_pred = y_pred_clipped.array().log().matrix();
    Eigen::VectorXd log_1_minus_y_pred = (1.0 - y_pred_clipped.array()).log().matrix();

    Eigen::VectorXd loss_vec = -(y_true.cwiseProduct(log_y_pred) +
                                (1.0 - y_true.array()).matrix().cwiseProduct(log_1_minus_y_pred));

    return loss_vec.mean();
}

Eigen::VectorXd CrossEntropyLoss::compute_gradient(
    const Eigen::VectorXd& y_true, 
    const Eigen::VectorXd& y_pred, 
    const Eigen::MatrixXd& X, 
    const Eigen::VectorXd& current_weight
) {
    return X.transpose() * (y_pred - y_true) / y_true.size();
}