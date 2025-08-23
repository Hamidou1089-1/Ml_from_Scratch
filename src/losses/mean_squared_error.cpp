//
// Created by hamid on 18/08/2025.
//

#include "mean_squared_error.h"



double MeanSquaredError::compute_loss(
    const Eigen::VectorXd& y_true, 
    const Eigen::VectorXd& y_pred, 
    const Eigen::VectorXd& current_weight
) {
    return 0.5 * (y_true - y_pred).squaredNorm() / y_true.size();
}

Eigen::VectorXd MeanSquaredError::compute_gradient(
    const Eigen::VectorXd& y_true, 
    const Eigen::VectorXd& y_pred, 
    const Eigen::MatrixXd& X, 
    const Eigen::VectorXd& current_weight
) {
    return -2.0 * X.transpose()*(y_true - y_pred) / y_true.size();
}