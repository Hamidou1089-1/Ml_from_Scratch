

#include "elastic_net.h"
#include <Eigen/Dense>
#include <memory>

elastic_net::elastic_net(
    double lambda1, 
    double lambda2, 
    LossT base_loss)
    : lambda1_(lambda1), lambda2_(lambda2), base_loss_(std::move(base_loss)) {}

double elastic_net::compute_loss(
    const Eigen::VectorXd& y_true, 
    const Eigen::VectorXd& y_pred, 
    const Eigen::VectorXd& current_weight
) {
    return base_loss_.compute_loss(y_true, y_pred) + lambda1_*current_weight.cwiseAbs().sum() + lambda2_*current_weight.squared_norm();
}

Eigen::VectorXd elastic_net::compute_gradient(
        const Eigen::VectorXd& y_true,
        const Eigen::VectorXd& y_pred,
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& current_weight) {
    return base_loss_.compute_gradient(y_true, y_pred, X, current_weight) + lambda1_*current_weight.cwiseSign() + 2.0*lambda2_*current_weight;
}