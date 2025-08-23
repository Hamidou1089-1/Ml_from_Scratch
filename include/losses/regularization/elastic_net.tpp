

template <LossConcept LossT>
elastic_net<LossT>::elastic_net(
    double lambda1, 
    double lambda2, 
    LossT base_loss)
    : lambda1_(lambda1), lambda2_(lambda2), base_loss_(std::move(base_loss)) {}


template <LossConcept LossT>
double elastic_net<LossT>::compute_loss(
    const Eigen::VectorXd& y_true, 
    const Eigen::VectorXd& y_pred, 
    const Eigen::VectorXd& current_weight
) {
    return base_loss_.compute_loss(y_true, y_pred) + lambda1_*current_weight.cwiseAbs().sum() + lambda2_*current_weight.squaredNorm();
}

template <LossConcept LossT>
Eigen::VectorXd elastic_net<LossT>::compute_gradient(
        const Eigen::VectorXd& y_true,
        const Eigen::VectorXd& y_pred,
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& current_weight) {
    return base_loss_.compute_gradient(y_true, y_pred, X, current_weight) + lambda1_*current_weight.cwiseSign() + 2.0*lambda2_*current_weight;
} 