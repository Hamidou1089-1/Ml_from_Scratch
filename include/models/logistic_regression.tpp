//
// Created by hamid on 19/08/2025.
//



template<OptimizerConcept OptimizerT, LossConcept LossT>
LogisticRegression<OptimizerT, LossT>::LogisticRegression(OptimizerT opt, LossT los, long nbr_iteration) : optimizer_(std::move(opt)), loss_(std::move(los)), nbr_iteration_(nbr_iteration) {}

template<OptimizerConcept OptimizerT, LossConcept LossT>
void LogisticRegression<OptimizerT, LossT>::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    Eigen::MatrixXd X_with_bias(X.rows(), X.cols() + 1);
    X_with_bias << X, Eigen::VectorXd::Ones(X.rows());

    weights_ = Eigen::VectorXd::Random(X.cols() + 1);

    for (int i = 0; i < nbr_iteration_; i++) {
        Eigen::VectorXd y_pred = predict(X_with_bias);
        Eigen::VectorXd gradient = loss_.compute_gradient(y, y_pred, X_with_bias, weights_);

        weights_ = optimizer_.step(weights_, gradient);

    }
}

template<OptimizerConcept OptimizerT, LossConcept LossT>
Eigen::VectorXd LogisticRegression<OptimizerT, LossT>::predict(const Eigen::MatrixXd& X) const{
    if (X.cols() == weights_.size() - 1) {
        std::cerr << "Warning: Auto-adding bias column to X" << std::endl;
        Eigen::MatrixXd X_with_bias(X.rows(), X.cols() + 1);
        X_with_bias << X, Eigen::VectorXd::Ones(X.rows());
        Eigen::VectorXd inv = (1.0 + (-X_with_bias*weights_).array().exp()).matrix();
        return inv.array().inverse().matrix();
    }
    else if (X.cols() != weights_.size()) {
        throw std::invalid_argument("X has wrong size");
        }
    Eigen::VectorXd inv = (1.0 + (-X*weights_).array().exp()).matrix();
    return inv.array().inverse().matrix();
}