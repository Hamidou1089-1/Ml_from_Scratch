//
// Created by hamid on 10/08/2025.
//


template <OptimizerConcept OptimizerT, LossConcept LossT>
LinearRegression<OptimizerT, LossT>::LinearRegression(OptimizerT opt, LossT los, long num_iterations)
    : optimizer_(std::move(opt)), loss_(std::move(los)), num_iterations_(num_iterations) {}

template <OptimizerConcept OptimizerT, LossConcept LossT>
void LinearRegression<OptimizerT, LossT>::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y_true) {
    Eigen::MatrixXd X_with_bias(X.rows(), X.cols() + 1);
    X_with_bias << X, Eigen::VectorXd::Ones(X.rows());

    weights_ = Eigen::VectorXd::Random(X.cols() + 1);

    for (int i = 0; i < num_iterations_; i++) {
        Eigen::VectorXd y_pred = X_with_bias * weights_;
        Eigen::VectorXd gradient = loss_.compute_gradient(y_true, y_pred, X_with_bias, weights_);
        weights_ = optimizer_.step(weights_, gradient);
    }
}

template <OptimizerConcept OptimizerT, LossConcept LossT>
Eigen::VectorXd LinearRegression<OptimizerT, LossT>::predict(const Eigen::MatrixXd& X) const{
    if (X.cols() == weights_.size() - 1) {
        std::cerr << "Warning: Auto-adding bias column to X" << std::endl;
        Eigen::MatrixXd X_with_bias(X.rows(), X.cols() + 1);
        X_with_bias << X, Eigen::VectorXd::Ones(X.rows());
        return X_with_bias * weights_;
    } else if (X.cols() != weights_.size()) {
        throw std::invalid_argument("Dimension mismatch...");
    }
    return X * weights_;
}



