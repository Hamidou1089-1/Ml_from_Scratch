//
// Created by hamid on 19/08/2025.
//



template<OptimizerConcept OptimizerT, LossConcept LossT>
LogisticRegression<OptimizerT, LossT>::LogisticRegression(OptimizerT opt, LossT los,  int max_iter, double tol , int patience) 
: optimizer_(std::move(opt)), loss_(std::move(los)), max_iterations_(max_iter), tol_(tol), patience_(patience) {}

template<OptimizerConcept OptimizerT, LossConcept LossT>
void LogisticRegression<OptimizerT, LossT>::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y_true) {
    X_with_bias_ = Eigen::MatrixXd(X.rows(), X.cols() + 1);
    X_with_bias_ << X, Eigen::VectorXd::Ones(X.rows());

    

    weights_ = Eigen::VectorXd::Random(X.cols() + 1);

    y_pred_ = Eigen::VectorXd(X.rows());
    grad_ = Eigen::VectorXd(X.cols() + 1);

    std::vector<double> loss_history;
    double best_loss = std::numeric_limits<double>::max();
    int no_improve_count = 0;
    int check_frequency = 100;

    for (int i = 0; i < max_iterations_; i++) {
        y_pred_.noalias() = predict(X_with_bias_);
        grad_.noalias() = loss_.compute_gradient(y_true, y_pred_, X_with_bias_, weights_);

        weights_ = optimizer_.step(weights_, grad_);


        if (i % check_frequency == 0) {
            double current_loss = loss_.compute_loss(y_true, y_pred_, weights_);
            
            
            double relative_improvement = (best_loss - current_loss) / std::abs(best_loss);
            if (relative_improvement < tol_) {
                no_improve_count += check_frequency;
            } else {
                no_improve_count = 0;
                best_loss = current_loss;
            }
            
            
            double grad_norm = grad_.norm();
            if (grad_norm < tol_) {
                std::cout << "Convergence par gradient norm à " << i << " itérations" << std::endl;
                break;
            }
            
            
            if (no_improve_count >= patience_) {
                std::cout << "Early stopping par patience à " << i << " itérations" << std::endl;
                break;
            }
        }

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