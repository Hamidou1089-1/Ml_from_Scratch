//
// Created by hamid on 10/08/2025.
//

#include "linear_regression.h"




LinearRegression::LinearRegression(OptimizerT opt, LossT los, long num_iterations) : optimizer_(std::move(opt)), loss_(std::move(los)), num_iterations_(num_iterations) {}

void LinearRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y_true) {
    weights_ = Eigen::VectorXd::Random(X.cols());



    for (int i = 0; i < num_iterations_; i++) {
        Eigen::VectorXd y_pred = X * weights_;
        Eigen::VectorXd gradient = loss_.compute_gradient(y_true, y_pred, X);
        weights_ = optimizer_.step(weights_, gradient);

    }

}

Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd& X) {
    return X * weights_;
}



