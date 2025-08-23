//
// Created by hamid on 19/08/2025.
//

#include "logistic_regression.h"
#include "gradient_descent.h"
#include "cross_entropy_loss.h"
#include <Eigen/Dense>
#include <iostream>




int main() {
    LogisticRegression<GradientDescent, CrossEntropyLoss> model(
    GradientDescent(0.001),
    CrossEntropyLoss(),
    100000
);

    Eigen::MatrixXd X = Eigen::MatrixXd::Random(100, 80);
    Eigen::MatrixXd X_with_bias(X.rows(), X.cols() + 1);
    X_with_bias << X, Eigen::VectorXd::Ones(X.rows());
    Eigen::VectorXd weights = Eigen::VectorXd::Random(81);

    Eigen::VectorXd inv = (1.0 + (-X_with_bias*weights).array().exp()).matrix();
    Eigen::VectorXd test = inv.array().inverse().matrix();
    Eigen::ArrayXd zeros = Eigen::ArrayXd::Zero(test.size());
    Eigen::ArrayXd ones = Eigen::ArrayXd::Ones(test.size());
    Eigen::VectorXd y_true = (test.array() < 0.5).select(zeros, ones);

    model.fit(X, y_true);
    Eigen::VectorXd output = model.predict(X_with_bias);
    output = (output.array() < 0.5).select(zeros, ones);;
    for (int i = 0; i < test.size(); i++) {
        std::cout << "y_true = " << y_true[i] << " " ;
        std::cout << "Prediction = " << output[i] << "\n";
}
    std::cout << std::endl;
return 0;


}