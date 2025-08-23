



#pragma once

#include <Eigen/Dense>
#include <memory>
#include "loss_concept.h"
#include "optimizer_concept.h"
#include <vector>
#include <iostream>



template <OptimizerConcept OptimizerT, LossConcept LossT>
class LinearRegression {
private:
    OptimizerT optimizer_;
    LossT loss_;
    Eigen::VectorXd weights_;
    mutable Eigen::MatrixXd X_with_bias_;
    mutable Eigen::VectorXd y_pred_, grad_;
    double tol_ = 1e-10;           
    int patience_ = 1000;          
    int max_iterations_;

public:
    LinearRegression(OptimizerT opt, LossT los, int max_iter, double tol = 1e-10, int patience = 1000);
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y_true);
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;
};

#include "linear_regression.tpp"
