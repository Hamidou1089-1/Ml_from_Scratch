


#pragma once

#include "loss_concept.h"
#include "optimizer_concept.h"
#include <memory>
#include <iostream>
#include <vector>

template <OptimizerConcept OptimizerT, LossConcept LossT>
class LogisticRegression {
private:
    long nbr_iteration_;
    Eigen::VectorXd weights_;
    OptimizerT optimizer_;
    LossT loss_;
    mutable Eigen::MatrixXd X_with_bias_;
    mutable Eigen::VectorXd y_pred_, grad_;
    double tol_ = 1e-10;           
    int patience_ = 1000;          
    int max_iterations_;

public:
    LogisticRegression(OptimizerT opt, LossT los, int max_iter, double tol = 1e-10, int patience = 1000);

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;
};

#include "logistic_regression.tpp"


