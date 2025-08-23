


#pragma once

#include "loss_concept.h"
#include "optimizer_concept.h"
#include <memory>
#include <iostream>

template <OptimizerConcept OptimizerT, LossConcept LossT>
class LogisticRegression {
private:
    long nbr_iteration_;
    Eigen::VectorXd weights_;
    OptimizerT optimizer_;
    LossT loss_;

public:
    LogisticRegression(OptimizerT opt, LossT los, long nbr_iteration);

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;
};

#include "logistic_regression.tpp"


