



#pragma once

#include <Eigen/Dense>
#include <memory>
#include "loss_concept.h"
#include "optimizer_concept.h"



template <OptimizerConcept OptimizerT, LossConcept LossT>
class LinearRegression {
private:
    OptimizerT optimizer_;
    LossT loss_;
    long num_iterations_;
    Eigen::VectorXd weights_;

public:
    LinearRegression(OptimizerT opt, LossT los, long num_iterations);
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y_true);
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;
};

#include "linear_regression.tpp"
