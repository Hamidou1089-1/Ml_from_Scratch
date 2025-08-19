//
// Created by hamid on 18/08/2025.
//

#pragma once

#include "LossFunction.h"


class MeanSquaredError: public LossFunction {


public:

    double loss(
        const VectorXd& y_true,
        const VectorXd& y_pred,
        const VectorXd& weights
    ) const override;

    VectorXd gradient(
        const VectorXd& y_true,
        const VectorXd& y_pred,
        const VectorXd& weights,
        const MatrixXd& X
    ) const override;
};


