//
// Created by hamid on 18/08/2025.
//

#pragma once

#include "Optimizer.h"

class GradientDescent: public Optimizer {
private:
    double learningRate;

public:
    GradientDescent(double learningRate);

    VectorXd step(const VectorXd& current_weights, const VectorXd& gradient) const override;
};


