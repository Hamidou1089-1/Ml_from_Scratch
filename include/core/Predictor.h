//
// Created by hamid on 18/08/2025.
//

#pragma once

#include "Estimator.h"

class Predictor : public Estimator {

public:
    virtual VectorXd predict(const MatrixXd& X) = 0;
};