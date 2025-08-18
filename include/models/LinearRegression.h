



#pragma once

#include "Predictor.h"
#include "Optimizer.h"
#include "LossFunction.h"
#include <memory>


class LinearRegression : public Predictor {
private:
    std::unique_ptr<Optimizer> _optimizer;
    std::unique_ptr<LossFunction> _loss;
    VectorXd _weights;
    double learningRate;
    long iterations;

public:

LinearRegression(std::unique_ptr<Optimizer> optimizer, std::unique_ptr<LossFunction> loss, double learningRate, long iterations);

void fit(const MatrixXd& X, const VectorXd& y) override;
VectorXd predict(const MatrixXd& X) const override;

};