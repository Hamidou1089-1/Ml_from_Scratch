//
// Created by hamid on 18/08/2025.
//

#pragma once



class Layer {


public:

    virtual VectorXd forward(const MatrixXd& input) = 0;
    virtual VectorXd backward(const VectorXd& gradient) = 0;
};