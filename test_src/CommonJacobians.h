//=========================================================================
// Nyx Solver - Non Linear Problems Solver
//
// Copyright 2018 Pierre Guilbert
// Author: Pierre Guilbert (spguilbert@gmail.com)
// Data: 02-11-2018
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//=========================================================================

#ifndef COMMON_JACOBIANS_H
#define COMMON_JACOBIANS_H

//STD
#include <vector>

// LOCAL
#include "Function.h"
#include "Jacobian.h"
#include "Tools.h"

//-------------------------------------------------------------------------
template <typename T>
class ParametricSphereJacobian : public nyx::Jacobian<T, 2, 3>
{
public:
  Eigen::Matrix<T, 3, 2> operator()(Eigen::Matrix<T, 2, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, 3, 2> Y = Eigen::Matrix<T, 3, 2>::Zero();

    // Explicit surface function differential of a sphere
    // X(0) represents the azimutal angle
    // X(1) represents the vertical angle
    Y(0, 0) = -std::sin(X(0)) * std::sin(X(1)); // DY0 / DX0
    Y(0, 1) = std::cos(X(0)) * std::cos(X(1)); // DY0 / DX1

    Y(1, 0) = std::cos(X(0)) * std::sin(X(1)); // DY1 / DX0
    Y(1, 1) = std::sin(X(0)) * std::cos(X(1)); // DY1 / DX1

    Y(2, 0) = 0; // DY1 / DX0
    Y(2, 1) = -std::sin(X(1)); // DY1 / DX1

    return Y;
  }
};

//-------------------------------------------------------------------------
template <typename T>
class ImplicitSphereJacobian : public nyx::Jacobian<T, 3, 1>
{
public:
  Eigen::Matrix<T, 1, 3> operator()(Eigen::Matrix<T, 3, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, 1, 3> Y = Eigen::Matrix<T, 1, 3>::Zero();

    // Implicit surface function differential of a sphere
    Y(0, 0) = X(0) / std::sqrt(X(0)*X(0) + X(1)*X(1) + X(2)*X(2)); // DY0 / DX0
    Y(0, 1) = X(1) / std::sqrt(X(0)*X(0) + X(1)*X(1) + X(2)*X(2)); // DY0 / DX1
    Y(0, 2) = X(2) / std::sqrt(X(0)*X(0) + X(1)*X(1) + X(2)*X(2)); // DY0 / DX2

    return Y;
  }
};

//-------------------------------------------------------------------------
template <typename T>
class EulerAngleSO3MappingJacobian : public nyx::Jacobian<T, 3, 9>
{
public:
  Eigen::Matrix<T, 9, 3> operator()(Eigen::Matrix<T, 3, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, 9, 3> Y = Eigen::Matrix<T, 9, 3>::Zero();

    double crx = std::cos(X(0)); double srx = std::sin(X(0));
    double cry = std::cos(X(1)); double sry = std::sin(X(1));
    double crz = std::cos(X(2)); double srz = std::sin(X(2));

    Y(0, 0) = 0; // dR11 /  drx
    Y(0, 1) = -sry * crz; // dR11 / dry
    Y(0, 2) = -srz * cry; // dR12 / drz

    Y(1, 0) = srz * srx + crz * sry * crx; // dR12 / drx
    Y(1, 1) = crz * cry * srx; // dR12 / dry
    Y(1, 2) = -crz * crx - srz * sry * srx; // dR12 / drz

    Y(2, 0) = srz * crx - crz * sry * srx; // dR13 / drx
    Y(2, 1) = crz * cry * crx; // dR13 / dry
    Y(2, 2) = crz * srx - srz * sry * crx; // dR13 / drz

    Y(3, 0) = 0; // dR21 / drx
    Y(3, 1) = -srz * sry; // dR21 / dry
    Y(3, 2) = crz * cry; // dR21 / drz

    Y(4, 0) = -crz * srx + srz * sry * crx; // dR22 / drx
    Y(4, 1) = srz * cry * srx; // dR22 / dry
    Y(4, 2) = -srz * crx + crz * sry * srx; // dR22 / drz

    Y(5, 0) = -crz * crx - srz * sry * srx; // dR23 / drx
    Y(5, 1) = srz * cry * crx; // dR23 / dry
    Y(5, 2) = srz * srx + crz * sry * crx; // dR23 / drz

    Y(6, 0) = 0; // dR31 / drx
    Y(6, 1) = -cry; // dR31 / dry
    Y(6, 2) = 0; // dR31 / drz

    Y(7, 0) = cry * crx; // dR32 / drx
    Y(7, 1) = -sry * srx; // dR32 / dry
    Y(7, 2) = 0; // dR32 / drz;

    Y(8, 0) = -cry * srx; // dR33 / drx
    Y(8, 1) = -sry * crx; // dR33 / dry
    Y(8, 2) = 0; // dR33 / drz

    return Y;
  }
};

//-------------------------------------------------------------------------
template <typename T>
class MultiVarPolynomialJacobian : public nyx::Jacobian<T, 2, 2>
{
public:
  Eigen::Matrix<T, 2, 2> operator()(Eigen::Matrix<T, 2, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, 2, 2> Y = Eigen::Matrix<T, 2, 2>::Zero();

    Y(0, 0) = 3.0 * std::pow(X(0), 2) * std::sqrt(X(1)); // dY1 / dX1
    Y(0, 1) = std::pow(X(0), 3) / (2.0 * std::sqrt(X(1))); // dY1 / dX2
    Y(1, 0) = 2.0 * X(0) * std::pow(X(1), 3); // dY2 / dX1
    Y(1, 1) = std::pow(X(0), 2) * 3.0 * std::pow(X(1), 2); // dY2 / dX2

    return Y;
  }
};

//-------------------------------------------------------------------------
template <typename T>
class NonLinearFunction2ParametersJacobian : public nyx::Jacobian<T, 6, 1>
{
public:
  NonLinearFunction2ParametersJacobian()
    : nyx::Jacobian<T, 6, 1>()
  {
    this->X.resize(1, 0);
  }

  Eigen::Matrix<T, 1, 6> operator()(Eigen::Matrix<T, 6, 1> W)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, 1, 6> Y = Eigen::Matrix<T, 1, 6>::Zero();

    Y(0) = std::exp(-std::pow(X[0] - W[1], 2) / (2.0 * W[2])); // df / dw0
    Y(1) = W[0] * (X[0] - W[1]) / W[2] * std::exp(-std::pow(X[0] - W[1], 2) / (2.0 * W[2])); // df / dw1
    Y(2) = W[0] * std::pow(X[0] - W[1], 2) / (2.0 * W[2] * W[2]) * std::exp(-std::pow(X[0] - W[1], 2) / (2.0 * W[2])); // df / dw2
    Y(3) = std::cos(W[4] * X[0]); // df / dw3
    Y(4) = -W[3] * X[0] * std::sin(W[4] * X[0]); // df / dw4
    Y(5) = X[0] * X[0];

    return Y;
  }

  std::vector<T> X;
};

//-------------------------------------------------------------------------
template <typename T>
class NonLinearFunction3Jacobian : public nyx::Jacobian<T, 2, 2>
{
public:
  Eigen::Matrix<T, 2, 2> operator()(Eigen::Matrix<T, 2, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, 2, 2> Y = Eigen::Matrix<T, 2, 2>::Zero();

    Y(0) = std::pow(X(0) - 4, 2.0) * std::pow(X(1) - 10.0, 2.0); // X(0) = 4 is root
    Y(1) = std::pow(X(0), 2.0) * std::pow(X(1) + 7, 2.0); // X(1) = -7 is root

    Y(0, 0) = 2 * (X(0) - 4) * std::pow(X(1) - 10.0, 2.0); // dY0 / dX0
    Y(0, 1) = 2 * (X(1) - 10.0) * std::pow(X(0) - 4, 2.0); // // dY0 / dX1

    Y(1, 0) = 2 * X(0) *std::pow(X(1) + 7, 2.0); // dY1 / dX0
    Y(1, 1) = 2 * (X(1) + 7) * std::pow(X(0), 2.0); // // dY1 / dX1

    return Y;
  }
};
#endif // COMMON_JACOBIANS_H