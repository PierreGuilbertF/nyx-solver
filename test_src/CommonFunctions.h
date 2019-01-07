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

#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

//STD
#include <vector>

// LOCAL
#include "Function.h"
#include "Jacobian.h"
#include "Tools.h"

//-------------------------------------------------------------------------
template <typename T>
class ParametricSphere : public nyx::Function<T, 2, 3>
{
public:
  Eigen::Matrix<T, 3, 1> operator()(Eigen::Matrix<T, 2, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, 3, 1> Y = Eigen::Matrix<T, 3, 1>::Zero();

    // Explicit surface function of a sphere
    // X(0) represents the azimutal angle
    // X(1) represents the vertical angle
    Y(0) = std::cos(X(0)) * std::sin(X(1));
    Y(1) = std::sin(X(0)) * std::sin(X(1));
    Y(2) = std::cos(X(1));

    return Y;
  }
};

//-------------------------------------------------------------------------
template <typename T>
class ImplicitSphere : public nyx::Function<T, 3, 1>
{
public:
  Eigen::Matrix<T, 1, 1> operator()(Eigen::Matrix<T, 3, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, 1, 1> Y = Eigen::Matrix<T, 1, 1>::Zero();

    // Implicit surface function of a sphere
    Y(0) = std::sqrt(X(0)*X(0) + X(1)*X(1) + X(2)*X(2));

    return Y;
  }
};

//-------------------------------------------------------------------------
template <typename T, unsigned int N>
class SquareRoot : public nyx::Function<T, N, N>
{
public:
  Eigen::Matrix<T, N, 1> operator()(Eigen::Matrix<T, N, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, N, 1> Y = Eigen::Matrix<T, N, 1>::Zero();
    Y.setZero();

    for (unsigned int k = 0; k < N; ++k)
    {
      Y(k) = std::sqrt(X(k));
    }

    return Y;
  }
};

//-------------------------------------------------------------------------
template <typename T>
class NonLinearFunction : public nyx::Function<T, 3, 3>
{
public:
  Eigen::Matrix<T, 3, 1> operator()(Eigen::Matrix<T, 3, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, 3, 1> Y = Eigen::Matrix<T, 3, 1>::Zero();

    Y(0) = std::pow(X(0) - 12.78, 3.0) * X(1) * X(2); // X(0) = 12.78 is root
    Y(1) = std::pow(X(1) + 23.78, 2.0) * std::exp(X(1) / 1000.0) * std::log(-X(1)); // X(1) = -23.78 is root
    Y(2) = std::pow(X(2) - 3.6, 2) * (std::sin(X(1) / 23.78 * nyx::pi / 2) - 1.0); // X(2) = 3.6 is root

    return Y;
  }
};

/**
* \class EulerAngleSO3Mapping
* \brief Euler angle mapping of the rotational matrix hyper-sphere
*
* Represent the mapping between R^3 and the SO3 manifold
* hypershpere. The matrix rotation is represented by a vector
* of R^n corresponding to its row-major coordinates in the canonical
* M3(R) basis.
*
* Euler angles conventions:
* Roll (rx), Pitch (ry) and Yaw (rz) such as
* For all R in SO3 there exist rx, ry, rz in
* [-pi, pi]x[-pi/2, pi/2]x[-pi, pi] such as
*
* R = Rz(rz) * Ry(ry) * Rx(rx) (i)
*
* with:
* Rx(x) = 1       0       0
*         0  cos(x) -sin(x)
*         0  sin(x)  cos(x)
*
* Ry(x) = cos(x)    0 sin(x)
*              0    1     0
*        -sin(x)    0 cos(x)
*
* Rz(x) = cos(x) -sin(x) 0
*         sin(x)  cos(x) 0
*              0       0 1
*
* This mapping between R^3 and SO3 suffers from one problem:
* as the algebraic topology hairy ball theorem claims all mapping
* between R^n and S^n has singulars points resulting in either
* a discontinuity point or a flat area, i.e a sub-manifold representing
* a lost of one or more degree of freedom.
* Using the Euler angles, this is famously known as the Gambal Lock issue
*
* \author $Author: Pierre Guilbert $
* \version $Revision: 1.0 $
* \date $Date: 02-11-2018 $
* Contact: spguilbert@gmail.com
*/
template <typename T>
class EulerAngleSO3Mapping : public nyx::Function<T, 3, 9>
{
public:
  Eigen::Matrix<T, 9, 1> operator()(Eigen::Matrix<T, 3, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, 9, 1> Y = Eigen::Matrix<T, 9, 1>::Zero();

    double crx = std::cos(X(0)); double srx = std::sin(X(0));
    double cry = std::cos(X(1)); double sry = std::sin(X(1));
    double crz = std::cos(X(2)); double srz = std::sin(X(2));

    // first rotation matrix row
    Y(0) = crz * cry;
    Y(1) = -srz * crx + crz * sry * srx;
    Y(2) = srz * srx + crz * sry * crx;
    // second rotation matrix row
    Y(3) = srz * cry;
    Y(4) = crz * crx + srz * sry * srx;
    Y(5) = -crz * srx + srz * sry * crx;
    // third rotation matrix row
    Y(6) = -sry;
    Y(7) = cry * srx;
    Y(8) = cry * crx;

    return Y;
  }
};

//-------------------------------------------------------------------------
template <typename T>
class MultiVarPolynomial : public nyx::Function<T, 2, 2>
{
public:
  Eigen::Matrix<T, 2, 1> operator()(Eigen::Matrix<T, 2, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, 2, 1> Y = Eigen::Matrix<T, 2, 1>::Zero();

    Y(0) = std::pow(X(0), 3) * std::sqrt(X(1));
    Y(1) = std::pow(X(0), 2) * std::pow(X(1), 3);

    return Y;
  }
};

//-------------------------------------------------------------------------
template <typename T>
class NonLinearFunction2 : public nyx::Function<T, 1, 1>
{
public:
  NonLinearFunction2()
    : nyx::Function<T, 1, 1>()
  {
    this->W.resize(6, 0);
    this->W[0] = 5.0;
    this->W[1] = 0.8;
    this->W[2] = 1.45;
    this->W[3] = 0.45;
    this->W[4] = 8.85;
    this->W[5] = 0.10;
  }

  Eigen::Matrix<T, 1, 1> operator()(Eigen::Matrix<T, 1, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, 1, 1> Y = Eigen::Matrix<T, 1, 1>::Zero();

    // parametric function
    Y(0) = W[0] * std::exp(-std::pow(X(0) - W[1], 2) / (2.0 * W[2])) + W[3] * std::cos(W[4] * X(0)) + W[5] * std::pow(X(0), 2);

    return Y;
  }

  std::vector<T> W;
};

//-------------------------------------------------------------------------
template <typename T>
class NonLinearFunction2Parameters : public nyx::Function<T, 6, 1>
{
public:
  NonLinearFunction2Parameters()
    : nyx::Function<T, 6, 1>()
  {
    this->X.resize(1, 0);
  }

  Eigen::Matrix<T, 1, 1> operator()(Eigen::Matrix<T, 6, 1> W)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, 1, 1> Y = Eigen::Matrix<T, 1, 1>::Zero();

    // parametric function
    Y(0) = W(0) * std::exp(-std::pow(X[0] - W(1), 2) / (2.0 * W(2))) + W(3) * std::cos(W(4) * X[0]) + W(5) * std::pow(X[0], 2);

    return Y;
  }

  std::vector<T> X;
};

//-------------------------------------------------------------------------
template <typename T>
class NonLinearFunction3 : public nyx::Function<T, 2, 2>
{
public:
  Eigen::Matrix<T, 2, 1> operator()(Eigen::Matrix<T, 2, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, 2, 1> Y = Eigen::Matrix<T, 2, 1>::Zero();

    Y(0) = std::pow(X(0) - 4, 2.0) * std::pow(X(1) - 10.0, 2.0); // X(0) = 4 is root
    Y(1) = std::pow(X(0), 2.0) * std::pow(X(1) + 7, 2.0); // X(1) = -7 is root

    return Y;
  }
};

#endif // COMMON_FUNCTIONS_H