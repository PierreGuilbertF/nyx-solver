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

// LOCAL
#include "Function.h"
#include "Jacobian.h"
#include "Tools.h"

//-------------------------------------------------------------------------
template <typename T>
class ParametricSphere : public nyx::Function<T>
{
public:
  ParametricSphere(unsigned int inDim, unsigned int outDim)
    : nyx::Function<T>(inDim, outDim)
  {
    //
  }

  Eigen::Matrix<T, Eigen::Dynamic, 1> operator()(Eigen::Matrix<T, Eigen::Dynamic, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, Eigen::Dynamic, 1> Y(this->outDim, 1);
    Y.setZero();

    // Check dimensions consistency
    if (X.rows() != this->inDim)
    {
      std::cout << "error in: " << __func__ << " expected vector of dim: "
        << this->inDim << " got dim: " << X.rows() << std::endl;
      return Y;
    }

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
class ParametricSphereJacobian : public nyx::Jacobian<T>
{
public:
  ParametricSphereJacobian(unsigned int inDim, unsigned int outDim)
        : nyx::Jacobian<T>(inDim, outDim)
  {
      //
  }

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> operator()(Eigen::Matrix<T, Eigen::Dynamic, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Y(this->outDim, this->inDim);
    Y.setZero();

    // Check dimensions consistency
    if (X.rows() != this->inDim)
    {
      std::cout << "error in: " << __func__ << " expected vector of dim: "
          << this->inDim << " got dim: " << X.rows() << std::endl;
      return Y;
    }

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
class ImplicitSphere : public nyx::Function<T>
{
public:
  ImplicitSphere(unsigned int inDim, unsigned int outDim)
    : nyx::Function<T>(inDim, outDim)
  {
    //
  }

  Eigen::Matrix<T, Eigen::Dynamic, 1> operator()(Eigen::Matrix<T, Eigen::Dynamic, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, Eigen::Dynamic, 1> Y(this->outDim, 1);
    Y.setZero();

    // Check dimensions consistency
    if (X.rows() != this->inDim)
    {
      std::cout << "error in: " << __func__ << " expected vector of dim: "
        << this->inDim << " got dim: " << X.rows() << std::endl;
      return Y;
    }

    // Implicit surface function of a sphere
    Y(0) = std::sqrt(X(0)*X(0) + X(1)*X(1) + X(2)*X(2));

    return Y;
  }
};

//-------------------------------------------------------------------------
template <typename T>
class ImplicitSphereJacobian : public nyx::Jacobian<T>
{
public:
  ImplicitSphereJacobian(unsigned int inDim, unsigned int outDim)
    : nyx::Jacobian<T>(inDim, outDim)
  {
    //
  }

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> operator()(Eigen::Matrix<T, Eigen::Dynamic, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Y(this->outDim, this->inDim);
    Y.setZero();

    // Check dimensions consistency
    if (X.rows() != this->inDim)
    {
      std::cout << "error in: " << __func__ << " expected vector of dim: "
        << this->inDim << " got dim: " << X.rows() << std::endl;
      return Y;
    }

    // Implicit surface function differential of a sphere
    Y(0, 0) = X(0) / std::sqrt(X(0)*X(0) + X(1)*X(1) + X(2)*X(2)); // DY0 / DX0
    Y(0, 1) = X(1) / std::sqrt(X(0)*X(0) + X(1)*X(1) + X(2)*X(2)); // DY0 / DX1
    Y(0, 2) = X(2) / std::sqrt(X(0)*X(0) + X(1)*X(1) + X(2)*X(2)); // DY0 / DX2

    return Y;
  }
};

//-------------------------------------------------------------------------
template <typename T>
class SquareRoot : public nyx::Function<T>
{
public:
  SquareRoot()
    : nyx::Function<T>()
  {
    //
  }

  SquareRoot(unsigned int inDim, unsigned int outDim)
    : nyx::Function<T>(inDim, outDim)
  {
    //
  }

  Eigen::Matrix<T, Eigen::Dynamic, 1> operator()(Eigen::Matrix<T, Eigen::Dynamic, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, Eigen::Dynamic, 1> Y(this->outDim, 1);
    Y.setZero();

    // Check dimensions consistency
    if (X.rows() != this->inDim)
    {
      std::cout << "error in: " << __func__ << " expected vector of dim: "
        << this->inDim << " got dim: " << X.rows() << std::endl;
      return Y;
    }

    if (this->inDim != this->outDim)
    {
      std::cout << "error in: " << __func__ << " input and output dimension should match" << std::endl;
      return Y;
    }

    for (unsigned int k = 0; k < this->outDim; ++k)
    {
      Y(k) = std::sqrt(X(k));
    }

    return Y;
  }

protected:

};

//-------------------------------------------------------------------------
template <typename T>
class NonLinearFunction : public nyx::Function<T>
{
public:
  NonLinearFunction()
    : nyx::Function<T>()
  {
    //
  }

  NonLinearFunction(unsigned int inDim, unsigned int outDim)
    : nyx::Function<T>(inDim, outDim)
  {
    //
  }

  Eigen::Matrix<T, Eigen::Dynamic, 1> operator()(Eigen::Matrix<T, Eigen::Dynamic, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, Eigen::Dynamic, 1> Y(this->outDim, 1);
    Y.setZero();

    // Check dimensions consistency
    if (X.rows() != this->inDim)
    {
      std::cout << "error in: " << __func__ << " expected vector of dim: "
        << this->inDim << " got dim: " << X.rows() << std::endl;
      return Y;
    }

    if (this->inDim != this->outDim)
    {
      std::cout << "error in: " << __func__ << " input and output dimension should match" << std::endl;
      return Y;
    }

    Y(0) = std::pow(X(0) - 12.78, 3.0) * X(1) * X(2); // X(0) = 12.78 is root
    Y(1) = std::pow(X(1) + 23.78, 2.0) * std::exp(X(1) / 1000.0) * std::log(-X(1)); // X(1) = -23.78 is root
    Y(2) = std::pow(X(2) - 3.6, 2) * (std::sin(X(1) / 23.78 * nyx::pi / 2) - 1.0); // X(2) = 3.6 is root

    return Y;
  }

protected:

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
class EulerAngleSO3Mapping : public nyx::Function<T>
{
public:
  EulerAngleSO3Mapping()
    : nyx::Function<T>(3, 9)
  {
    //
  }

  EulerAngleSO3Mapping(unsigned int inDim, unsigned int outDim)
    : nyx::Function<T>(inDim, outDim)
  {
    //
  }

  Eigen::Matrix<T, Eigen::Dynamic, 1> operator()(Eigen::Matrix<T, Eigen::Dynamic, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, Eigen::Dynamic, 1> Y(this->outDim, 1);
    Y.setZero();

    // Check dimensions consistency
    if (X.rows() != this->inDim)
    {
      std::cout << "error in: " << __func__ << " expected vector of dim: "
        << this->inDim << " got dim: " << X.rows() << std::endl;
      return Y;
    }

    if ((this->inDim != 3) || (this->outDim != 9))
    {
      std::cout << "error in: " << __func__ << " inDim should be 3 and outDim should be 9" << std::endl;
      return Y;
    }

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

protected:

};

template <typename T>
class EulerAngleSO3MappingJacobian : public nyx::Function<T>
{
public:
  EulerAngleSO3MappingJacobian()
    : nyx::Function<T>(3, 9)
  {
    //
  }

  EulerAngleSO3MappingJacobian(unsigned int inDim, unsigned int outDim)
    : nyx::Function<T>(inDim, outDim)
  {
    //
  }

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> operator()(Eigen::Matrix<T, Eigen::Dynamic, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Y(this->outDim, this->inDim);
    Y.setZero();

    // Check dimensions consistency
    if (X.rows() != this->inDim)
    {
      std::cout << "error in: " << __func__ << " expected vector of dim: "
        << this->inDim << " got dim: " << X.rows() << std::endl;
      return Y;
    }

    if ((this->inDim != 3))
    {
      std::cout << "error in: " << __func__ << " inDim should be 3" << std::endl;
      return Y;
    }

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

protected:

};

//-------------------------------------------------------------------------
template <typename T>
class MultiVarPolynomial : public nyx::Function<T>
{
public:
  MultiVarPolynomial()
    : nyx::Function<T>(2, 2)
  {
    //
  }

  MultiVarPolynomial(unsigned int inDim, unsigned int outDim)
    : nyx::Function<T>(inDim, outDim)
  {
    //
  }

  Eigen::Matrix<T, Eigen::Dynamic, 1> operator()(Eigen::Matrix<T, Eigen::Dynamic, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, Eigen::Dynamic, 1> Y(this->outDim, 1);
    Y.setZero();

    // Check dimensions consistency
    if (X.rows() != this->inDim)
    {
      std::cout << "error in: " << __func__ << " expected vector of dim: "
        << this->inDim << " got dim: " << X.rows() << std::endl;
      return Y;
    }

    if (this->inDim != this->outDim)
    {
      std::cout << "error in: " << __func__ << " input and output dimension should match" << std::endl;
      return Y;
    }

    Y(0) = std::pow(X(0), 3) * std::sqrt(X(1));
    Y(1) = std::pow(X(0), 2) * std::pow(X(1), 3);

    return Y;
  }

protected:

};

//-------------------------------------------------------------------------
template <typename T>
class MultiVarPolynomialJacobian : public nyx::Function<T>
{
public:
  MultiVarPolynomialJacobian()
    : nyx::Function<T>(2, 2)
  {
    //
  }

  MultiVarPolynomialJacobian(unsigned int inDim, unsigned int outDim)
    : nyx::Function<T>(inDim, outDim)
  {
    //
  }

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> operator()(Eigen::Matrix<T, Eigen::Dynamic, 1> X)
  {
    // init output and set all values to zero
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Y(this->outDim, this->inDim);
    Y.setZero();

    // Check dimensions consistency
    if (X.rows() != this->inDim)
    {
      std::cout << "error in: " << __func__ << " expected vector of dim: "
        << this->inDim << " got dim: " << X.rows() << std::endl;
      return Y;
    }

    if (this->inDim != this->outDim)
    {
      std::cout << "error in: " << __func__ << " input and output dimension should match" << std::endl;
      return Y;
    }

    Y(0, 0) = 3.0 * std::pow(X(0), 2) * std::sqrt(X(1)); // dY1 / dX1
    Y(0, 1) = std::pow(X(0), 3) / (2.0 * std::sqrt(X(1))); // dY1 / dX2
    Y(1, 0) = 2.0 * X(0) * std::pow(X(1), 3); // dY2 / dX1
    Y(1, 1) = std::pow(X(0), 2) * 3.0 * std::pow(X(1), 2); // dY2 / dX2

    return Y;
  }

protected:

};

#endif // COMMON_FUNCTIONS_H