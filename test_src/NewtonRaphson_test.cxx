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

// LOCAL
#include "NewtonRaphson_test.h"
#include "Function.h"
#include "NumericalDiff.h"
#include "NewtonRaphson.h"
#include "Tools.h"

// STD
#include <iostream>

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

    Y(0) = std::pow(X(0) - 12.78, 3.0) * std::sqrt(X(0)) * std::cos(X(2) / 3.6 * nyx::pi / 2.0); // X(0) = 12.78 is root
    Y(1) = std::pow(X(1) + 23.78, 2.0) * std::exp(X(1) / 1000.0) * std::log(-X(1)); // X(1) = -23.78 is root
    Y(2) = std::pow(X(2) - 3.6, 2) * (std::sin(X(1) / 23.78 * nyx::pi / 2) - 1.0); // X(2) = 64.67 is root

    return Y;
  }

protected:

};

//-------------------------------------------------------------------------
int TestNewtonRaphsonMethod()
{
  unsigned int nbrErr = 0;

  // Function we want to know the root
  NonLinearFunction<double> F(3, 3);

  // Jacobian of the function, here
  // we use a numerical differenciation jacobian
  nyx::NumericalDiff<NonLinearFunction<double>, double> J(F);

  // Now, instanciate the Newton Raphson solver
  // using the numerical differenciation jacobian
  nyx::NewtonRaphson<NonLinearFunction<double>,
                     nyx::NumericalDiff<NonLinearFunction<double>, double>,
                     double> NewtonRaphsonSolver(F, J);

  // Solve F(X) = Y
  Eigen::Matrix<double, 3, 1> X0, Y;
  Y << 0, 0, 0;
  X0 << 8, -45, 2.76;
  Eigen::Matrix<double, 3, 1> Xestimated = NewtonRaphsonSolver(Y, X0);

  // Expected root
  Eigen::Matrix<double, 3, 1> Xs;
  Xs << 12.78, -23.78, 64.67;

  std::cout << "Xestimated: " << Xestimated.transpose() << std::endl;

  if (nbrErr == 0)
  {
    std::cout << "Test: " << __func__ << " SUCCEEDED" << std::endl;
  }
  else
  {
    std::cout << "Test: " << __func__ << " FAILED" << std::endl;
  }
  return nbrErr;
}