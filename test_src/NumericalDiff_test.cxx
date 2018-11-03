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
#include "NumericalDiff_test.h"
#include "Function.h"
#include "NumericalDiff.h"
#include "Tools.h"

// STD
#include <iostream>

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
int AutomaticStepDiffTest()
{
  unsigned int nbrErr = 0;

  SquareRoot<double> vectSqrt(3, 3);
  nyx::NumericalDiff<SquareRoot<double>, double> diff(vectSqrt);

  Eigen::Matrix<double, 3, 1> X;
  X << 10.0, 101.67, 1254.980;
  Eigen::Matrix<double, 3, 3> J = diff(X);
  
  Eigen::Matrix<double, 3, 3> realJ = Eigen::Matrix<double, 3, 3>::Zero();
  realJ(0, 0) = 1.0 / (2.0 * std::sqrt(X(0)));
  realJ(1, 1) = 1.0 / (2.0 * std::sqrt(X(1)));
  realJ(2, 2) = 1.0 / (2.0 * std::sqrt(X(2)));

  for (unsigned int i = 0; i < 3; ++i)
  {
    for (unsigned int j = 0; j < 3; ++j)
    {
      if (!nyx::IsEqual(J(i, j), realJ(i, j), 1e-8))
        nbrErr++;
    }
  }

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