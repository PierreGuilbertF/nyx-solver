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

#include "NumericalDiff.h"

#ifndef NUMERICAL_DIFF_TXX
#define NUMERICAL_DIFF_TXX

//-------------------------------------------------------------------------
template <typename F, typename T>
NumericalDiff<F, T>::NumericalDiff()
{
  // Update the input / output dimensions
  this->inDim = this->Function.GetInDim();
  this->outDim = this->Function.GetOutDim();
  
  // default method is symmetric quotient
  this->Method = DifferentiationMethod::SymmetricQuotient;
}

//-------------------------------------------------------------------------
template <typename F, typename T>
NumericalDiff<F, T>::NumericalDiff(F argFunction)
{
  this->Function = argFunction;

  // Update the input / output dimensions
  this->inDim = this->Function.GetInDim();
  this->outDim = this->Function.GetOutDim();

  // default method is symmetric quotient
  this->Method = DifferentiationMethod::SymmetricQuotient;
}

//-------------------------------------------------------------------------
template <typename F, typename T>
void NumericalDiff<F, T>::ComputeJacobian(Eigen::Matrix<T, Eigen::Dynamic, 1> X)
{
  // init Jacobian matrix and set coefficients to zero
  this->Jacobian = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(this->outDim, this->inDim);
  this->Jacobian.setZero();

  // check that the input vector length is consistent
  if (X.rows() != this->inDim)
  {
    std::cout << "error in: " << __func__ << " expected vector of dim: "
              << this->inDim << " got dim: " << X.rows() << std::endl;
    return;
  }

  // automatically adapt the step
  this->h = Eigen::Matrix<T, Eigen::Dynamic, 1>(this->inDim);
  const double epsilon = std::sqrt(std::numeric_limits<T>::epsilon());
  for (unsigned int k = 0; k < this->inDim; ++k)
  {
    this->h(k) = epsilon * X(k);
  }

  // Compute the approximation
  for (unsigned int partialDiffIndex = 0; partialDiffIndex < this->inDim; ++partialDiffIndex)
  {
    // directional small step
    Eigen::Matrix<T, Eigen::Dynamic, 1> dX(this->inDim);
    dX.setZero(); dX(partialDiffIndex) = this->h(partialDiffIndex);

    // Compute the partial differentiation vector
    Eigen::Matrix<T, Eigen::Dynamic, 1> partialDiffVector;
    switch (this->Method)
    {
    case DifferentiationMethod::NewtonQuotient:
      partialDiffVector = (this->Function(X + dX) - this->Function(X)) / this->h(partialDiffIndex);
        break;       // and exits the switch
    case DifferentiationMethod::SymmetricQuotient:
      partialDiffVector = (this->Function(X + dX) - this->Function(X - dX)) / (2.0 * this->h(partialDiffIndex));
      break;
    case DifferentiationMethod::SecondOrderQuotient:
      partialDiffVector = -this->Function(X+2.0*dX) + 8.0*this->Function(X+dX) - 8.0*this->Function(X-dX) + this->Function(X-2.0*dX);
      partialDiffVector /= 12.0 * this->h(partialDiffIndex);
      break;
    }

    this->Jacobian.col(partialDiffIndex) = partialDiffVector;
  }
}

//-------------------------------------------------------------------------
template <typename F, typename T>
void NumericalDiff<F, T>::SetDifferentiationMethod(DifferentiationMethod method)
{
  this->Method = method;
}

#endif // NUMERICAL_DIFF_TXX