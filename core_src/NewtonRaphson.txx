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

#include "NewtonRaphson.h"

#ifndef NEWTON_RAPHSON_TXX
#define NEWTON_RAPHSON_TXX

//-------------------------------------------------------------------------
template <typename F, typename J, typename T>
NewtonRaphson<F, J, T>::NewtonRaphson(F argFunc, J argJaco)
{
  this->MaxIteration = 25;
  this->NbrIterationMade = 0;
  this->Function = argFunc;
  this->Jacobian = argJaco;
}

//-------------------------------------------------------------------------
template <typename F, typename J, typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> NewtonRaphson<F, J, T>::SolveEquation(Eigen::Matrix<T, Eigen::Dynamic, 1> Y,
                                                                          Eigen::Matrix<T, Eigen::Dynamic, 1> X0)
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> X = X0;
  while (this->ShouldIterate())
  {
    std::cout << "X: " << X.transpose() << std::endl;
    std::cout << "Jacobian: " << this->Jacobian(X) << std::endl;
    std::cout << "Function: " << this->Function(X) << std::endl;
    X = X - this->Jacobian(X).inverse() * (this->Function(X) - Y);
  }

  return X;
}

//-------------------------------------------------------------------------
template <typename F, typename J, typename T>
bool NewtonRaphson<F, J, T>::ShouldIterate()
{
  this->NbrIterationMade++;
  return this->NbrIterationMade <= this->MaxIteration;
}

#endif // NEWTON_RAPHSON_TXX