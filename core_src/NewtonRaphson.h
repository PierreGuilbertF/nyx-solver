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

#ifndef NEWTON_RAPHSON_H
#define NEWTON_RAPHSON_H

// STD
#include <numeric>
#include <iostream>
#include <cmath>

// Eigen
#include <Eigen/Dense>

namespace nyx
{
  /**
  * \class NewtonRaphson
  * \brief Solve non-linear equations system using Newton Raphson method
  *
  * Iterative method to solve non-linear equations system using
  * the Newton Raphson method
  *
  * Let's F be a function F: U -> R^n, U an open of R^n.
  * The goal is to find X in R^n such as:
  *
  * F(X) = 0 (i)
  *
  * The idea of the method is to solve this non-linear equation
  * system iteratively by approximating the F function by its
  * first order taylor expansion. This correspod to approximate
  * the R^n embedded manifold { F(X), X in R^n} by its local tangent
  * hyperplane.
  * Let's n in N the current iteration and Xn the current approximation
  * in R^n
  * 
  * F(X) ~ F(Xn) + JF(Xn)(X - Xn) (ii)
  *
  * We defines Xn++ such as F(Xn++) = 0. Replacing in (ii):
  *
  * 0 = F(Xn) + JF(Xn)(Xn++ - Xn) (iii)
  * => Xn++ = Xn - JF(Xn)^(-1) * F(Xn)
  *
  * We then iterate until convergence (if there is convergence).
  *
  * Study of convergence:
  *
  * \author $Author: Pierre Guilbert $
  * \version $Revision: 1.0 $
  * \date $Date: 02-11-2018 $
  * Contact: spguilbert@gmail.com
  */

  template <typename F, typename J, typename T>
  class NewtonRaphson
  {
  public:
    /// Constructor
    NewtonRaphson(F argFunc, J argJaco);

    /// Solve the equation F(X) = Y
    Eigen::Matrix<T, Eigen::Dynamic, 1> operator()(Eigen::Matrix<T, Eigen::Dynamic, 1> Y,
                                                   Eigen::Matrix<T, Eigen::Dynamic, 1> X0)
    {
      return this->SolveEquation(Y, X0);
    }
  protected:
    /// Maximum iteration of Newton Raphson method
    unsigned int MaxIteration;
    unsigned int NbrIterationMade;

    /// Function that represents the non
    /// linear equation system: F(X) = Y
    F Function;

    /// Jacobian of the function F
    J Jacobian;

    /// Solve the equation F(X) = Y
    Eigen::Matrix<T, Eigen::Dynamic, 1> SolveEquation(Eigen::Matrix<T, Eigen::Dynamic, 1> Y,
                                                      Eigen::Matrix<T, Eigen::Dynamic, 1> X0);

    /// Indicate if the algortihm should continue
    bool ShouldIterate();
  };

// methods implementation
#include "NewtonRaphson.txx"
} // namespace nyx

#endif // NEWTON_RAPHSON_H