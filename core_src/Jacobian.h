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

#ifndef JACOBIAN_H
#define JACOBIAN_H

// STD
#include <iostream>

// Eigen
#include <Eigen/Dense>

namespace nyx
{
  /**
  * \class Jacobian
  * \brief Represente the differential of a function from a real finite
  *        vectorial space to another real finite vectorial space   
  *
  * Let's E and F be two reals finite vectorial spaces and f be
  * a function from E to F. Using the incomplete basis theorem
  * we can create a basis Be of E and a basis Bf of F.
  *
  * We suppose that f is differentiable over E. By definition,
  * it exists a differential operator df such that:
  *
  * for all X in E
  * f(X + dX) = f(X) + df(X)(dX) + o(dX), with df(X) being a linear application
  *
  * Since we are working with finite dimension vectorial space, df(X) can be
  * represented by a matrix Mat(Be, Bf, df(X)) noted the jacobian matrix of
  * f relatively to the basis Be and Bf at the point X. 
  *
  * Here we propose to represent the differential operator of f as a function
  * from R^inDim -> Mat(inDim, outDim)
  *
  * \author $Author: Pierre Guilbert $
  * \version $Revision: 1.0 $
  * \date $Date: 02-11-2018 $
  * Contact: spguilbert@gmail.com
  */
  template <typename T, unsigned int N, unsigned int M>
  class Jacobian
  {
  public:
    /// default constructor of the function
    Jacobian();

    /// Evaluate the function at the point X
    virtual Eigen::Matrix<T, M, N> operator()(Eigen::Matrix<T, N, 1> X)
    {
      // init output and set all values to zero
      Eigen::Matrix<T, M, N> Y = Eigen::Matrix<T, M, N>::Zero();

      return Y;
    }

    /// Get in / out dimensions
    unsigned int GetInDim();
    unsigned int GetOutDim();
  };

  // methods implementation
#include "Jacobian.txx"

} // namespace nyx

#endif // JACOBIAN_H