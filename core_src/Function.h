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

#ifndef FUNCTION_H
#define FUNCTION_H

// STD
#include <iostream>

// Eigen
#include <Eigen/Dense>

namespace nyx
{
  /**
  * \class Function
  * \brief Represente a function from a real finite vectorial space
  *        to another real finite vectorial space
  *
  * Let's E and F be two reals finite vectorial spaces and f be
  * a function from E to F. Using the incomplete basis theorem
  * we can create a basis Be of E and a basis Bf of F.
  *
  * Given these two basis, it exists canonicals vectorial
  * space isomorphisms from E and R^dim(E) and F and R^dim(F):
  * IsoE : E -> R^dim(E), x -> Mat(Be, x)
  * IsoF:  F -> R^dim(F), x -> Mat(Bf, x)
  *
  * it also exists a canonical bijection between F(E -> F) and
  * F(R^dim(E) -> R^dim(F)):
  * let x belong to E and X being the coordinate representation
  * of x in the Be basis
  * let y = f(x) belong to F and Y = F(X) being the coordinate representation
  * of y in the Bf basis. hence, we have:
  *
  * f(x) = (IsoF^(-1) o F o IsoE)(x) (1)
  * and then,
  * F = IsoF o f o IsoE^(-1)
  *
  * Here we propose to represent f by its isomorphic correspondant F
  * But the user should not forget that F is just a representation using
  * an ismorphism of the real function f.
  *
  *
  * \author $Author: Pierre Guilbert $
  * \version $Revision: 1.0 $
  * \date $Date: 02-11-2018 $
  * Contact: spguilbert@gmail.com
  */
  template <typename T, unsigned int N, unsigned int M>
  class Function
  {
  public:
    /// default constructor of the function
    Function();

    /// Evaluate the function at the point X
    virtual Eigen::Matrix<T, M, 1> operator()(Eigen::Matrix<T, N, 1> X)
    {
      // init output and set all values to zero
      Eigen::Matrix<T, M, 1> Y = Eigen::Matrix<T, M, 1>::Zero();
      Y.setZero();
      return Y;
    }

    /// Get in / out dimensions
    unsigned int GetInDim();
    unsigned int GetOutDim();
  };

  // methods implementation
#include "Function.txx"

} // namespace nyx
#endif // FUNCTION_H