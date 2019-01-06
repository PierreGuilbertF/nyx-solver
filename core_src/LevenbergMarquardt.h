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

#ifndef LEVENBERG_MARQUARDT_H
#define LEVENBERG_MARQUARDT_H

// STD
#include <numeric>
#include <iostream>
#include <cmath>
#include <vector>

// Eigen
#include <Eigen/Dense>

namespace nyx
{
  /**
  * \class LevenbergMarquardt
  * \brief Solve non-linear least-square function minimization
  *
  * Iterative method to solve non-linear least-square function
  * minimization using the Levenberg-Marquardt algorithm
  *
  * Let's {Fi} be a familly of functions Fi: U -> R, U an open of R^n.
  * The goal is to find W in R^n such as:
  *
  * W = argmin[S(W)} (i)
  * 
  * with:
  *
  * S(W) = ||F0(W)||^2 + ... + ||Fn(W)||^2
  *
  * Usually, the functions Fi are defined from a input dataset
  * we want to fit using a parametric function. Let's define the
  * set {Yi, Zi} with Yi in R^n1 and Zi in R^n2. We note F
  * F: R^n1 x R^n -> R^n2
  *    (Y, W)     -> F(Y, W)
  *
  * From F, we define Fi as being:
  * Fi(W) = F(Yi, W) - Zi
  *
  * The idea of the method is to solve this non-linear least square
  * minimization problem iteratively by approximating the Fi functions
  * by their first order tailor expansion. This corresponds to approximate
  * the R^n2 embedded manifold { Fi(W), W in R^n} by its local tangent hyperplane
  *
  * Fi(W + dW) ~ Fi(W) + JFi(W) * dW
  *
  * The sum S has its minimum at a zero gradient with respect to W
  *
  * S(W + dW) ~ sum[||Fi(W) + JFi(W) * dW||^2]
  *           = [Fi(W) + JFi(W) * dW]' * [Fi(W) + JFi(W) * dW]
  *           = Fi(W)'*Fi(W) - 2*Fi(W)'*JFi(W)*dW + dW'*JFi(W)'*JFi(W)*dW
  *
  * Taking the derivative of S(W + dW) with respect to dW and setting the result to zero:
  *
  * JFi(W)'*JFi(W)*dW = JFi(W) * Fi(W) (iii)
  *
  * The equation (iii) is the one used in the Gauss-Newton algorithm. Solving
  * (iii) provides the step dW to update the weights of the parametric function
  * in order to minimize S. In the levenberg-marquardt algorithm, a diagonal
  * loading is add in order to improve the illness of the problem:
  *
  * [J'*J + lambda*diag(J'*J)] * dW = J' * F(W)
  *
  * The non negative damping factor lambda is adjusted at each iteration. If reduction of S
  * is rapid, a smaller value can be used bringing the algorithm closer to the Gauss-Newton
  * algorithm. A larger value is closer to the gradient-descent algorithm.
  *
  * \author $Author: Pierre Guilbert $
  * \version $Revision: 1.0 $
  * \date $Date: 02-11-2018 $
  * Contact: spguilbert@gmail.com
  */
  template <typename F, typename J, typename T>
  class LevenbergMarquardt
  {
  public:
    /// Default constructor
    LevenbergMarquardt();

  protected:
    /// The residual functions Fi
    std::vector<F> residualsFunctions;

    /// weights vector
    Eigen::Matrix<T, Eigen::Dynamic, 1> W;
  };

// methods implementation
#include "NumericalDiff.txx"

} // namespace nyx

#endif // LEVENBERG_MARQUARDT_H
