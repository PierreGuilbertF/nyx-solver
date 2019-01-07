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
#include "Jacobian_test.h"
#include "CommonFunctions.h"
#include "CommonJacobians.h"
#include "Tools.h"

// STD
#include <cmath>
#include <vector>

//-------------------------------------------------------------------------
unsigned int TestJacobian()
{
  unsigned int nbrErr = 0;

  // in this test we propose to compute the normal of the 2-sphere
  // of radius 1 using 2 methods:
  // - First by using an explicit parametrization of the 2-sphere, computing its
  //   tangent plane and taking the normal of the tangent plane
  // - Secondly, by using an implicit parametrization of the 2-sphere and computing
  //   the gradient of the potential function at the desired point.

  // instanciate sphere mapping function
  ParametricSphere<double> explicitSphere;
  ParametricSphereJacobian<double> explicitSphereJ;
  ImplicitSphereJacobian<double> implicitSphereJ;

  // Points to check
  std::vector<Eigen::Matrix<double, 2, 1> > points;
  points.push_back(Eigen::Matrix<double, 2, 1>(0.456, 1.123));
  points.push_back(Eigen::Matrix<double, 2, 1>(-0.3672, 0.4246));
  points.push_back(Eigen::Matrix<double, 2, 1>(1.2356, -0.1462));
  points.push_back(Eigen::Matrix<double, 2, 1>(-1.872, -0.897));

  for (unsigned int k = 0; k < points.size(); ++k)
  {
    // parametric sphere jacobian calculous
    Eigen::Matrix<double, 3, 1> X = explicitSphere(points[k]);
    Eigen::Matrix<double, 3, 2> J = explicitSphereJ(points[k]);

    // implicit sphere jacobian calculous
    Eigen::Matrix<double, 1, 3> Jexpl = implicitSphereJ(X);

    Eigen::Matrix<double, 3, 1> n1 = (Jexpl / Jexpl.norm()).transpose();
    Eigen::Matrix<double, 3, 1> n2 = J.col(0).cross(J.col(1)) / (J.col(0).cross(J.col(1))).norm();
    double s = std::abs(n1.dot(n2));

    if (!nyx::IsEqual(1.0, s, 1e-8))
    {
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