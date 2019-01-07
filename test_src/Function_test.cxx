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
#include "Function_test.h"
#include "CommonFunctions.h"
#include "CommonJacobians.h"
#include "Tools.h"

// STD
#include <cmath>
#include <vector>

//-------------------------------------------------------------------------
unsigned int TestFunction()
{
  unsigned int nbrErr = 0;

  // instanciate sphere mapping function
  ParametricSphere<double> sphere;

  // check dimensions
  if ((sphere.GetInDim() != 2) || (sphere.GetOutDim() != 3))
    nbrErr++;

  // check some specific points
  std::vector<Eigen::Matrix<double, 3, 1> > controlsPoints;
  Eigen::Matrix<double, 3, 1> ctrlPt;
  ctrlPt << 0, 0, 1.0; controlsPoints.push_back(ctrlPt); // (0, 0)
  ctrlPt << 0, 0, 1.0; controlsPoints.push_back(ctrlPt); // (theta, 0)
  ctrlPt << 1, 0, 0; controlsPoints.push_back(ctrlPt); // (0, pi / 2)
  ctrlPt << std::sqrt(2.0) / 2.0, std::sqrt(2.0) / 2.0, 0; // (pi / 4, pi / 2)
  controlsPoints.push_back(ctrlPt);
  ctrlPt << 1.0 / std::sqrt(3), 1.0 / std::sqrt(3), 1.0 / std::sqrt(3); // (pi / 4, pi / 4)
  controlsPoints.push_back(ctrlPt);

  std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1> > testsPoints;
  Eigen::Matrix<double, 2, 1> X;
  X << 0, 0; testsPoints.push_back(sphere(X));
  X << 1.0, 0; testsPoints.push_back(sphere(X));
  X << 0, nyx::pi / 2.0; testsPoints.push_back(sphere(X));
  X << nyx::pi / 4.0, nyx::pi / 2.0; testsPoints.push_back(sphere(X));
  X << nyx::pi / 4.0, std::atan(std::sqrt(2) / 1.0); testsPoints.push_back(sphere(X));

  for (unsigned int k = 0; k < testsPoints.size(); ++k)
  {
    if (!nyx::IsEqual((testsPoints[k] - controlsPoints[k]).norm(), 0.0))
      nbrErr++;
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