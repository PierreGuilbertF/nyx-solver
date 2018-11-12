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

// STD
#include <fstream>
#include <iostream>

// LOCAL
#include "LevenbergMarquardt_test.h"
#include "LevenbergMarquardt.h"
#include "CommonFunctions.h"
#include "Tools.h"

//-------------------------------------------------------------------------
int TestLevenbergMarquardtCosineFunction()
{
  unsigned int nbrErr = 0;

  // file to export samples
  std::ofstream file;
  file.open("samplesData.csv");
  if (!file.is_open())
  {
    return 1; // return with error
  }
  file << "X, F(X)," << std::endl;

  // Non linear function to create the samples data
  NonLinearFunction2<double> F;

  // Create samples data
  unsigned int Nsamples = 500;
  double xmin = -5.0;
  double xmax = 5.0;
  double dx = (xmax - xmin) / static_cast<double>(Nsamples);
  std::vector<Eigen::Matrix<double, 1, 1> > X, Y;
  for (unsigned int k = 0; k < Nsamples; ++k)
  {
    Eigen::Matrix<double, 1, 1> x;
    x(0) = xmin + static_cast<double>(k) * dx;
    Eigen::Matrix<double, 1, 1> y = F(x);

    X.push_back(x);
    Y.push_back(y);
  }

  for (unsigned int k = 0; k < X.size(); ++k)
  {
    file << X[k] << "," << Y[k] << "," << std::endl;
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