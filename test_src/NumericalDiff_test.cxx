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
#include "CommonFunctions.h"
#include "CommonJacobians.h"
#include "Tools.h"

// STD
#include <iostream>

//-------------------------------------------------------------------------
int NumericalDiffSquareRoot()
{
  unsigned int nbrErr = 0;

  SquareRoot<double, 3> vectSqrt;
  nyx::NumericalDiff<SquareRoot<double, 3>, double> diff(vectSqrt);

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

//-------------------------------------------------------------------------
int NumericalDiffEulerAngleMapping()
{
  unsigned int nbrErr = 0;

  EulerAngleSO3Mapping<double> eulMapping;
  nyx::NumericalDiff<EulerAngleSO3Mapping<double>, double> numJacobian(eulMapping);
  EulerAngleSO3MappingJacobian<double> analyticjacobian;

  // test random samples with fixed seed
  unsigned int nbrSample = 25;
  std::srand(1992);

  for (unsigned int k = 0; k < nbrSample; ++k)
  {
    Eigen::Matrix<double, 3, 1> X;
    X(0) = 2.0 * nyx::pi * (static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX) - 0.5);
    X(1) = 2.0 * nyx::pi * (static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX) - 0.5);
    X(2) = 2.0 * nyx::pi * (static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX) - 0.5);

    Eigen::Matrix<double, 9, 3> J1 = numJacobian(X);
    Eigen::Matrix<double, 9, 3> J2 = analyticjacobian(X);
    for (unsigned int i = 0; i < 9; ++i)
    {
      for (unsigned int j = 0; j < 3; ++j)
      {
        if (!nyx::IsEqual(J1(i, j), J2(i, j), 5e-7))
          nbrErr++;
      }
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

//-------------------------------------------------------------------------
int NumericalDiffNonLinearFunction()
{
  unsigned int nbrErr = 0;

  Eigen::Matrix<double, 1, 1> X;
  X(0) = 0.78;
  Eigen::Matrix<double,6 , 1> W;
  W << 5.0, 0.8, 1.45, 0.45, 8.85, 0.10;
  NonLinearFunction2Parameters<double> f;
  f.X[0] = X(0);
  nyx::NumericalDiff<NonLinearFunction2Parameters<double>, double> J1(f);
  NonLinearFunction2ParametersJacobian<double> J2;
  J2.X[0] = X(0);

  //
  Eigen::MatrixXd diffJ = J1(W) - J2(W);

  for (unsigned int i = 0; i < 2; ++i)
  {
    for (unsigned int j = 0; j < 2; ++j)
    {
      if (!nyx::IsEqual(diffJ(i, j), 0.0, 1e-7))
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

//-------------------------------------------------------------------------
int NumericalDiffMethods()
{
  unsigned int nbrErr = 0;

  MultiVarPolynomial<double> f;
  nyx::NumericalDiff<MultiVarPolynomial<double>, double> J1;
  MultiVarPolynomialJacobian<double> J2;

  Eigen::Matrix<double, 2, 1> X;
  X << -23.4, 11.23;

  // Faster, less accurate
  J1.SetDifferentiationMethod(nyx::DifferentiationMethod::NewtonQuotient);
  Eigen::MatrixXd diffJ1 = J1(X) - J2(X);
  // average, average accurate
  J1.SetDifferentiationMethod(nyx::DifferentiationMethod::SymmetricQuotient);
  Eigen::MatrixXd diffJ2 = J1(X) - J2(X);
  // Slower, more accurate
  J1.SetDifferentiationMethod(nyx::DifferentiationMethod::SecondOrderQuotient);
  Eigen::MatrixXd diffJ3 = J1(X) - J2(X);

  for (unsigned int i = 0; i < 2; ++i)
  {
    for (unsigned int j = 0; j < 2; ++j)
    {
      if (!nyx::IsEqual(diffJ1(i, j), 0.0, 1e-2))
        nbrErr++;
      if (!nyx::IsEqual(diffJ2(i, j), 0.0, 1e-3))
        nbrErr++;
      if (!nyx::IsEqual(diffJ3(i, j), 0.0, 1e-3))
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