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
#include "main_nyx_test.h"
#include "Function_test.h"
#include "Jacobian_test.h"
#include "NumericalDiff_test.h"
#include "NewtonRaphson_test.h"

// STD
#include <iostream>

// Eigen
#include <Eigen/Dense>

int main(int argc, char *argv[])
{
  int nbrErr = 0;

  // Function tests
  nbrErr += TestFunction();

  // Jacobian tests
  nbrErr += TestJacobian();

  // Numerical Differentiation tests
  nbrErr += NumericalDiffSquareRoot();
  nbrErr += NumericalDiffEulerAngleMapping();
  nbrErr += NumericalDiffMethods();

  // Newton Raphson methos tests
  nbrErr += TestNewtonRaphsonMethod();

  if (nbrErr == 0)
  {
    std::cout << __func__ << " SUCCEEDED" << std::endl;
    return EXIT_SUCCESS;
  }
  else
  {
    std::cout << __func__ << " FAILED, nbrErr: " << nbrErr << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}