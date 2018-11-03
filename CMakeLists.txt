#=========================================================================
# Nyx Solver - Non Linear Problems Solver
#
# Copyright 2018 Pierre Guilbert
# Author: Pierre Guilbert (spguilbert@gmail.com)
# Data: 02-11-2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#=========================================================================

project(NyxSolverTests)
cmake_minimum_required(VERSION 3.1)

# set path to the nyx tests / core sources folder
set(nyx_tests_folder ${CMAKE_CURRENT_SOURCE_DIR}/test_src)
set(nyx_core_folder ${CMAKE_CURRENT_SOURCE_DIR}/core_src)

# set the nyx tests sources required to build the tests
set(nyx_tests_sources ${nyx_tests_folder}/main_nyx_test.cxx
					  ${nyx_tests_folder}/Function_test.cxx
                      ${nyx_tests_folder}/NumericalDiff_test.cxx
					  ${nyx_core_folder}/Function.txx
                      ${nyx_core_folder}/NumericalDiff.txx)

# required include directories
set(nyx_core_include_dir ${CMAKE_CURRENT_SOURCE_DIR}/core_src)
set(nyx_tests_include_dir ${CMAKE_CURRENT_SOURCE_DIR}/test_src)

add_executable(nyx_tests ${nyx_tests_sources})
target_include_directories(nyx_tests PRIVATE ${nyx_core_include_dir})
target_include_directories(nyx_tests PRIVATE ${nyx_tests_include_dir})

# dependencies management
# === Eigen ===
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)
target_include_directories(nyx_tests PRIVATE ${EIGEN3_INCLUDE_DIR})