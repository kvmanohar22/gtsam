/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file EigenOptimizer.h
 *
 * @brief optimizer linear factor graph using eigen solver as backend
 *
 * @date Aug 2019
 * @author Mandy Xie
 */
#pragma once

#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/linear/LinearSolver.h>
#include <Eigen/Sparse>
#include <string>

namespace gtsam {

class SparseEigenSolver : public gtsam::LinearSolver {
  public:

    typedef enum {
      QR,
      CHOLESKY
    } SparseEigenSolverType;


    explicit SparseEigenSolver(SparseEigenSolverType type);

    bool isIterative() override;

    bool isSequential() override;

    VectorValues solve(const GaussianFactorGraph &gfg, const Ordering &ordering) override;

  protected:

    SparseEigenSolverType solverType = QR;

  };
}  // namespace gtsam
