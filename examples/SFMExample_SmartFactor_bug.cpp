/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    SFMExample_SmartFactor.cpp
 * @brief   A structure-from-motion problem on a simulated dataset, using smart projection factor
 * @author  Duy-Nguyen Ta
 * @author  Jing Dong
 * @author  Frank Dellaert
 */

// In GTSAM, measurement functions are represented as 'factors'.
// The factor we used here is SmartProjectionPoseFactor.
// Every smart factor represent a single landmark, seen from multiple cameras.
// The SmartProjectionPoseFactor only optimizes for the poses of a camera,
// not the calibration, which is assumed known.
#include <gtsam/slam/SmartProjectionPoseFactor.h>

// For an explanation of these headers, see SFMExample.cpp
#include "SFMdata.h"
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <gtsam/nonlinear/ISAM2.h>

#include <unordered_map>

using namespace std;
using namespace gtsam;

// Make the typename short so it looks much cleaner
typedef SmartProjectionPoseFactor<Cal3_S2> SmartFactor;

// create a typedef to the camera type
typedef PinholePose<Cal3_S2> Camera;

/* ************************************************************************* */
int main(int argc, char* argv[]) {

  // Define the camera calibration parameters
  Cal3_S2::shared_ptr K(new Cal3_S2(50.0, 50.0, 0.0, 50.0, 50.0));

  // Define the camera observation noise model
  noiseModel::Isotropic::shared_ptr measurementNoise =
      noiseModel::Isotropic::Sigma(2, 1.0); // one pixel in u and v

  // Create the set of ground-truth landmarks and poses
  vector<Point3> points = createPoints();
  vector<Pose3> poses = createPoses();

  // Create a factor graph
  NonlinearFactorGraph graph;

  ISAM2DoglegParams doglegparams = ISAM2DoglegParams();
  doglegparams.verbose = false;
  ISAM2Params isam2_params;
  isam2_params.evaluateNonlinearError = true;
  isam2_params.relinearizeThreshold = 0.0;
  isam2_params.enableRelinearization = true;
  isam2_params.optimizationParams = doglegparams;
  isam2_params.relinearizeSkip = 1;
  ISAM2 isam2(isam2_params);

  // Simulated measurements from each camera pose, adding them to the factor graph
  unordered_map<int, SmartFactor::shared_ptr> smart_factors;
  for (size_t j = 0; j < points.size(); ++j) {

    // every landmark represent a single landmark, we use shared pointer to init the factor, and then insert measurements.
    SmartFactor::shared_ptr smartfactor(new SmartFactor(measurementNoise, K));

    for (size_t i = 0; i < poses.size(); ++i) {

      // generate the 2D measurement
      Camera camera(poses[i], K);
      Point2 measurement = camera.project(points[j]);

      // call add() function to add measurement into a single factor, here we need to add:
      //    1. the 2D measurement
      //    2. the corresponding camera's key
      //    3. camera noise model
      //    4. camera calibration

      // add only first 3 measurements and update the later measurements incrementally
      if(i < 3)
        smartfactor->add(measurement, i);
    }

    // insert the smart factor in the graph
    smart_factors[j] = smartfactor;
    graph.push_back(smartfactor);
  }

  // Add a prior on pose x0. This indirectly specifies where the origin is.
  // 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
  noiseModel::Diagonal::shared_ptr noise = noiseModel::Diagonal::Sigmas(
      (Vector(6) << Vector3::Constant(0.3), Vector3::Constant(0.1)).finished());
  graph.emplace_shared<PriorFactor<Pose3> >(0, poses[0], noise);

  // Because the structure-from-motion problem has a scale ambiguity, the problem is
  // still under-constrained. Here we add a prior on the second pose x1, so this will
  // fix the scale by indicating the distance between x0 and x1.
  // Because these two are fixed, the rest of the poses will be also be fixed.
  graph.emplace_shared<PriorFactor<Pose3> >(1, poses[1], noise); // add directly to graph

  // Create the initial estimate to the solution
  // Intentionally initialize the variables off from the ground truth
  Values initialEstimate;
  Pose3 delta(Rot3::Rodrigues(-0.1, 0.2, 0.25), Point3(0.05, -0.10, 0.20));
  for (size_t i = 0; i < 3; ++i)
    initialEstimate.insert(i, poses[i].compose(delta));
  initialEstimate.print("Initial Estimates:\n");

  // Optimize the graph and print results
  isam2.update(graph, initialEstimate);
  Values result = isam2.calculateEstimate();
  result.print("Results:\n");

  // we add new measurements from this pose
  size_t pose_idx = 3; 

  // Now update existing smart factors with new observations
  for(size_t j = 0; j < points.size(); ++j)
  {
    SmartFactor::shared_ptr smartfactor = smart_factors[j];

    // add the 4th measurement
    Camera camera(poses[pose_idx], K);
    Point2 measurement = camera.project(points[j]);
    smartfactor->add(measurement, pose_idx);
  }

  graph.resize(0);
  initialEstimate.clear();

  // update initial estimate for the new pose
  initialEstimate.insert(pose_idx, poses[pose_idx].compose(delta));

  // this should break the system
  isam2.update(graph, initialEstimate);

  return 0;
}
/* ************************************************************************* */

