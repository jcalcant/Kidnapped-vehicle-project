/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::max_element;

void ParticleFilter::init(double gps_x, double gps_y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  std::default_random_engine gen;

  // Create normal distributions for x, y and theta
  normal_distribution<double> distr_x(gps_x, std[0]);
  normal_distribution<double> distr_y(gps_y, std[1]);
  normal_distribution<double> distr_theta(theta, std[2]);

  // Sample from these normal distributions
  for (int i = 0; i < num_particles; ++i) {
    // instantiate particle
    Particle p;
    p.id = i;
    p.x = distr_x(gen);
    p.y = distr_y(gen);
    p.theta = distr_theta(gen);
    p.weight = 1.0;
    // add to particles and weights vectors
    particles.push_back(p);
    weights.push_back(p.weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  for (unsigned int i = 0; i < particles.size(); i++) {
    // add measurement predictions
    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;

      // add random gaussian noise
      normal_distribution<double> noise_x(0, std_pos[0]);
      normal_distribution<double> noise_y(0, std_pos[1]);
      normal_distribution<double> noise_theta(0, std_pos[2]);

      particles[i].x += noise_x(gen);
      particles[i].y += noise_y(gen);
      particles[i].theta += noise_theta(gen);
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  for (unsigned int i = 0; i < observations.size(); i++) {
    vector<double> distances;
    // calculate distance between observation and predicted measurement
    for (unsigned int j = 0; j < predicted.size(); j++) {
      distances.push_back(dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y));
    }
    // save the index of the minimum difference
    int landmark_index = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));
    observations[i].id = predicted[landmark_index].id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int p = 0; p < num_particles; p++) { //iterate over all particles
    double particle_x = particles[p].x;
    double particle_y = particles[p].y;
    double particle_theta = particles[p].theta;

    vector<LandmarkObs> transformed_observations; //observations transformed to map coordinates
    for (unsigned int i = 0; i < observations.size(); i++) { //iterate over all observations
      // transform to map x coordinate
      double x_map;
      double x_obs = observations[i].x;
      double y_obs = observations[i].y;

      x_map = particle_x + (cos(particle_theta) * x_obs) - (sin(particle_theta) * y_obs);
      // transform to map y coordinate
      double y_map;
      y_map = particle_y + (sin(particle_theta) * x_obs) + (cos(particle_theta) * y_obs);

      LandmarkObs obs;
      obs.x = x_map;
      obs.y = y_map;
      transformed_observations.push_back(obs);
    }

    vector<LandmarkObs> predictions;
    //Loop through each map landmark and append if it is within the sensor's range
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      //Get id and x,y coordinates
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      if (dist(landmark_x, landmark_y, particle_x, particle_y) <= sensor_range) {
        predictions.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
      }
    }

    dataAssociation(predictions, transformed_observations);

    // calculate the probability for each particle given its observations and the predicted landmarks
    double particle_weight = 1.0;
    for (unsigned int o = 0; o < transformed_observations.size(); o++) {
      double obs_x, obs_y, pred_x, pred_y;
      obs_x = transformed_observations[o].x;
      obs_y = transformed_observations[o].y;

      for (unsigned int p = 0; p < predictions.size(); p++) {
        if (predictions[p].id == transformed_observations[o].id) {
          pred_x = predictions[p].x;
          pred_y = predictions[p].y;
        }
      }

      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];

      // calculate normalization term
      double gauss_norm;
      gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

      // calculate exponent
      double exponent;
      exponent = (pow(obs_x - pred_x, 2) / (2 * pow(sig_x, 2)))
                 + (pow(obs_y - pred_y, 2) / (2 * pow(sig_y, 2)));

      // calculate weight using normalization terms and exponent
      double w;
      w = gauss_norm * exp(-exponent);

      particle_weight *= w;
    }
    particles[p].weight = particle_weight;

    // update weights vector
    weights[p] = particle_weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::default_random_engine gen;
  std::discrete_distribution<> dist(weights.begin(), weights.end()); //weighted distribution to sample from
  std::vector<Particle> new_particles;

  //sample new particles until num_particles (particles.size() ) is reached
  while (new_particles.size() < particles.size()) {
    int id = dist(gen);
    new_particles.push_back(particles[id]);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
