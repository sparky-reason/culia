#pragma once

#include "types.h"

/**
* Class responsible for animating the Julia set, i.e. calculating the value of c over time.
* 
* c is varied with constant angular velocity over the complex plane.
* The radius (modulus) of c is adapted smoothly to keep the center point of the Julia set close to a desired iteration number.
*/
class JuliaAnimator
{
public:
	/**
	* Get Julia set constant.
	* 
	* @param tick  time in ms
	*/
	complex_t get_c(unsigned int tick);

private:
	// Get number of iterations of center point of Julia set for a given c
	unsigned int get_iterations(const complex_t& c);

	const complex_t i = complex_t(0, 1);

	const float_t OMEGA = 3e-4f;  // angular velocity

	const unsigned int MAX_ITER = 255;  // maximum number of iterations for get_iterations
	const unsigned int TARGET_ITER = 128;  // target number of iterations for the center point
	const float_t SQR_BAILOUT = 16.0f;  // squared bailout value for aborting the iteration in get_iterations
	const float_t R_SMOOTHING_ALPHA = 1e-3f;  // smoothing value for updating the radius velocity, higher -> less momentum

	unsigned int last_tick = 0;  // last seen tick
	float_t r = 0.3f;  // c radius
	float_t r_vel = 0.0f;  // c radius velocity
};