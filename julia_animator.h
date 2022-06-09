#pragma once

#include "types.h"


class JuliaAnimator {
public:
	complex_t get_c(unsigned int tick);

private:
	unsigned int get_iterations(const complex_t& c);

	const complex_t i = complex_t(0, 1);

	const float_t OMEGA = 3e-4;

	const unsigned int MAX_ITER = 255;
	const unsigned int TARGET_ITER = 128;
	const float_t SQR_BAILOUT = 16.0f;
	const float_t R_SMOOTHING_ALPHA = 1e-3;

	unsigned int last_tick = 0;
	float_t r = 0.3;
	float_t r_vel = 0;
};