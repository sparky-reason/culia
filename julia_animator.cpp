#include "julia_animator.h"

using namespace std;


complex_t JuliaAnimator::get_c(unsigned int tick)
{
	float_t t = tick;
	float_t dt = last_tick == 0 ? 0.0f : tick - last_tick;
	last_tick = tick;

	complex_t candidate_c = r * exp(i * OMEGA * t);
	float_t r_direction = float_t(get_iterations(candidate_c)) - float_t(TARGET_ITER);

	r_vel = exp(-dt * R_SMOOTHING_ALPHA) * r_vel + (1.0f - exp(-dt * R_SMOOTHING_ALPHA)) * r_direction;
	r *= exp(1e-2 * r_vel * OMEGA * dt);

	return r * exp(i * OMEGA * t);
}

unsigned int JuliaAnimator::get_iterations(const complex_t& c)
{
	complex_t z = 0;
	for (unsigned int i = 0; i < MAX_ITER; ++i) {
		z = z * z + c;
		if (z.real() * z.real() + z.imag() * z.imag() > SQR_BAILOUT)
			return i;
	}
	return MAX_ITER;
}
