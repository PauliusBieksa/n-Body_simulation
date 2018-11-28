#pragma once

#define DURATION 5.0 // Duration for the simulation
#define DELTA_TIME 0.01 // duration of a single step
#define N_BODIES 50 // Number of bodies simulated
#define G_CONSTANT 4.302e-3 // Gravitational constant measured in - pc / M * (km/s)(km/s)
							// pc - parcec, M - solar mass unit
#define MASS_LOWER_BOUND 0.5 // Lower bound for randomly generated mass. 1 unit represents 1 solar unit
#define MASS_HIGHER_BOUND 1.5 // Lower bound for randomly generated mass. 1 unit represents 1 solar unit
#define POS_LOWER_BOUND -1.0 // Lowest posible x and/or y component for a starting position of the bodies. 1 unit represents 1 parcec
#define POS_HIGHER_BOUND 1.0 // Highest posible x and/or y component for a starting position of the bodies. 1 unit represents 1 parcec

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 800

#define RANDOM_SEED 99 // Seed used to get pseudo-random numbers