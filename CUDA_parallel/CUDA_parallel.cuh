#pragma once

#include <vector>

class Par_handler
{
public:
	Par_handler(int n_of_bodies);
	~Par_handler();
	void physics_step(std::vector<double> &bodies_formated, const double &dt);
private:
	double *before, *after;
};