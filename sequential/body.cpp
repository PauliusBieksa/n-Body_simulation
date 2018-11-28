#include "body.h"

void Body::step(const double &dt)
{
	vel += (force / mass) * dt;
	pos += vel * dt;
}