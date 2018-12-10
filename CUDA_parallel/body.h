#pragma once

#include <glm/glm.hpp>

class Body
{
public:
	glm::dvec2 pos;
	glm::dvec2 vel;
	double mass;
	glm::dvec2 force = glm::dvec2(0.0f);

	Body::Body() { pos = glm::dvec2(0.0f); }
	Body::Body(glm::dvec2 p, double m) { pos = p; vel = glm::dvec2(0.0); mass = m; }
	Body::Body(glm::dvec2 p, glm::dvec2 v, double m) { pos = p; pos = v; mass = m; }

private:
};