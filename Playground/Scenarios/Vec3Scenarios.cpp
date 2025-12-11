#include "Vec3Scenarios.h"

void Vec3Scenario::Run()
{
	float x = 1.0f;
	float y = 2.0f;
	float z = 3.0f;

	vx::Vec3(x, y, z).Reciprocal();
}
