#pragma once

#include "doctest.h"
#include "Vec4.h"
#include "Vec3.h"

inline void CHECK_APPROX_EQ(float a, float b, float eps = 1e-6f)
{
	CHECK(std::abs(a - b) <= eps);
}

inline void CHECK_APPROX_EQ(double a, double b, double eps = 1e-6f)
{
	CHECK(std::abs(a - b) <= eps);
}



inline void CHECK_APPROX_EQ(vx::Vec3 a, vx::Vec3 b, float eps = 1e-6f)
{
	CHECK(a.IsApprox(b, eps * eps));
}


inline void CHECK_APPROX_EQ(vx::Vec4 a, vx::Vec4 b, float eps = 1e-6f)
{
	CHECK(a.IsApprox(b, eps * eps));
}