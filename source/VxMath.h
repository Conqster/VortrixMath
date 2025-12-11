#pragma once
#include "Core.h"

namespace vx
{


	static constexpr float kVxPi = 3.14159265358979323846f;
	static constexpr float kVxTau = 6.283185307179586f;
	static constexpr float kEpsilon = 1e-6f;


	template<typename T>
	VX_FORCE_INLINE T VxMax(T a, T b)
	{
		return a < b ? b : a;
	}

	template<typename T>
	VX_FORCE_INLINE T VxMin(T a, T b)
	{
		return a < b ? a : b;
	}


	VX_FORCE_INLINE float VxAbs(float v)
	{
		return std::fabs(v);
	}

	VX_FORCE_INLINE double VxAbs(double v)
	{
		return std::abs(v);
	}



	template<typename T>
	VX_FORCE_INLINE T VxClamp(T v, T _min, T _max)
	{
		return VxMin(VxMax(v, _min), _max);
	}

	VX_FORCE_INLINE float VxSqr(float v)
	{
		return v * v;
	}

	VX_FORCE_INLINE double VxSqr(double v)
	{
		return v * v;
	}

	VX_FORCE_INLINE float VxSqrt(float v)
	{
		return std::sqrt(v);
	}


	VX_FORCE_INLINE double VxSqrt(double v)
	{
		return std::sqrt(v);
	}

	VX_FORCE_INLINE float VxApprox(float a, float b, float eps = 1e-6f)
	{
		float diff = VxAbs(a - b);
		float scale = VxMax(VxMax(VxAbs(a), VxAbs(b)), 1.0f);
		return diff <= eps * scale;
	}

	VX_FORCE_INLINE double VxApprox(double a, double b, double eps = 1e-6f)
	{
		float diff = VxAbs(a - b);
		float scale = VxMax(VxMax(VxAbs(a), VxAbs(b)), 1.0);
		return diff <= eps * scale;
	}
}