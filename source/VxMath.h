#pragma once
#include "Core.h"

namespace vx
{


	static constexpr float kVxPi = 3.14159265358979323846f;
	static constexpr float kVxTau = 6.283185307179586f;
	static constexpr float kEpsilon = 1e-6f;


	template<typename T>
	inline constexpr T DegToRad(T deg) { return deg * static_cast<T>(kVxPi) / static_cast<T>(180.0f);
	}
	template<typename T>
	inline constexpr T RadToDeg(T rad) { return rad * static_cast<T>(180.0f) / static_cast<T>(kVxPi); }



	template<typename T>
	VX_INLINE T VxMax(T a, T b)
	{
		return a < b ? b : a;
	}

	template<typename T>
	VX_INLINE T VxMin(T a, T b)
	{
		return a < b ? a : b;
	}


	VX_INLINE float VxAbs(float v)
	{
		return std::fabs(v);
	}

	VX_INLINE double VxAbs(double v)
	{
		return std::abs(v);
	}



	template<typename T>
	VX_INLINE T VxClamp(T v, T _min, T _max)
	{
		return VxMin(VxMax(v, _min), _max);
	}

	VX_INLINE float VxSqr(float v)
	{
		return v * v;
	}

	VX_INLINE double VxSqr(double v)
	{
		return v * v;
	}

	VX_INLINE float VxSqrt(float v)
	{
		return std::sqrt(v);
	}


	VX_INLINE double VxSqrt(double v)
	{
		return std::sqrt(v);
	}

	VX_INLINE float VxApprox(float a, float b, float eps = 1e-6f)
	{
		float diff = VxAbs(a - b);
		float scale = VxMax(VxMax(VxAbs(a), VxAbs(b)), 1.0f);
		return diff <= eps * scale;
	}

	VX_INLINE double VxApprox(double a, double b, double eps = 1e-6f)
	{
		float diff = VxAbs(a - b);
		float scale = VxMax(VxMax(VxAbs(a), VxAbs(b)), 1.0);
		return diff <= eps * scale;
	}


	/// Polynomial approximation
	/// prone to small error (float) 23-bit mantissa
	/// sin(pi), mathematically = 0, numerically very small, not zero
	VX_INLINE float VxSin(float v)
	{
		return std::sin(v);
	}
	/// Polynomial approximation
	/// prone to small error (float) 23-bit mantissa
	/// cos(pi), mathematically = 0, numerically very small, not zero
 	VX_INLINE double VxSin(double v)
	{
		return std::sin(v);
	}
	/// Polynomial approximation
	/// prone to small error (float) 23-bit mantissa
	/// cos(pi/2), mathematically = 0, numerically very small, not zero
	VX_INLINE float VxCos(float v)
	{
		return std::cos(v);
	}
	/// Polynomial approximation
	/// prone to small error (float) 23-bit mantissa
	/// cos(pi/2), mathematically = 0, numerically very small, not zero
	VX_INLINE double VxCos(double v)
	{
		return std::cos(v);
	}


	VX_INLINE float VxAcos(float v)
	{
		return std::acos(v);
	}

	VX_INLINE double VxAcos(double v)
	{
		return std::acos(v);
	}
}