#pragma once
#include "Core.h"

namespace vx
{

	namespace simd
	{
		VX_INLINE constexpr float Lane128(const __m128& v, int idx)
		{
			switch (idx)
			{
			case 0: return _mm_cvtss_f32(v);
			case 1: return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)));
			case 2: return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2)));
			case 3: return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3)));
			default: return _mm_cvtss_f32(v);
			}
		}

		template<int X, int Y, int Z, int W>
		VX_INLINE constexpr __m128 SignMask()
		{
			return _mm_castsi128_ps(_mm_set_epi32(
							(W < 0) ? 0x80000000 : 0x00000000, 
							(Z < 0) ? 0x80000000 : 0x00000000, 
							(Y < 0) ? 0x80000000 : 0x00000000, 
							(X < 0) ? 0x80000000 : 0x00000000));
		}

		template<bool X, bool Y, bool Z, bool W>
		VX_INLINE constexpr __m128 LaneMask()
		{
			return _mm_castsi128_ps(_mm_set_epi32(
				W ? -1 : 0,
				Z ? -1 : 0,
				Y ? -1 : 0,
				X ? -1 : 0));
		}

		template<int X, int Y, int Z, int W>
		VX_INLINE constexpr __m128 FlipSign(__m128 v)
		{
			return _mm_xor_ps(v, SignMask<X, Y, Z, W>());
		}

		VX_INLINE __m128 Lerp(const __m128& a, const __m128& b, float t)
		{
			__m128 tt = _mm_set1_ps(t);
			return _mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_set1_ps(1.0f), tt), a), 
				_mm_mul_ps(tt, b));
		}
		VX_INLINE __m128 Xor(const __m128& v, const __m128& mask)
		{
			return _mm_xor_ps(v, mask);
		}


		template<int X, int Y, int Z, int W>
		VX_INLINE __m128 Swizzle(__m128 v)
		{
			VX_ASSERT(X >=0 && X <= 3, "X out of [0, 3] range");
			VX_ASSERT(Y >=0 && Y <= 3, "X out of [0, 3] range");
			VX_ASSERT(Z >=0 && Z <= 3, "X out of [0, 3] range");
			VX_ASSERT(W >=0 && W <= 3, "X out of [0, 3] range");
			return _mm_shuffle_ps(v, v, _MM_SHUFFLE(W, Z, Y, X));
		}

		template<int X, int Y, int Z, int W>
		VX_INLINE __m128 Swizzle(__m128 v0, __m128 v1)
		{
			VX_ASSERT(X >= 0 && X <= 3, "X out of [0, 3] range");
			VX_ASSERT(Y >= 0 && Y <= 3, "X out of [0, 3] range");
			VX_ASSERT(Z >= 0 && Z <= 3, "X out of [0, 3] range");
			VX_ASSERT(W >= 0 && W <= 3, "X out of [0, 3] range");
			return _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(W, Z, Y, X));
		}

	}

	static constexpr float kVxPi = 3.14159265358979323846f;
	static constexpr float kVxTau = 6.283185307179586f;
	static constexpr float kEpsilon = 1e-6f;
	static constexpr float kQuietNaNf = std::numeric_limits<float>::quiet_NaN();
	static constexpr int kQuietNaNi = std::numeric_limits<int>::quiet_NaN();


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


	VX_INLINE float VxLerp(float a, float b, float t)
	{
		return ((1.0f - t) * a) + t * b;
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

	VX_INLINE float VxAtan2(float y, float x)
	{
		return std::atan2(y, x);
	}
}