#pragma once

#include "Core.h"

/*
	* Vec3
	* 3D vector class
	*
	* Represents a 3-component vector with support for scalar
	* and vector operations, normalisation, dot and cross products.
	*
	* Layout
	* - [x, y, z]: Cartesian components
	* 
	* in memory extends to 4 dimension (x, y, z, z)
	*/
class alignas(16) Vec3
{
public:
	union
	{
		float mData32[4];
		__m128 mValue;
	};

	Vec3()
	{
#if USE_SIMD_SSE
		mValue = _mm_setzero_ps();
		for (int i = 0; i < 4; i++)
			mData32[i] = 0;
#else
#endif // USE_SIMD_SSE

	}
	Vec3(float x, float y, float z)
	{
		mValue = _mm_set_ps(z, z, y, x);
	}
	explicit Vec3(float scalar)
	{
		mValue = _mm_set1_ps(scalar);
	}
	Vec3(__m128 vec) : mValue(vec) {}

#if USE_SIMD_SSE
	FORCE_INLINE float X() const { return _mm_cvtss_f32(mValue); }
	FORCE_INLINE float Y() const { return _mm_cvtss_f32(_mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(1, 1, 1, 1))); }
	FORCE_INLINE float Z() const { return _mm_cvtss_f32(_mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(2, 2, 2, 2))); }
#else
	
	FORCE_INLINE float X() const { return mData32[0]; }
	FORCE_INLINE float Y() const { return mData32[1]; }
	FORCE_INLINE float Z() const { return mData32[2]; }
#endif // USE_SIMD_SSE

	FORCE_INLINE static float GetLane(const Vec3& v, int idx);

	FORCE_INLINE static Vec3 Zero() { return Vec3(_mm_setzero_ps()); }
	FORCE_INLINE void ToZero() { mValue = _mm_setzero_ps(); }

	FORCE_INLINE static Vec3 Broadcast(float scalar);
	FORCE_INLINE Vec3 LoadAligned(const float* v);
	FORCE_INLINE Vec3 Load(const float* v);
	FORCE_INLINE void Store(float* o_v) const;

	FORCE_INLINE Vec3 operator+(const Vec3& rhs) const;
	FORCE_INLINE Vec3& operator+=(const Vec3& rhs);
	FORCE_INLINE Vec3 operator-(const Vec3& rhs) const;
	FORCE_INLINE Vec3& operator -=(const Vec3& rhs);
	FORCE_INLINE Vec3 operator*(const float scalar) const;
	FORCE_INLINE Vec3& operator*=(const float scalar);
	/// for testing vexctorised division
	FORCE_INLINE Vec3 Divide(const float scalar);
	Vec3 operator/(const float scalar) const;
	Vec3& operator/=(const float scalar);

	FORCE_INLINE float MinComponent() const;
	FORCE_INLINE float MaxComponent() const;

	FORCE_INLINE int MinAxis() const;
	FORCE_INLINE int MaxAxis() const;

	FORCE_INLINE static Vec3 Min(const Vec3& lhs, const Vec3& rhs);
	FORCE_INLINE static Vec3 Max(const Vec3& lhs, const Vec3& rhs);

	FORCE_INLINE float Dot(const Vec3& rhs) const;
	/// <summary>
	/// Computes the dot product of two 4-dimensional vectors.
	/// l . r = (l.x * r.x) + (l.y + r.y) + (l.z * r.z) + (l.w + r.w)
	/// </summary>
	/// <param name="lhs">The left-hand Vec4 vector (const reference).</param>
	/// <param name="rhs">The right-hand Vec4 vector (const reference).</param>
	/// <returns>The scalar dot product of the two vectors as a float.</returns>
	FORCE_INLINE static float Dot(const Vec3& lhs, const Vec3& rhs);

	FORCE_INLINE float LengthSq() const;
	FORCE_INLINE float Length() const;

	FORCE_INLINE Vec3 Normalised_NOT_SIMD() const;
	FORCE_INLINE Vec3 Normalised() const;
	FORCE_INLINE Vec3& Normalise();

	FORCE_INLINE Vec3 Inverted() const;
	FORCE_INLINE Vec3& Invert();

	FORCE_INLINE static Vec3 Cross_NOT_SIMD(const Vec3& lhs, const Vec3& rhs);
	FORCE_INLINE static Vec3 Cross(const Vec3& lhs, const Vec3& rhs);
	/// solve cross on x, y, z, first three lane
	/// and store on first three lane,
	/// fourth lane (w) constant
	/// 
	//FORCE_INLINE Vec3& Cross3(const Vec4& rhs);


	FORCE_INLINE friend std::ostream& operator<<(std::ostream& os, const Vec3& v)
	{
		os << "Vec3(" << v.X() << ", " << v.Y() << ", " << v.Z() << ")";
		return os;
	}
};
