#pragma once

#include "Core.h"



namespace vx
{
	//class Vec4;
	//
	//class alignas(16) Mat44
	//{
	//public:
	//	union {
	//		float data[16];
	//		Vec4 col[4];
	//	};
	//};


	//helper to get lane 


	/// Vec4 is maintained to support Mat44 op & stores
	class alignas(16) Vec4
	{
	public:

		union
		{
			float mData32[4];
			__m128 mValue;
		};

		Vec4()
		{
#if VX_USE_SSE
			mValue = _mm_setzero_ps();
#else
			for (int i = 0; i < 4; i++)
				mData32[i] = 0;
#endif // USE_SIMD_SSE
		}
		Vec4(float _x, float _y, float _z, float _w)
		{
#if VX_USE_SSE
			mValue = _mm_setr_ps(_x, _y, _z, _w);
#else
			mData32[0] = _x;
			mData32[1] = _y;
			mData32[2] = _z;
			mData32[3] = _w;
#endif // USE_SIMD_SSE

		}

		/// w = z
		Vec4(float _x, float _y, float _z) 
		{
#if VX_USE_SSE
			mValue = _mm_setr_ps(_x, _y, _z, _z);
#else
			mData32[0] = _x;
			mData32[1] = _y;
			mData32[2] = _z;
			mData32[3] = _z;
#endif // USE_SIMD_SSE

		}

		explicit Vec4(float scalar)
		{
#if VX_USE_SSE
			mValue = _mm_set1_ps(scalar);
#else
			for (int i = 0; i < 4; i++)
				mData32[i] = scalar;
#endif // USE_SIMD_SSE

		}

#if VX_USE_SSE
		Vec4(__m128 vec) : mValue(vec) {}

		VX_FORCE_INLINE float X() const { return _mm_cvtss_f32(mValue); }
		VX_FORCE_INLINE float Y() const { return _mm_cvtss_f32(_mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(1, 1, 1, 1))); }
		VX_FORCE_INLINE float Z() const { return _mm_cvtss_f32(_mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(2, 2, 2, 2))); }
		VX_FORCE_INLINE float W() const { return _mm_cvtss_f32(_mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(3, 3, 3, 3))); }
#else
		Vec4(__m128 vec) : mValue(vec) {}

		FORCE_INLINE float X() const { return mData32[0]; }
		FORCE_INLINE float Y() const { return mData32[1]; }
		FORCE_INLINE float Z() const { return mData32[2]; }
		FORCE_INLINE float W() const { return mData32[3]; }
#endif // USE_SIMD_SSE

		VX_FORCE_INLINE static float GetLane(const Vec4& v, int idx);

		VX_FORCE_INLINE static Vec4 Zero();
		VX_FORCE_INLINE void ToZero();

		VX_FORCE_INLINE static Vec4 One() { return Vec4(1.0f); }
		VX_FORCE_INLINE Vec4 Abs() const;
		VX_FORCE_INLINE Vec4 Sign() const;

		/// IsNaN
		/// checks is this vector contains a component which is NaN
		VX_FORCE_INLINE bool IsNaN() const;

		//VPHX_INLINE Vec3 SplatX() const;
		VX_FORCE_INLINE static Vec4 Broadcast(float scalar);

		VX_FORCE_INLINE static Vec4 LoadAligned(const float* v);
		VX_FORCE_INLINE static Vec4 Load(const float* v);
		VX_FORCE_INLINE void Store(float* o_v) const;

		VX_FORCE_INLINE Vec4 operator+(const Vec4& rhs) const;
		VX_FORCE_INLINE Vec4& operator+=(const Vec4& rhs);
		VX_FORCE_INLINE Vec4 operator-(const Vec4& rhs) const;
		VX_FORCE_INLINE Vec4& operator -=(const Vec4& rhs);
		VX_FORCE_INLINE Vec4 operator*(const float scalar) const;
		VX_FORCE_INLINE Vec4& operator*=(const float scalar);
		/// for testing vexctorised division
		VX_FORCE_INLINE Vec4 Divide(const float scalar);
		VX_FORCE_INLINE Vec4 operator/(const float scalar) const;
		/// coponent wisw
		VX_FORCE_INLINE Vec4 operator/(const Vec4& rhs) const;
		VX_FORCE_INLINE Vec4& operator/=(const float scalar);

		VX_FORCE_INLINE float MinComponent() const;
		VX_FORCE_INLINE float MaxComponent() const;

		VX_FORCE_INLINE int MaxAxis() const;
		VX_FORCE_INLINE int MinAxis() const;

		VX_FORCE_INLINE static Vec4 Min(const Vec4& lhs, const Vec4& rhs);
		VX_FORCE_INLINE static Vec4 Max(const Vec4& lhs, const Vec4& rhs);

		/// Comparison
		VX_FORCE_INLINE bool operator == (const Vec4& rhs) const;
		VX_FORCE_INLINE bool operator != (const Vec4& rhs) const { return !(*this == rhs); }

		VX_FORCE_INLINE float Dot(const Vec4& rhs) const;
		/// <summary>
		/// Computes the dot product of two 4-dimensional vectors.
		/// l . r = (l.x * r.x) + (l.y + r.y) + (l.z * r.z) + (l.w + r.w)
		/// </summary>
		/// <param name="lhs">The left-hand Vec4 vector (const reference).</param>
		/// <param name="rhs">The right-hand Vec4 vector (const reference).</param>
		/// <returns>The scalar dot product of the two vectors as a float.</returns>
		VX_FORCE_INLINE static float Dot(const Vec4& lhs, const Vec4& rhs);

		VX_FORCE_INLINE float LengthSq() const;
		VX_FORCE_INLINE float Length() const;

		VX_FORCE_INLINE Vec4 Normalised_NOT_SIMD() const;
		VX_FORCE_INLINE Vec4 Normalised() const;
		VX_FORCE_INLINE Vec4& Normalise();

		VX_FORCE_INLINE Vec4 Inverted() const;
		VX_FORCE_INLINE Vec4& Invert();

		/// Reciprocal
		/// @eturns a reciprocated vector of this vector (1/this)
		VX_FORCE_INLINE Vec4 Reciprocal() const;

		VX_FORCE_INLINE static Vec3 Cross3_NOT_SIMD(const Vec4& lhs, const Vec4& rhs);
		VX_FORCE_INLINE static Vec3 Cross3(const Vec4& lhs, const Vec4& rhs);
		/// solve cross on x, y, z, first three lane
		/// and store on first three lane,
		/// fourth lane (w) constant
		/// 
		//FORCE_INLINE Vec3& Cross3(const Vec4& rhs);

		VX_FORCE_INLINE friend std::ostream& operator<<(std::ostream& os, const Vec4& v)
		{
			os << "Vec4(" << v.X() << ", " << v.Y() << ", " << v.Z() << ", " << v.W() << ")";
			return os;
		}
	};

}

#include "Vec4.inl"