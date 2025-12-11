#pragma once

#include "Core.h"
#include "VxMath.h"



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

		Vec4();
		Vec4(float x, float y, float z, float w);
		/// w = z
		Vec4(float x, float y, float z);
		explicit Vec4(float scalar);
		Vec4(__m128 vec) : mValue(vec) {}

		VX_FORCE_INLINE float X() const { return mFloat[0]; }
		VX_FORCE_INLINE float Y() const { return mFloat[1]; }
		VX_FORCE_INLINE float Z() const { return mFloat[2]; }
		VX_FORCE_INLINE float W() const { return mFloat[3]; }

		VX_FORCE_INLINE void SetX(float v) { mFloat[0] = v; }
		VX_FORCE_INLINE void SetY(float v) { mFloat[1] = v; }
		VX_FORCE_INLINE void SetZ(float v) { mFloat[2] = v; }
		VX_FORCE_INLINE void SetW(float v) { mFloat[3] = v; }

		__m128 Value() const;

		VX_FORCE_INLINE float& operator[](uint32_t i);
		VX_FORCE_INLINE float const& operator[](uint32_t i) const;

		VX_FORCE_INLINE static float GetLane(const Vec4& v, int idx);

		VX_FORCE_INLINE void ToZero();
		VX_FORCE_INLINE static Vec4 Zero();
		VX_FORCE_INLINE static Vec4 One() { return Vec4(1.0f); }

		VX_FORCE_INLINE Vec4 Abs() const;
		VX_FORCE_INLINE Vec4 Sign() const;
		/// IsNaN
		/// checks is this vector contains a component which is NaN
		VX_FORCE_INLINE bool IsNaN() const;
		VX_FORCE_INLINE bool IsZero(float eps = 1e-6f) const;
		VX_FORCE_INLINE bool IsApprox(const Vec4& rhs, float eps_sq = 1e-12f);

		VX_FORCE_INLINE static Vec4 LoadAligned(const float* v);
		VX_FORCE_INLINE static Vec4 Load(const float* v);
		VX_FORCE_INLINE void Store(float* o_v) const;

		VX_FORCE_INLINE Vec4 operator+(const Vec4& rhs) const;
		VX_FORCE_INLINE Vec4& operator+=(const Vec4& rhs);
		VX_FORCE_INLINE Vec4 operator-(const Vec4& rhs) const;
		VX_FORCE_INLINE Vec4& operator -=(const Vec4& rhs);
		VX_FORCE_INLINE Vec4 operator*(const float scalar) const;
		///component wise multiply
		VX_FORCE_INLINE Vec4 operator*(const Vec4& rhs) const;
		VX_FORCE_INLINE Vec4& operator*=(const float scalar);
		VX_FORCE_INLINE Vec4& operator*=(const Vec4& rhs);

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
		VX_FORCE_INLINE static Vec4 Clamp(const Vec4& v, const Vec4& min, const Vec4& max);

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

		VX_FORCE_INLINE Vec4 Normalised() const;
		VX_FORCE_INLINE Vec4& Normalise();

		VX_FORCE_INLINE Vec4 Inverted() const;
		VX_FORCE_INLINE Vec4& Invert();

		/// Reciprocal
		/// @eturns a reciprocated vector of this vector (1/this)
		VX_FORCE_INLINE Vec4 Reciprocal() const;


		VX_FORCE_INLINE Vec3 XYZ() const;
		VX_FORCE_INLINE Vec4 XYZZ() const;
		VX_FORCE_INLINE Vec4 XYZ0() const;
		VX_FORCE_INLINE Vec4 XYZ1() const;

		VX_FORCE_INLINE static Vec4 Broadcast(float scalar);
		VX_FORCE_INLINE Vec4 SplatX() const;
		VX_FORCE_INLINE Vec4 SplatY() const;
		VX_FORCE_INLINE Vec4 SplatZ() const;
		VX_FORCE_INLINE Vec4 SplatW() const;

		/// solve cross on x, y, z, first three lane
		/// and store on first three lane,
		/// fourth lane (w) constant
		VX_FORCE_INLINE static Vec3 Cross3(const Vec4& lhs, const Vec4& rhs);
		VX_FORCE_INLINE friend std::ostream& operator<<(std::ostream& os, const Vec4& v)
		{
			os << "Vec4(" << v.X() << ", " << v.Y() << ", " << v.Z() << ", " << v.W() << ")";
			return os;
		}

	private:
		union
		{
			float mFloat[4];
			__m128 mValue;
		};


	};

}

#include "Vec4.inl"