#pragma once

#include "Core.h"
#include "VxMath.h"


namespace vx
{
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

		Vec3();
		Vec3(float x, float y, float z);
		explicit Vec3(float scalar);
		Vec3(const Vec3& rhs) = default;
		Vec3(const Vec4& rhs);
		Vec3(__m128 vec);


		VX_INLINE float X() const { return mFloats[0]; }
		VX_INLINE float Y() const { return mFloats[1]; }
		VX_INLINE float Z() const { return mFloats[2]; }

		VX_INLINE void SetX(float v) { mFloats[0] = v; }
		VX_INLINE void SetY(float v) { mFloats[1] = v; }
		VX_INLINE void SetZ(float v) { mFloats[2] = v; }

		__m128 Value() const;

		VX_INLINE float& operator[](uint32_t i);
		VX_INLINE float const& operator[](uint32_t i) const;

		VX_INLINE static float GetLane(const Vec3& v, int idx);

		VX_INLINE void ToZero();
		VX_INLINE static Vec3 Zero();
		VX_INLINE static Vec3 One() { return Vec3(1.0f); }
		VX_INLINE static const Vec3 Up() { return Vec3(0.0f, 1.0f, 0.0f); }
		VX_INLINE static const Vec3 Right() { return Vec3(1.0f, 0.0f, 0.0f); }
		VX_INLINE static const Vec3 Forward() { return Vec3(0.0f, 0.0f, 1.0f); }

		VX_INLINE Vec3 Abs() const;
		VX_INLINE Vec3 Sign() const;
		/// IsNaN
		/// checks is this vector contains a component which is NaN
		VX_INLINE bool IsNaN() const;
		VX_INLINE bool IsZero(float eps = 1e-6f) const;
		VX_INLINE bool IsApprox(const Vec3& rhs, float eps_sq = 1e-12f) const;

		VX_INLINE Vec3 operator+(const Vec3& rhs) const;
		VX_INLINE Vec3& operator+=(const Vec3& rhs);
		VX_INLINE Vec3 operator-(const Vec3& rhs) const;
		VX_INLINE Vec3& operator -=(const Vec3& rhs);
		VX_INLINE Vec3 operator*(const float scalar) const;
		VX_INLINE friend Vec3 operator*(const float lhs, const Vec3& rhs);
		VX_INLINE Vec3& operator*=(const float scalar);
		VX_INLINE Vec3 operator/(const float scalar) const;
		VX_INLINE Vec3& operator/=(const float scalar);
		VX_INLINE Vec3 operator -() const;

		///component wise
		///component wise multiply
		VX_INLINE Vec3 operator*(const Vec3& rhs) const;
		VX_INLINE Vec3& operator*=(const Vec3& rhs);
		/// coponent wise divide
		VX_INLINE Vec3 operator/(const Vec3& rhs) const;
		VX_INLINE Vec3& operator/=(const Vec3& rhs);

		/// Comparison
		VX_INLINE bool operator == (const Vec3& rhs) const;
		VX_INLINE bool operator != (const Vec3& rhs) const { return !(*this == rhs); }

		VX_INLINE float MinComponent() const;
		VX_INLINE float MaxComponent() const;

		VX_INLINE int MinAxis() const;
		VX_INLINE int MaxAxis() const;

		VX_INLINE static Vec3 Min(const Vec3& lhs, const Vec3& rhs);
		VX_INLINE static Vec3 Max(const Vec3& lhs, const Vec3& rhs);
		VX_INLINE static Vec3 Clamp(const Vec3& v, const Vec3& min, const Vec3& max);

		VX_INLINE float Dot(const Vec3& rhs) const;
		/// <summary>
		/// Computes the dot product of two 4-dimensional vectors.
		/// l . r = (l.x * r.x) + (l.y + r.y) + (l.z * r.z) + (l.w + r.w)
		/// </summary>
		/// <param name="lhs">The left-hand Vec4 vector (const reference).</param>
		/// <param name="rhs">The right-hand Vec4 vector (const reference).</param>
		/// <returns>The scalar dot product of the two vectors as a float.</returns>
		VX_INLINE static float Dot(const Vec3& lhs, const Vec3& rhs);

		VX_INLINE float Angle(const Vec3& rhs) const;
		VX_INLINE float CosAngle(const Vec3& rhs) const;

		/// solve cross on x, y, z, first three lane
		/// and store on first three lane,
		/// fourth lane (w) constant
		VX_INLINE Vec3 Cross(const Vec3& rhs);
		/// solve cross on x, y, z, first three lane
		/// and store on first three lane,
		/// fourth lane (w) constant
		VX_INLINE static Vec3 Cross(const Vec3& lhs, const Vec3& rhs);
		/// a * (b x c) Dot(a, Cross(b, c)
		/// Signed Volume 6
		VX_INLINE float ScalarTriple(const Vec3& b, const Vec3& c) const;


		VX_INLINE float LengthSq() const;
		VX_INLINE float Length() const;

		VX_INLINE Vec3 Normalised() const;
		VX_INLINE Vec3& Normalise();

		VX_INLINE Vec3 Inverted() const;
		VX_INLINE Vec3& Invert();

		///Component wise Square root
		VX_INLINE Vec3 Sqrt() const;
		///Component wise Square root in place 
		VX_INLINE Vec3& SqrtAssign();


		/// Reciprocal
		/// @eturns a reciprocated vector of this vector (1/this)
		VX_INLINE Vec3 Reciprocal() const;

		VX_INLINE static Vec3 Broadcast(float scalar);
		VX_INLINE Vec3 SplatX() const;
		VX_INLINE Vec3 SplatY() const;
		VX_INLINE Vec3 SplatZ() const;

		VX_INLINE Vec3 LoadAligned(const float* v);
		VX_INLINE Vec3 Load(const float* v);

		VX_INLINE friend std::ostream& operator<<(std::ostream& os, const Vec3& v)
		{
			os << "Vec3(" << v.X() << ", " << v.Y() << ", " << v.Z() << ")";
			return os;
		}

	private:
		union
		{
			float mFloats[4];
			__m128 mValue;
		};
	};
}
#include "Vec3.inl"