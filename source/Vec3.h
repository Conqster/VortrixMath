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


		VX_FORCE_INLINE float X() const { return mFloat[0]; }
		VX_FORCE_INLINE float Y() const { return mFloat[1]; }
		VX_FORCE_INLINE float Z() const { return mFloat[2]; }

		VX_FORCE_INLINE void SetX(float v) { mFloat[0] = v; }
		VX_FORCE_INLINE void SetY(float v) { mFloat[1] = v; }
		VX_FORCE_INLINE void SetZ(float v) { mFloat[2] = v; }

		__m128 Value() const;

		VX_FORCE_INLINE float& operator[](uint32_t i);
		VX_FORCE_INLINE float const& operator[](uint32_t i) const;

		VX_FORCE_INLINE static float GetLane(const Vec3& v, int idx);

		VX_FORCE_INLINE void ToZero();
		VX_FORCE_INLINE static Vec3 Zero();
		VX_FORCE_INLINE static Vec3 One() { return Vec3(1.0f); }
		VX_FORCE_INLINE static const Vec3 Up() { return Vec3(0.0f, 1.0f, 0.0f); }
		VX_FORCE_INLINE static const Vec3 Right() { return Vec3(1.0f, 0.0f, 0.0f); }
		VX_FORCE_INLINE static const Vec3 Forward() { return Vec3(0.0f, 0.0f, 1.0f); }

		VX_FORCE_INLINE Vec3 Abs() const;
		VX_FORCE_INLINE Vec3 Sign() const;
		/// IsNaN
		/// checks is this vector contains a component which is NaN
		VX_FORCE_INLINE bool IsNaN() const;
		VX_FORCE_INLINE bool IsZero(float eps = 1e-6f) const;
		VX_FORCE_INLINE bool IsApprox(const Vec3& rhs, float eps_sq = 1e-12f);

		VX_FORCE_INLINE Vec3 LoadAligned(const float* v);
		VX_FORCE_INLINE Vec3 Load(const float* v);
		VX_FORCE_INLINE void Store(float* o_v) const;

		VX_FORCE_INLINE Vec3 operator+(const Vec3& rhs) const;
		VX_FORCE_INLINE Vec3& operator+=(const Vec3& rhs);
		VX_FORCE_INLINE Vec3 operator-(const Vec3& rhs) const;
		VX_FORCE_INLINE Vec3& operator -=(const Vec3& rhs);
		VX_FORCE_INLINE Vec3 operator*(const float scalar) const;
		///component wise multiply
		VX_FORCE_INLINE Vec3 operator*(const Vec3& rhs) const;
		VX_FORCE_INLINE Vec3& operator*=(const float scalar);
		VX_FORCE_INLINE Vec3& operator*=(const Vec3& rhs);

		Vec3 operator/(const float scalar) const;
		/// coponent wisw
		VX_FORCE_INLINE Vec3 operator/(const Vec3& rhs) const;
		Vec3& operator/=(const float scalar);

		VX_FORCE_INLINE float MinComponent() const;
		VX_FORCE_INLINE float MaxComponent() const;

		VX_FORCE_INLINE int MinAxis() const;
		VX_FORCE_INLINE int MaxAxis() const;

		VX_FORCE_INLINE static Vec3 Min(const Vec3& lhs, const Vec3& rhs);
		VX_FORCE_INLINE static Vec3 Max(const Vec3& lhs, const Vec3& rhs);
		VX_FORCE_INLINE static Vec3 Clamp(const Vec3& v, const Vec3& min, const Vec3& max);

		/// Comparison
		VX_FORCE_INLINE bool operator == (const Vec3& rhs) const;
		VX_FORCE_INLINE bool operator != (const Vec3& rhs) const { return !(*this == rhs); }


		VX_FORCE_INLINE float Dot(const Vec3& rhs) const;
		/// <summary>
		/// Computes the dot product of two 4-dimensional vectors.
		/// l . r = (l.x * r.x) + (l.y + r.y) + (l.z * r.z) + (l.w + r.w)
		/// </summary>
		/// <param name="lhs">The left-hand Vec4 vector (const reference).</param>
		/// <param name="rhs">The right-hand Vec4 vector (const reference).</param>
		/// <returns>The scalar dot product of the two vectors as a float.</returns>
		VX_FORCE_INLINE static float Dot(const Vec3& lhs, const Vec3& rhs);

		VX_FORCE_INLINE float LengthSq() const;
		VX_FORCE_INLINE float Length() const;

		VX_FORCE_INLINE Vec3 Normalised() const;
		VX_FORCE_INLINE Vec3& Normalise();

		VX_FORCE_INLINE Vec3 Inverted() const;
		VX_FORCE_INLINE Vec3& Invert();

		/// Reciprocal
		/// @eturns a reciprocated vector of this vector (1/this)
		VX_FORCE_INLINE Vec3 Reciprocal() const;

		VX_FORCE_INLINE static Vec3 Broadcast(float scalar);
		VX_FORCE_INLINE Vec3 SplatX() const;
		VX_FORCE_INLINE Vec3 SplatY() const;
		VX_FORCE_INLINE Vec3 SplatZ() const;

		/// solve cross on x, y, z, first three lane
		/// and store on first three lane,
		/// fourth lane (w) constant
		VX_FORCE_INLINE static Vec3 Cross(const Vec3& lhs, const Vec3& rhs);


		VX_FORCE_INLINE friend std::ostream& operator<<(std::ostream& os, const Vec3& v)
		{
			os << "Vec3(" << v.X() << ", " << v.Y() << ", " << v.Z() << ")";
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
#include "Vec3.inl"