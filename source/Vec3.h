#pragma once

#include "Core.h"

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

		/// @return x component value
		VX_INLINE float X() const { return mFloats[0]; }
		/// @return y component value
		VX_INLINE float Y() const { return mFloats[1]; }
		/// @return z component value
		VX_INLINE float Z() const { return mFloats[2]; }

		/// set x component
		VX_INLINE void SetX(float v) { mFloats[0] = v; }
		/// set y component
		VX_INLINE void SetY(float v) { mFloats[1] = v; }
		/// set z component
		VX_INLINE void SetZ(float v) { mFloats[2] = v; }

		/// Get SIMD value, Layout (x, y, z, z)
		__m128 Value() const;

		/// Component accessor by index
		/// @param i Index (0 = x, 1 = y, 2 = z)
		/// @return reference to component
		VX_INLINE float& operator[](uint32_t i);
		/// Component accessor by index const
		/// @param i Index (0 = x, 1 = y, 2 = z)
		/// @return const reference to component
		VX_INLINE float const& operator[](uint32_t i) const;

		/// Read component by simd register lane index
		/// @param v vector
		/// @param idx Lane index [0..2]
		VX_INLINE static float GetLane(const Vec3& v, int idx);

		/// Set both components to zero
		VX_INLINE void ToZero();
		/// Vector with all zero 
		VX_INLINE static Vec3 Zero();
		/// Vector with all one 
		VX_INLINE static Vec3 One() { return Vec3(1.0f); }
		/// vector [1, 0, 0]
		VX_INLINE static const Vec3 Right() { return Vec3(1.0f, 0.0f, 0.0f); }
		/// vector [0, 1, 0]
		VX_INLINE static const Vec3 Up() { return Vec3(0.0f, 1.0f, 0.0f); }
		/// vector [0, 0, 1]
		VX_INLINE static const Vec3 Forward() { return Vec3(0.0f, 0.0f, 1.0f); }

		/// Component-wise absolute value
		VX_INLINE Vec3 Abs() const;
		/// Component-wise sign 
		VX_INLINE Vec3 Sign() const;
		/// Check if any component is NaN
		/// @return true if any component is NaN
		VX_INLINE bool IsNaN() const;
		/// Chech if vector approx zero
		/// @return tolerance allowed absolute error
		VX_INLINE bool IsZero(float eps = 1e-6f) const;
		/// Approximate equality check
		/// @return rhs Vector to compare
		/// @param tolerance_sq Square tolerance
		VX_INLINE bool IsApprox(const Vec3& rhs, float eps_sq = 1e-12f) const;
		/// Check if vector is normalised
		/// @param tolerance allowed length error
		VX_INLINE bool IsNormalised(float tolerance = 1e-6f) const;

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

		/// smallest component value
		VX_INLINE float MinComponent() const;
		/// Largest component value
		VX_INLINE float MaxComponent() const;

		/// Index of smallest component
		/// @return 0 for x. 
		/// @return 1 for y.
		/// @return 2 for z.
		VX_INLINE int MinAxis() const;
		/// Index of largest component
		/// @return 0 for x. 
		/// @return 1 for y.
		/// @return 2 for z.
		VX_INLINE int MaxAxis() const;

		/// @return a vector from the smallest component of lhs & rhs vectors
		VX_INLINE static Vec3 Min(const Vec3& lhs, const Vec3& rhs);
		/// @return a vector from the largest component of lhs & rhs vectors
		VX_INLINE static Vec3 Max(const Vec3& lhs, const Vec3& rhs);
		/// Clamp each component between min & max
		/// @param v in vector
		/// @return v in range [min, max]
		VX_INLINE static Vec3 Clamp(const Vec3& v, const Vec3& min, const Vec3& max);

		/// Dot product
		VX_INLINE float Dot(const Vec3& rhs) const;
		/// Dot product
		VX_INLINE static float Dot(const Vec3& lhs, const Vec3& rhs);

		/// The unsigned angle (radians) between this and to
		/// @param this vector from which the angular difference is measured.
		/// @param to vector to which the angular difference is measured.
		/// @return unsigned angle (radians) between the two vectors
		VX_INLINE float Angle(const Vec3& rhs) const;
		/// Cosine of angle between vectors
		VX_INLINE float CosAngle(const Vec3& rhs) const;

		/// Cross product
		/// Operates on x, y, z lanes; w lane preserved
		VX_INLINE Vec3 Cross(const Vec3& rhs) const;
		/// Cross product
		/// Operates on x, y, z lanes; w lane preserved
		VX_INLINE static Vec3 Cross(const Vec3& lhs, const Vec3& rhs);
		/// Scalar triple product (Signed Volume 6_
		/// @return a · (b x c)
		VX_INLINE float ScalarTriple(const Vec3& b, const Vec3& c) const;

		///@return the squared length (magnitude) of the vector
		VX_INLINE float LengthSq() const;
		///@return the length (magnitude) of the vector
		VX_INLINE float Length() const;

		/// normlise this vector
		/// @return normalise copy of this 
		/// of length 1.
		VX_INLINE Vec3 Normalised() const;
		/// normlise this vector in place
		/// @return reference to this
		VX_INLINE Vec3& Normalise();

		/// Negate Vec3
		/// @return negate copy of this
		VX_INLINE Vec3 Inverted() const;
		/// Negate in place 
		/// @return reference to this
		VX_INLINE Vec3& Invert();
		/// @return any perpendicular vector to this.
		VX_INLINE Vec3 Perpendicular() const;
		/// @return any normalisedperpendicular vector to this.
		VX_INLINE Vec3 NormalisedPerpendicular() const;

		/// Projects this vector onto nor
		/// assume nor is of length 1.
		/// @param nor Normalised direction
		/// @returns (this · nor) nor
		VX_INLINE Vec3 Project(const Vec3& nor) const;
		/// Reject this vector onto nor
		/// assume nor is of length 1.
		/// @param nor Normalised direction
		/// @return (this · nor) nor - this
		VX_INLINE Vec3 Reject(const Vec3& nor) const;
		/// Reflect this vector about a normal
		/// assume nor is of length 1.
		/// @param nor Normalised direction
		/// @return R = V - 2 * V.Dot(N) * N
		VX_INLINE Vec3 Reflect(const Vec3& nor) const;
		/// Linear interpolation
		/// @param from Start vector 
		/// @param to End vector
		/// @param t range [0, 1]
		VX_INLINE static Vec3 Lerp(const Vec3& lhs, const Vec3& rhs, float t);

		///Component wise Square root
		VX_INLINE Vec3 Sqrt() const;
		///Component wise Square root in place 
		VX_INLINE Vec3& SqrtAssign();

		/// Flip this vector component sign 
		template<int X, int Y, int Z>
		VX_INLINE void FlipSignAssign();
		/// Flip this vector component sign 
		/// @return this vector flipped
		template<int X, int Y, int Z>
		VX_INLINE [[nodiscard]] Vec3 FlipSign() const;
		/// swizzle (shhuffle) components
		template<int X, int Y, int Z>
		VX_INLINE [[nodiscard]] Vec3 Swizzle() const;

		/// Component-wise Reciprocal
		/// @returns a reciprocated vector of this vector (1/this)
		VX_INLINE Vec3 Reciprocal() const;

		/// broadcast scalar to all components
		VX_INLINE static Vec3 Broadcast(float scalar);
		/// replicate x component to all lanes
		VX_INLINE Vec3 SplatX() const;
		/// replicate z component to all lanes
		VX_INLINE Vec3 SplatY() const;
		/// replicate z component to all lanes
		VX_INLINE Vec3 SplatZ() const;

		/// Load aligned float array
		VX_INLINE Vec3 LoadAligned(const float* v);
		/// Load unaligned float array
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