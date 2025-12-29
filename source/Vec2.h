#pragma once

#include "Core.h"
#include "VxMath.h"


namespace vx
{

	class alignas(8) Vec2
	{
	public:
		Vec2() = default;
		Vec2(float _x, float _y) : x(_x), y(_y){}
		explicit Vec2(float scalar) : x(scalar), y(scalar){}
		Vec2(const Vec2& rhs) = default;
		explicit Vec2(__m128 vec);

		VX_INLINE float X() const { return x; }
		VX_INLINE float Y() const { return y; }

		VX_INLINE void SetX(float v) { x = v; }
		VX_INLINE void SetY(float v) { y = v; }

		/// Component accessor
		/// @param i range[0 1]
		/// @return i(0) = x
		/// @return i(1) = y
		VX_INLINE float& operator[](uint32_t i);
		VX_INLINE float const& operator[](uint32_t i) const;

		VX_INLINE void ToZero();
		VX_INLINE static Vec2 Zero() { return Vec2(0.0f); }
		VX_INLINE static Vec2 One() { return Vec2(1.0f); }
		VX_INLINE static Vec2 Right() { return Vec2(1.0f, 0.0f); }
		VX_INLINE static Vec2 Up() { return Vec2(0.0f, 1.0f); }

		VX_INLINE Vec2 Abs() const;
		VX_INLINE Vec2 Sign() const;
		/// IsNaN
		/// @return true if any component is NaN
		VX_INLINE bool IsNaN() const;
		/// @return true if vector approximate zero within tolerance
		VX_INLINE bool IsZero(float tolerance = 1e-6f) const;
		VX_INLINE bool IsApprox(const Vec2& rhs, float tolerance_sq = 1e-12f) const;
		VX_INLINE bool IsNormalised(float tolerance = 1e-6f) const;

		/// Comparison
		VX_INLINE bool operator == (const Vec2& rhs) const;
		VX_INLINE bool operator != (const Vec2& rhs) const { return !(*this == rhs); }

		VX_INLINE float MinComponent() const;
		VX_INLINE float MaxComponent() const;

		/// @return a vector from the smallest component of lhs & rhs vectors
		VX_INLINE static Vec2 Min(const Vec2& lhs, const Vec2& rhs);
		/// @return a vector from the largest component of lhs & rhs vectors
		VX_INLINE static Vec2 Max(const Vec2& lhs, const Vec2& rhs);
		VX_INLINE static Vec2 Clamp(const Vec2& v, const Vec2& min, const Vec2& max);

		VX_INLINE float Dot(const Vec2& rhs) const;
		/// The unsigned angle (radians) between this and to
		/// @param this vector from which the angular difference is measured.
		/// @param to vector to which the angular difference is measured.
		/// @return unsigned angle (radians) between the two vectors
		VX_INLINE float Angle(const Vec2& to) const;
		/// The signed angle (radians) between this and to
		/// @param this vector from which the angular difference is measured.
		/// @param to vector to which the angular difference is measured.
		/// @return signed angle (radians) between the two vectors
		VX_INLINE float SignedAngle(const Vec2& to) const;

		///@return the squared length (magnitude) of the vector
		VX_INLINE float LengthSq() const;
		///@return the length (magnitude) of the vector
		VX_INLINE float Length() const;

		/// @return a normlised vector of this vector 
		/// of length 1.
		VX_INLINE Vec2 Normalised() const;
		/// normlises this vector to length 1.
		/// @return a ref to this vector normalised.
		VX_INLINE Vec2& Normalise();

		VX_INLINE Vec2 Inverted() const;
		VX_INLINE Vec2& Invert();
		/// Compute a perpendicular vector to the vector 
		/// @return perpendicular vector to this.
		VX_INLINE Vec2 Perpendicular() const;

		/// Projects this vector onto rhs
		/// assume rhs is of length 1
		/// @returns (this · rhs) rhs
		VX_INLINE Vec2 Project(const Vec2& rhs) const;
		/// Reject (a,b) = a - Project(a, b)
		/// @return a rejection vector of this from b 
		/// assume onto is of length 1.
		VX_INLINE Vec2 Reject(const Vec2& onto) const;
		VX_INLINE Vec2 Reflect(const Vec2& nor) const;
		/// Linear interpolate between vectors lhs & rhs by t
		/// @param t range [0, 1]
		VX_INLINE static Vec2 Lerp(const Vec2& from, const Vec2& to, float t);

		///Component wise Square root
		VX_INLINE Vec2 Sqrt() const;
		///Component wise Square root in place 
		VX_INLINE Vec2& SqrtAssign();


		VX_INLINE Vec2 operator+(const Vec2& rhs) const;
		VX_INLINE Vec2& operator+=(const Vec2& rhs);
		VX_INLINE Vec2 operator-(const Vec2& rhs) const;
		VX_INLINE Vec2& operator -=(const Vec2& rhs);
		VX_INLINE Vec2 operator*(const float scalar) const;
		VX_INLINE friend Vec2 operator*(const float lhs, const Vec2& rhs);
		VX_INLINE Vec2& operator*=(const float scalar);
		VX_INLINE Vec2 operator/(const float scalar) const;
		VX_INLINE Vec2& operator/=(const float scalar);
		VX_INLINE Vec2 operator -() const;

		///component wise
		///component wise multiply
		VX_INLINE Vec2 operator*(const Vec2& rhs) const;
		VX_INLINE Vec2& operator*=(const Vec2& rhs);
		/// coponent wise divide
		VX_INLINE Vec2 operator/(const Vec2& rhs) const;
		VX_INLINE Vec2& operator/=(const Vec2& rhs);

		VX_INLINE friend std::ostream& operator<<(std::ostream& os, const Vec2& v)
		{
			os << "Vec2(" << v.x << ", " << v.y << ")";
			return os;
		}

#ifdef VX_USE_SSE
		VX_INLINE void Store(__m128 v)
		{
			_mm_storel_pi(reinterpret_cast<__m64*>(this), v);
		}
#endif // VX_USE_SSE

	private:

		VX_INLINE __m128 SimdValue() const;
		union
		{
			struct { float x, y; };
			//for iterative support
			float mFloats[2];
		};
	};
}
#include "Vec2.inl"