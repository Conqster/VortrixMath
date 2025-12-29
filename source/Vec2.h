#pragma once

#include "Core.h"
#include "VxMath.h"


namespace vx
{

	class alignas(8) Vec2
	{
	public:
		Vec2();
		Vec2(float _x, float _y) : x(_x), y(_y){}
		explicit Vec2(float scalar) : x(scalar), y(scalar){}
		Vec2(const Vec2& rhs) = default;

		VX_INLINE float X() const { return x; }
		VX_INLINE float Y() const { return y; }

		VX_INLINE void SetX(float v) { x = v; }
		VX_INLINE void SetY(float v) { y = v; }

		VX_INLINE float& operator[](uint32_t i);
		VX_INLINE float const& operator[](uint32_t i) const;

		VX_INLINE void ToZero();
		VX_INLINE static Vec2 Zero() { return Vec2(0.0f); }
		VX_INLINE static Vec2 One() { return Vec2(1.0f); }

		VX_INLINE Vec2 Abs() const;
		VX_INLINE Vec2 Sign() const;
		/// IsNaN
		/// checks is this vector contains a component which is NaN
		VX_INLINE bool IsNaN() const;
		VX_INLINE bool IsZero(float tolerance = 1e-6f) const;
		VX_INLINE bool IsApprox(const Vec2& rhs, float tolerance_sq = 1e-12f) const;
		VX_INLINE bool IsNormalised(float tolerance = 1e-6f) const;

		/// Comparison
		VX_INLINE bool operator == (const Vec2& rhs) const;
		VX_INLINE bool operator != (const Vec2& rhs) const { return !(*this == rhs); }

		VX_INLINE float MinComponent() const;
		VX_INLINE float MaxComponent() const;

		VX_INLINE static Vec2 Min(const Vec2& lhs, const Vec2& rhs);
		VX_INLINE static Vec2 Max(const Vec2& lhs, const Vec2& rhs);
		VX_INLINE static Vec2 Clamp(const Vec2& v, const Vec2& min, const Vec2& max);

		VX_INLINE float Dot(const Vec2& rhs) const;
		VX_INLINE float SignedAngle(const Vec2& rhs) const;

		VX_INLINE float LengthSq() const;
		VX_INLINE float Length() const;

		VX_INLINE Vec2 Normalised() const;
		VX_INLINE Vec2& Normalise();

		VX_INLINE Vec2 Inverted() const;
		VX_INLINE Vec2& Invert();

		VX_INLINE Vec2 Perpendicular() const;

		/// Projects this vector onto rhs
		/// assume rhs is of length 1
		/// @returns (this · rhs) rhs
		VX_INLINE Vec2 Project(const Vec2& rhs) const;
		/// Reject (a,b) = a - Project(a, b)
		VX_INLINE Vec2 Reject(const Vec2& onto) const;
		VX_INLINE Vec2 Reflect(const Vec2& nor) const;
		VX_INLINE static Vec2 Lerp(const Vec2& lhs, const Vec2& rhs, float t);

		///Component wise Square root
		VX_INLINE Vec2 Sqrt() const;
		///Component wise Square root in place 
		VX_INLINE Vec2& SqrtAssign();

		VX_INLINE friend std::ostream& operator<<(std::ostream& os, const Vec2& v)
		{
			os << "Vec2(" << v.x << ", " << v.y << ")";
			return os;
		}
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