#include "Vec2.h"


namespace vx
{
	inline VX_INLINE float& Vec2::operator[](uint32_t i)
	{
		VX_ASSERT(i < 2, "Trying to access invalid Vec2 index");
		return mFloats[i];
	}
	inline VX_INLINE float const& Vec2::operator[](uint32_t i) const
	{
		VX_ASSERT(i < 2, "Trying to access invalid Vec2 index");
		return mFloats[i];
	}

	inline VX_INLINE void Vec2::ToZero()
	{
		x = 0.0f;
		y = 0.0f;
	}
	VX_INLINE Vec2 Vec2::Abs() const
	{
		return Vec2(VxAbs(x), VxAbs(y));
	}

	VX_INLINE Vec2 Vec2::Sign() const
	{
		return Vec2(std::copysign(1.0f, x), std::copysign(1.0f, y));
	}

	VX_INLINE bool Vec2::IsNaN() const
	{
		return (std::isnan(x) || std::isnan(y));
	}

	inline VX_INLINE bool Vec2::IsZero(float tolerance) const
	{
		return x <= tolerance && y <= tolerance;
	}
	inline VX_INLINE bool Vec2::IsApprox(const Vec2& rhs, float tolerance_sq) const
	{
		return (rhs - *this).LengthSq() <= tolerance_sq;
	}

	inline VX_INLINE bool Vec2::IsNormalised(float tolerance) const
	{
		return VxAbs(LengthSq() - 1.0f) <= tolerance;
	}

	VX_INLINE bool Vec2::operator == (const Vec2& rhs) const
	{
		return x == rhs.x && y == rhs.y;
	}

	inline VX_INLINE float Vec2::MinComponent() const
	{
#ifdef VX_USE_SSE
		__m128 v = SimdValue();
		v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)));
		return _mm_cvtss_f32(v);

#else
		return VxMin(x, y);
#endif // VX_USE_SSE
	}

	inline VX_INLINE float Vec2::MaxComponent() const
	{
#ifdef VX_USE_SSE
		__m128 v = SimdValue();
		v = _mm_max_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)));
		return _mm_cvtss_f32(v);
#else
		return VxMax(x, y);
#endif // VX_USE_SSE

	}

	inline VX_INLINE Vec2 Vec2::Min(const Vec2& lhs, const Vec2& rhs)
	{
#ifdef VX_USE_SSE
		__m128 v = _mm_min_ps(lhs.SimdValue(), rhs.SimdValue());
		Vec2 out;
		_mm_storel_pi(reinterpret_cast<__m64*>(&out), v);
		return out;
#else
		return Vec2(std::min(lhs.x, rhs.x),
			std::min(lhs.y, rhs.y));
#endif // VX_USE_SSE
	}

	inline VX_INLINE Vec2 Vec2::Max(const Vec2& lhs, const Vec2& rhs)
	{
#ifdef VX_USE_SSE
		__m128 v = _mm_max_ps(lhs.SimdValue(), rhs.SimdValue());
		Vec2 out;
		_mm_storel_pi(reinterpret_cast<__m64*>(&out), v);
		return out;
#else
		return Vec2(std::max(lhs.x, rhs.x),
			std::max(lhs.y, rhs.y));
#endif // VX_USE_SSE
	}

	inline VX_INLINE Vec2 Vec2::Clamp(const Vec2& v, const Vec2& min, const Vec2& max)
	{
#ifdef VX_USE_SSE
		__m128 r = _mm_max_ps(_mm_min_ps(v.SimdValue(), max.SimdValue()), min.SimdValue());
		Vec2 out;
		_mm_storel_pi(reinterpret_cast<__m64*>(&out), r);
		return out;
#else
		return Max(Min(v, max), min);
#endif // VX_USE_SSE
	}

	inline VX_INLINE float Vec2::Dot(const Vec2& rhs) const
	{
#ifdef VX_USE_SSE
		/// 0x71 -> 0111 0001 : op first 3 & store 1 (first)
		/// 0xf1 -> 1111 0001 : op first 4 & store 1 (first)
		/// 
		/// 0x77 -> 0111 0111 : op first 3 & store 3 (first)
		/// 0xff -> 1111 1111 : op first 4 & store 4 (first)
		/// 
		/// 0x7f -> 0111 1111 : op first 3 & store 4 (first)
		/// 0x31 -> 0011 0001 : op first 2 & store 4 (first)
		/// 
		/// as 0111 0001 
		/// high nibble 0111 [bit 4 - 7] (nibble 1 = 4bits, 0.5bytes) 
		/// low nibble 0001	 [bit 0 - 3]
		/// using with _mm_dp_ps 
		/// high nibble defines, the bits to op on (multply its components) 
		/// 0111 x, y, z, without w 
		/// low nibble defines, the bits to store result
		/// 0001 only x excluding y, z, and w

		//dot product op first 3 & store 1 (x) then extract 1 (0:x)
		return _mm_cvtss_f32(_mm_dp_ps(SimdValue(), rhs.SimdValue(), 0x31));

#else
		float dot = 0.0f;
		for (int i = 0; i < 2; ++i)
			dot += (mFloats[i] * rhs.mFloats[i]);
		return dot;
#endif // VX_USE_SSE
	}


	inline VX_INLINE float Vec2::LengthSq() const
	{
#ifdef VX_USE_SSE
		//dot product op first 2 & store 1 (x) then extract 1 (0:x) 0x31 ->0011 0001
		__m128 v = SimdValue();
		return _mm_cvtss_f32(_mm_dp_ps(v, v, 0x31));
#else
		float len_sq = 0.0f;
		for (int i = 0; i < 2; ++i)
			len_sq += (mFloats[i] * mFloats[i]);
		return len_sq;
#endif // VX_USE_SSE
	}

	inline VX_INLINE float Vec2::Length() const
	{
#ifdef VX_USE_SSE
		//dot product op first 2 & store 1 (x) then extract 1 (0:x) 0x31 ->0011 0001
		__m128 v = SimdValue();
		return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(v, v, 0x31)));
#else
		float len_sq = 0.0f;
		for (int i = 0; i < 2; ++i)
			len_sq += (mFloats[i] * mFloats[i]);
		return std::sqrt(len_sq);
#endif // VX_USE_SSE
	}

	inline VX_INLINE Vec2 Vec2::Normalised() const
	{
		Vec2 r;
		r.Normalise();
		return r;
	}

	inline VX_INLINE Vec2& Vec2::Normalise()
	{
#ifdef VX_USE_SSE
		/// 0x3f -> 0011 1111 : op first 2 & store 4 (first)
		///         0zyx dddd 
		__m128 v = SimdValue();
		__m128 dot = _mm_dp_ps(v, v, 0x3f);
		__m128 safe_ep = _mm_max_ps(dot, _mm_set_ps1(1e-6f));
		v = _mm_div_ps(v, _mm_sqrt_ps(safe_ep));

		_mm_storel_pi(reinterpret_cast<__m64*>(this), v);
#else
		float length_sq = 0.0f;
		for (int i = 0; i < 3; ++i)
			length_sq += (mFloats[i] * mFloats[i]);
		if (length_sq > 1e-6f)
		{
			const float inv = 1.0f / std::sqrt(length_sq);
			for (int i = 0; i < 3; ++i)
				mFloats[i] *= inv;
		}
#endif // VX_USE_SSE

		return *this;
	}


	inline VX_INLINE Vec2 Vec2::Inverted() const
	{
		//I think simd xor_ps is a waste
		return Vec2(-x, -y);
	}

	inline VX_INLINE Vec2& Vec2::Invert()
	{
		//I think simd xor_ps is a waste
		x = -x;
		y = -y;
		return *this;
	}


	inline VX_INLINE Vec2 Vec2::Perpendicular() const
	{
		return Vec2(-y, x);
	}

	inline VX_INLINE Vec2 Vec2::Project(const Vec2& rhs) const
	{
#ifdef VX_USE_SSE
		/// 0x3f -> 0011 1111 : op first 2 & store 4 (first)
		///         0zyx dddd 
		__m128 v = SimdValue();
		__m128 n = rhs.SimdValue();
		__m128 dot = _mm_mul_ps(v, n);
		dot = _mm_hadd_ps(dot, dot);
		v = _mm_mul_ps(dot, n);
		Vec2 out;
		_mm_storel_pi(reinterpret_cast<__m64*>(&out), v);
		return out;
#else
		return Dot(rhs) * rhs;
#endif // VX_USE_SSE
	}

	inline VX_INLINE Vec2 Vec2::Reject(const Vec2& rhs) const
	{
#ifdef VX_USE_SSE
		/// 0x3f -> 0011 1111 : op first 2 & store 4 (first)
		///         0zyx dddd 
		__m128 v = SimdValue();
		__m128 n = rhs.SimdValue();
		__m128 dot = _mm_mul_ps(v, n);
		dot = _mm_hadd_ps(dot, dot);
		v = _mm_mul_ps(dot, _mm_sub_ps(v, n));
		Vec2 out;
		_mm_storel_pi(reinterpret_cast<__m64*>(&out), v);
		return out;
#else
		return (Dot(rhs) * (rhs - *this));
#endif // VX_USE_SSE
	}

	inline VX_INLINE Vec2 Vec2::Reflect(const Vec2& nor) const
	{
		/// R = V - 2 * V.Dot(N) * N
#ifdef VX_USE_SSE
		__m128 v = SimdValue();
		__m128 n = nor.SimdValue();
		//all lane
		__m128 d = _mm_mul_ps(v, n);
		d = _mm_hadd_ps(d, d);
		__m128 _2_d = _mm_mul_ps(_mm_set1_ps(2.0f), d);
		//__m128 r = _mm_mul_ps(_mm_sub_ps(v, _2_d), n);
		__m128 r = _mm_sub_ps(n, _mm_sub_ps(v, _2_d));
		Vec2 out;
		_mm_storel_pi(reinterpret_cast<__m64*>(&out), r);
		return out;
#else
		return *this - (2 * Dot(nor)) * nor;
#endif // VX_USE_SSE
	}

	inline VX_INLINE Vec2 Vec2::Lerp(const Vec2& lhs, const Vec2& rhs, float t)
	{
#ifdef VX_USE_SSE
		__m128 r = simd::Lerp(lhs.SimdValue(), rhs.SimdValue(), t);
		Vec2 out;
		_mm_storel_pi(reinterpret_cast<__m64*>(&out), r);
		return out;
#else
		return Vec2(
			VxLerp(lhs.x, rhs.x, t),
			VxLerp(lhs.y, rhs.y, t));
#endif // VX_USE_SSE
	}


	inline VX_INLINE Vec2 Vec2::Sqrt() const
	{
		return Vec2(VxSqrt(x), VxSqrt(y));
	}

	inline VX_INLINE Vec2& Vec2::SqrtAssign()
	{
		x = VxSqrt(x);
		y = VxSqrt(y);
		return *this;
	}

	inline VX_INLINE __m128 Vec2::SimdValue() const
	{
		return _mm_setr_ps(x, y, 0.0f, 0.0f);
	}

}