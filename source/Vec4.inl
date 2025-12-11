#include "Vec4.h"
#include "Vec3.h"

namespace vx
{
	inline Vec4::Vec4()
	{
#ifdef VX_USE_SSE
		mValue = _mm_setzero_ps();
#else
		for (int i = 0; i < 4; i++)
			mFloat[i] = 0;
#endif // USE_SIMD_SSE
	}
	inline Vec4::Vec4(float x, float y, float z, float w)
	{
#ifdef VX_USE_SSE
		mValue = _mm_setr_ps(x, y, z, w);
#else
		mFloat[0] = x;
		mFloat[1] = y;
		mFloat[2] = z;
		mFloat[3] = w;
#endif // USE_SIMD_SSE

	}
	inline Vec4::Vec4(float x, float y, float z)
	{
#ifdef VX_USE_SSE
		mValue = _mm_setr_ps(x, y, z, z);
#else
		mFloat[0] = x;
		mFloat[1] = y;
		mFloat[2] = z;
		mFloat[3] = z;
#endif // USE_SIMD_SSE
	}

	inline Vec4::Vec4(float scalar)
	{
#ifdef VX_USE_SSE
		mValue = _mm_set1_ps(scalar);
#else
		for (int i = 0; i < 4; i++)
			mFloat[i] = scalar;
#endif // USE_SIMD_SSE
	}

	VX_FORCE_INLINE __m128 Vec4::Value() const
	{
#ifdef VX_USE_SSE
		return mValue;
#else
		return _mm_load_ps(mFloat);
#endif // VX_USE_SSE
	}

	inline VX_FORCE_INLINE float& Vec4::operator[](uint32_t i)
	{
		VX_ASSERT(i < 4, "Trying to access invalid Vec4 index");
		return mFloat[i];
	}
	inline VX_FORCE_INLINE float const& Vec4::operator[](uint32_t i) const
	{
		VX_ASSERT(i < 4, "Trying to access invalid Vec4 index");
		return mFloat[i];
	}
	inline VX_FORCE_INLINE float Vec4::GetLane(const Vec4& v, int idx)
	{
#ifdef VX_USE_SSE
		return Get_m128Lane(v.mValue, idx);
#else
		return v.mFloat[idx];
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE Vec4 Vec4::Zero()
	{
#ifdef VX_USE_SSE
		//return Vec4(_mm_setzero_ps());
		return _mm_setzero_ps();
#else
		return Vec4(0.0f);
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE void Vec4::ToZero()
	{
#ifdef VX_USE_SSE
		mValue = _mm_setzero_ps();
#else
		for (int i = 0; i < 4; i++)
			mFloat[i] = 0.0f;
#endif // USE_SIMD_SSE
	}

	VX_FORCE_INLINE Vec4 Vec4::Abs() const
	{
#ifdef VX_USE_SSE
		return _mm_max_ps(_mm_sub_ps(_mm_setzero_ps(), mValue), mValue);
#else
		return Vec4(std::fabs(mFloat[0]), std::fabs(mFloat[1]), fabs(mFloat[2]), fabs(mFloat[3]));
#endif // USE_SIMD_SSE
	}

	VX_FORCE_INLINE Vec4 Vec4::Sign() const
	{
#ifdef VX_USE_SSE
		return _mm_or_ps(_mm_and_ps(mValue, _mm_set_ps1(-1.0f)), _mm_set_ps1(1.0f));
#else
		return Vec4(std::copysign(1.0f, mFloat[0]), std::copysign(1.0f, mFloat[1]), std::copysign(1.0f, mFloat[2]), std::copysign(1.0f, mFloat[3]));
#endif // USE_SIMD_SSE
	}

	VX_FORCE_INLINE bool Vec4::IsNaN() const
	{
		return (std::isnan(mFloat[0]) || 
			    std::isnan(mFloat[1]) ||
				std::isnan(mFloat[2]) ||
				std::isnan(mFloat[3]));
	}

	inline VX_FORCE_INLINE bool Vec4::IsZero(float eps) const
	{
		return mFloat[0] <= eps &&
			mFloat[1] <= eps &&
			mFloat[2] <= eps &&
			mFloat[3] <= eps;
	}

	inline VX_FORCE_INLINE bool Vec4::IsApprox(const Vec4& rhs, float eps_sq)
	{
		return (rhs - *this).LengthSq() <= eps_sq;
	}

	inline VX_FORCE_INLINE Vec4 Vec4::LoadAligned(const float* v)
	{
#ifdef VX_USE_SSE
		return _mm_load_ps(v);
#else
		return Vec4(v[0], v[1], v[2], v[3]);
#endif // USE_SIMD_SSE

	}

	inline VX_FORCE_INLINE Vec4 Vec4::Load(const float* v)
	{
#ifdef VX_USE_SSE
		return _mm_loadu_ps(v);
#else
		return Vec4(v[0], v[1], v[2], v[3]);
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE void Vec4::Store(float* o_v) const
	{
#ifdef VX_USE_SSE
		_mm_store_ps(o_v, mValue);
#else
#warning not store not support without SIMD
		//return Vec4(v[0], v[1], v[2], v[3]);
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE Vec4 Vec4::operator+(const Vec4& rhs) const
	{
#ifdef VX_USE_SSE
		return _mm_add_ps(mValue, rhs.mValue);
#else
		return Vec4(mFloat[0] + rhs.mFloat[0],
			mFloat[1] + rhs.mFloat[1],
			mFloat[2] + rhs.mFloat[2],
			mFloat[3] + rhs.mFloat[3]);
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE Vec4& Vec4::operator+=(const Vec4& rhs)
	{
#ifdef VX_USE_SSE
		mValue = _mm_add_ps(mValue, rhs.mValue);
#else
		for (int i = 0; i < 4; ++i)
			mFloat[i] += rhs.mFloat[i];
#endif // USE_SIMD_SSE

		return *this;
	}

	inline VX_FORCE_INLINE Vec4 Vec4::operator-(const Vec4& rhs) const
	{
#ifdef VX_USE_SSE
		return _mm_sub_ps(this->mValue, rhs.mValue);
#else
		return Vec4(mFloat[0] - rhs.mFloat[0],
			mFloat[1] - rhs.mFloat[1],
			mFloat[2] - rhs.mFloat[2],
			mFloat[3] - rhs.mFloat[3]);
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE Vec4& Vec4::operator-=(const Vec4& rhs)
	{
#ifdef VX_USE_SSE
		mValue = _mm_sub_ps(this->mValue, rhs.mValue);
#else
		for (int i = 0; i < 4; ++i)
			mFloat[i] -= rhs.mFloat[i];
#endif // USE_SIMD_SSE

		return *this;
	}

	inline VX_FORCE_INLINE Vec4 Vec4::operator*(const float scalar) const
	{
#ifdef VX_USE_SSE
		//broad cast or load scalar
		return _mm_mul_ps(mValue, _mm_set_ps1(scalar));
#else
		return Vec4(mFloat[0] * scalar,
			mFloat[1] * scalar,
			mFloat[2] * scalar,
			mFloat[3] * scalar);
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE Vec4 Vec4::operator*(const Vec4& rhs) const
	{
#ifdef VX_USE_SSE
		//broad cast or load scalar
		return _mm_mul_ps(mValue, rhs.mValue);
#else
		return Vec4(mFloat[0] * rhs.mFloat[0],
			mFloat[1] * rhs.mFloat[1],
			mFloat[2] * rhs.mFloat[2],
			mFloat[3] * rhs.mFloat[3]);
#endif // VX_USE_SSE
	}

	inline VX_FORCE_INLINE Vec4& Vec4::operator*=(const float scalar)
	{
#ifdef VX_USE_SSE
		mValue = _mm_mul_ps(mValue, _mm_set_ps1(scalar));
#else
		for (int i = 0; i < 4; ++i)
			mFloat[i] *= scalar;
#endif // USE_SIMD_SSE

		return *this;
	}

	inline VX_FORCE_INLINE Vec4& Vec4::operator*=(const Vec4& rhs)
	{
#ifdef VX_USE_SSE
		//broad cast or load scalar
		mValue = _mm_mul_ps(mValue, rhs.mValue);
#else
		for (int i = 0; i < 4; ++i)
			mFloat[i] *= rhs.mFloat[i];
#endif // VX_USE_SSE

		return *this;
	}

	inline Vec4 Vec4::operator/(const float scalar) const
	{
#ifdef VX_USE_SSE
		//return _mm_div_ps(value, _mm_set_ps1(scalar));//<- expensive div per lane
		//return _mm_mul_ps(value, _mm_set_ps1(1.0f /scalar)); <- loss precision due 1/s, most case fastest 

		//one lane division, shuffle x lane across, then mul
		__m128 v = _mm_div_ss(_mm_set_ps1(1.0f), _mm_set_ps1(scalar));
		return _mm_mul_ps(mValue, _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)));
		return _mm_mul_ps(mValue, _mm_set_ps1(scalar));
#else
		return Vec4(mFloat[0] / scalar,
			mFloat[1] / scalar,
			mFloat[2] / scalar,
			mFloat[3] / scalar);
#endif // USE_SIMD_SSE
	}


	Vec4 Vec4::operator/(const Vec4& rhs) const
	{
#ifdef VX_USE_SSE
		//return _mm_div_ps(value, _mm_set_ps1(scalar));//<- expensive div per lane
		//return _mm_mul_ps(value, _mm_set_ps1(1.0f /scalar)); <- loss precision due 1/s, most case fastest 

		//one lane division, shuffle x lane across, then mul
		return _mm_div_ps(mValue, rhs.mValue);
#else
		return Vec4(mFloat[0] / rhs.mFloat[0],
			mFloat[1] / rhs.mFloat[1],
			mFloat[2] / rhs.mFloat[2],
			mFloat[3] / rhs.mFloat[3]);
#endif // USE_SIMD_SSE
	}

	inline Vec4& Vec4::operator/=(const float scalar)
	{
#ifdef VX_USE_SSE
		//return _mm_div_ps(value, _mm_set_ps1(scalar));//<- expensive div per lane
		//return _mm_mul_ps(value, _mm_set_ps1(1.0f /scalar)); <- loss precision due 1/s, most case fastest 

		//one lane division, shuffle x lane across, then mul
		__m128 v = _mm_div_ss(_mm_set_ps1(1.0f), _mm_set_ps1(scalar));
		mValue = _mm_mul_ps(mValue, _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)));
#else
		for (int i = 0; i < 4; ++i)
			mFloat[i] /= scalar;
#endif // USE_SIMD_SSE

		return *this;
	}

	inline VX_FORCE_INLINE float Vec4::MinComponent() const
	{
#ifdef VX_USE_SSE
		//__m128 shuf1 = _mm_shuffle_ps(value, value, _MM_SHUFFLE(2, 3, 0, 1)); // y z w x
		//__m128 min1 = _mm_min_ps(value, shuf1); // min(x,y) min(y,z) min(z,w) min(w,x)
		//__m128 shuf2 = _mm_shuffle_ps(min1, min1, _MM_SHUFFLE(1, 0, 3, 2)); // z w x y
		//__m128 min2 = _mm_min_ps(min1, shuf2); // min(min(x,y),min(z,w)) min(min(y,z),min(w,x)) ...
		//return _mm_cvtss_f32(min2); // extract the first element

		__m128 v = mValue;
		v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1))); //min x & z min z & w
		v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
		return _mm_cvtss_f32(v);
#else
		return (X() < Y()) ?
			((X() < Z()) ?
				((X() < W()) ? mFloat[0] : mFloat[3]) :
				(Z() < W() ? mFloat[2] : mFloat[3])) :
			((Y() < Z()) ?
				((Y() < W()) ? mFloat[1] : mFloat[3]) :
				(Z() < W() ? mFloat[2] : mFloat[3]));

#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE float Vec4::MaxComponent() const
	{
#ifdef VX_USE_SSE
		//__m128 shuf1 = _mm_shuffle_ps(value, value, _MM_SHUFFLE(2, 3, 0, 1)); // y z w x
		//__m128 max1 = _mm_max_ps(value, shuf1); // max(x,y) max(y,z) max(z,w) max(w,x)
		//__m128 shuf2 = _mm_shuffle_ps(max1, max1, _MM_SHUFFLE(1, 0, 3, 2)); // z w x y
		//__m128 max2 = _mm_max_ps(max1, shuf2); // max(max(x,y),max(z,w)) max(max(y,z),max(w,x)) ...
		//return _mm_cvtss_f32(max2); // extract the first element

		__m128 v = mValue;
		v = _mm_max_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1))); //min x & z (shuffle 
		v = _mm_max_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
		return _mm_cvtss_f32(v);
#else
		return (X() > Y()) ?
			((X() > Z()) ?
				((X() > W()) ? mFloat[0] : mFloat[3]) :
				(Z() > W() ? mFloat[2] : mFloat[3])) :
			((Y() > Z()) ?
				((Y() > W()) ? mFloat[1] : mFloat[3]) :
				(Z() > W() ? mFloat[2] : mFloat[3]));
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE int Vec4::MaxAxis() const
	{
		return (X() > Y()) ? ((X() > Z()) ? ((X() > W()) ? 0 : 3) : (Z() > W() ? 2 : 3)) : ((Y() > Z()) ? ((Y() > W()) ? 1 : 3) : (Z() > W() ? 2 : 3));
	}

	inline VX_FORCE_INLINE int Vec4::MinAxis() const
	{
		return (X() < Y()) ? ((X() < Z()) ? ((X() < W()) ? 0 : 3) : (Z() < W() ? 2 : 3)) : ((Y() < Z()) ? ((Y() < W()) ? 1 : 3) : (Z() < W() ? 2 : 3));
	}

	inline VX_FORCE_INLINE Vec4 Vec4::Min(const Vec4& lhs, const Vec4& rhs)
	{
#ifdef VX_USE_SSE
		return _mm_min_ps(lhs.mValue, rhs.mValue);
#else
		return Vec4(std::min(lhs.mFloat[0], rhs.mFloat[0]),
			std::min(lhs.mFloat[1], rhs.mFloat[1]),
			std::min(lhs.mFloat[2], rhs.mFloat[2]),
			std::min(lhs.mFloat[3], rhs.mFloat[3]));
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE Vec4 Vec4::Max(const Vec4& lhs, const Vec4& rhs)
	{
#ifdef VX_USE_SSE
		return _mm_max_ps(lhs.mValue, rhs.mValue);
#else
		return Vec4(std::max(lhs.mFloat[0], rhs.mFloat[0]),
			std::max(lhs.mFloat[1], rhs.mFloat[1]),
			std::max(lhs.mFloat[2], rhs.mFloat[2]),
			std::max(lhs.mFloat[3], rhs.mFloat[3]));
#endif // USE_SIMD_SSE

	}

	inline VX_FORCE_INLINE Vec4 Vec4::Clamp(const Vec4& v, const Vec4& min, const Vec4& max)
	{
		return Max(Min(v, max), min);
	}


	VX_FORCE_INLINE bool Vec4::operator == (const Vec4& rhs) const
	{
#ifdef VX_USE_SSE
		__m128i cmp = _mm_castps_si128(_mm_cmpeq_ps(mValue, rhs.mValue));
		 int mask = _mm_movemask_ps(_mm_castsi128_ps(cmp));
		return mask == 0b1111;
#else
		uint32_t v[4];
		v[0] = mFloat[0] == rhs.mFloat[0] ? 0xffffffffu : 0;
		v[1] = mFloat[1] == rhs.mFloat[1] ? 0xffffffffu : 0;
		v[2] = mFloat[2] == rhs.mFloat[2] ? 0xffffffffu : 0;
		v[3] = mFloat[3] == rhs.mFloat[3] ? 0xffffffffu : 0;

		int mask = (v[0] >> 31) |
						 ((v[1] >> 31) << 1) |
						 ((v[2] >> 31) << 2) |
						 ((v[3] >> 31) << 3);
		return mask == 0b1111;
#endif // USE_SIMD_SSE
	}


	inline VX_FORCE_INLINE float Vec4::Dot(const Vec4& rhs) const
	{
#ifdef VX_USE_SSE
		/// 0x71 -> 0111 0001 : op first 3 & store 1 (first)
		/// 0xf1 -> 1111 0001 : op first 4 & store 1 (first)
		/// 
		/// 0x77 -> 0111 0111 : op first 3 & store 3 (first)
		/// 0xff -> 1111 1111 : op first 4 & store 4 (first)
		/// 
		/// 0x7f -> 0111 1111 : op first 4 & store 4 (first)
		/// 
		/// as 0111 0001 
		/// high nibble 0111 [bit 4 - 7] (nibble 1 = 4bits, 0.5bytes) 
		/// low nibble 0001	 [bit 0 - 3]
		/// using with _mm_dp_ps 
		/// high nibble defines, the bits to op on (multply its components) 
		/// 0111 x, y, z, without w 
		/// low nibble defines, the bits to store result
		/// 0001 only x excluding y, z, and w

		//dot product op first 4 & store 1 (x) then extract 1 (0:x)
		return _mm_cvtss_f32(_mm_dp_ps(mValue, rhs.mValue, 0xf1));
#else
		float dot = 0.0f;
		for (int i = 0; i < 4; ++i)
			dot += (mFloat[i] * rhs.mFloat[i]);
		return dot;
#endif // USE_SIMD_SSE

	}

	inline VX_FORCE_INLINE float Vec4::Dot(const Vec4& lhs, const Vec4& rhs)
	{
#ifdef VX_USE_SSE
		/// 0x71 -> 0111 0001 : op first 3 & store 1 (first)
		/// 0xf1 -> 1111 0001 : op first 4 & store 1 (first)
		/// 
		/// 0x77 -> 0111 0111 : op first 3 & store 3 (first)
		/// 0xff -> 1111 1111 : op first 4 & store 4 (first)
		/// 
		/// 0x7f -> 0111 1111 : op first 3 & store 4 (first)
		/// 
		/// as 0111 0001 
		/// high nibble 0111 [bit 4 - 7] (nibble 1 = 4bits, 0.5bytes) 
		/// low nibble 0001	 [bit 0 - 3]
		/// using with _mm_dp_ps 
		/// high nibble defines, the bits to op on (multply its components) 
		/// 0111 x, y, z, without w 
		/// low nibble defines, the bits to store result
		/// 0001 only x excluding y, z, and w

		//dot product op first 4 & store 1 (x) then extract 1 (0:x)
		return _mm_cvtss_f32(_mm_dp_ps(lhs.mValue, rhs.mValue, 0xf1));
#else
		float dot = 0.0f;
		for (int i = 0; i < 4; ++i)
			dot += (lhs.mFloat[i] * rhs.mFloat[i]);
		return dot;
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE float Vec4::LengthSq() const
	{
#ifdef VX_USE_SSE
		return _mm_cvtss_f32(_mm_dp_ps(mValue, mValue, 0xf1));
#else
		float len_sq = 0.0f;
		for (int i = 0; i < 4; ++i)
			len_sq += (mFloat[i] * mFloat[i]);
		return len_sq;
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE float Vec4::Length() const
	{
#ifdef VX_USE_SSE
		return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(mValue, mValue, 0xf1)));
#else
		float len_sq = 0.0f;
		for (int i = 0; i < 4; ++i)
			len_sq += (mFloat[i] * mFloat[i]);
		return std::sqrt(len_sq);
#endif // USE_SIMD_SSE
	}



	inline VX_FORCE_INLINE Vec4 Vec4::Normalised() const
	{
#ifdef VX_USE_SSE
		/// 0x71 -> 0111 0001 : op first 3 & store 1 (first)
		/// 0xf1 -> 1111 0001 : op first 4 & store 1 (first)
		/// 
		/// 0x77 -> 0111 0111 : op first 3 & store 3 (first)
		/// 0xff -> 1111 1111 : op first 4 & store 4 (first)
		/// 
		/// 0x7f -> 0111 1111 : op first 3 & store 4 (first)
		/// _mm_dp_ps(value, value, 0xff); //<-- dot4 [op all & store all]
		/// _mm_sqrt_ps <-- sqrt of result
		/// div vale
		//return _mm_div_ps(value, _mm_sqrt_ps(_mm_dp_ps(value, value, 0xff)));

		__m128 dot = _mm_dp_ps(mValue, mValue, 0xff);
		__m128 safe_ep = _mm_max_ps(dot, _mm_set_ps1(1e-6f));
		return _mm_div_ps(mValue, _mm_sqrt_ps(safe_ep));
#else
		Vec4 result = *this;
		float length_sq = 0.0f;
		for (int i = 0; i < 4; ++i)
			length_sq += (mFloat[i] * mFloat[i]);
		if (length_sq > 1e-6f)
		{
			const float inv = 1.0f / std::sqrt(length_sq);
			for (int i = 0; i < 4; ++i)
				result.mFloat[i] *= inv;
		}
		return result;
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE Vec4& Vec4::Normalise()
	{
#ifdef VX_USE_SSE
		/// 0x71 -> 0111 0001 : op first 3 & store 1 (first)
		/// 0xf1 -> 1111 0001 : op first 4 & store 1 (first)
		/// 
		/// 0x77 -> 0111 0111 : op first 3 & store 3 (first)
		/// 0xff -> 1111 1111 : op first 4 & store 4 (first)
		/// 
		/// 0x7f -> 0111 1111 : op first 3 & store 4 (first)
		/// _mm_dp_ps(value, value, 0xff); //<-- dot4 [op all & store all]
		/// _mm_sqrt_ps <-- sqrt of result
		/// div vale
		//return _mm_div_ps(value, _mm_sqrt_ps(_mm_dp_ps(value, value, 0xff)));

		__m128 dot = _mm_dp_ps(mValue, mValue, 0xff);
		__m128 safe_ep = _mm_max_ps(dot, _mm_set_ps1(1e-6f));
		mValue = _mm_div_ps(mValue, _mm_sqrt_ps(safe_ep));
		//single lane div
		safe_ep = _mm_div_ss(_mm_set_ps1(1.0f), safe_ep);

		//mValue = _mm_mul_ps(mValue, _mm_shuffle_ps(safe_ep, safe_ep, _MM_SHUFFLE(0, 0, 0, 0)));
#else
		float length_sq = 0.0f;
		for (int i = 0; i < 4; ++i)
			length_sq += (mFloat[i] * mFloat[i]);
		if (length_sq > 1e-6f)
		{
			const float inv = 1.0f / std::sqrt(length_sq);
			for (int i = 0; i < 4; ++i)
				mFloat[i] *= inv;
		}
#endif // USE_SIMD_SSE


		return *this;
	}

	inline VX_FORCE_INLINE Vec4 Vec4::Inverted() const
	{
#ifdef VX_USE_SSE
		//return _mm_mul_ps(mValue, _mm_set_ps1(-1.0f)); <-- extra cycle
		return _mm_xor_ps(mValue, _mm_set_ps1(-0.0f)); //<-- 1 cycle (bitwise)
#else
		return Vec4(-mFloat[0], -mFloat[1], -mFloat[2], -mFloat[3]);
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE Vec4& Vec4::Invert()
	{
#ifdef VX_USE_SSE
		mValue = _mm_xor_ps(mValue, _mm_set_ps1(-0.0f));
#else
		for (int i = 0; i < 4; ++i)
			mFloat[i] = -mFloat[i];
#endif // USE_SIMD_SSE

		return *this;
	}

	VX_FORCE_INLINE Vec4 Vec4::Reciprocal() const
	{
		return One() / mValue;
	}

	inline VX_FORCE_INLINE Vec3 Vec4::XYZ() const
	{
#ifdef VX_USE_SSE
		return _mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(2, 2, 1, 0));
#else
		return Vec3(mFloat[0], mFloat[1], mFloat[2]);
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE Vec4 Vec4::XYZZ() const
	{
#ifdef VX_USE_SSE
		return _mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(2, 2, 1, 0));
#else
		return Vec3(mFloat[0], mFloat[1], mFloat[2], mFloat[2]);
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE Vec4 Vec4::XYZ0() const
	{
		Vec4 result = *this;
		result[3] = 0.0f;
		return result;
	}

	inline VX_FORCE_INLINE Vec4 Vec4::XYZ1() const
	{
		Vec4 result = *this;
		result[3] = 1.0f;
		return result;
	}

	inline VX_FORCE_INLINE Vec4 Vec4::Broadcast(float scalar)
	{
#ifdef VX_USE_SSE
		return _mm_set_ps1(scalar);
#else
		return Vec4(scalar, scalar, scalar, scalar);
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE Vec4 Vec4::SplatX() const
	{
#ifdef VX_USE_SSE
		return _mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(0, 0, 0, 0));
#else
		return Vec3(mFloat[0], mFloat[0], mFloat[0], mFloat[0]);
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE Vec4 Vec4::SplatY() const
	{
#ifdef VX_USE_SSE
		return _mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(1, 1, 1, 1));
#else
		return Vec3(mFloat[1], mFloat[1], mFloat[1], mFloat[1]);
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE Vec4 Vec4::SplatZ() const
	{
#ifdef VX_USE_SSE
		return _mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(2, 2, 2, 2));
#else
		return Vec3(mFloat[2], mFloat[2], mFloat[2], mFloat[2]);
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE Vec4 Vec4::SplatW() const
	{
#ifdef VX_USE_SSE
		return _mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(3, 3, 3, 3));
#else
		return Vec3(mFloat[3], mFloat[3], mFloat[3], mFloat[3]);
#endif // USE_SIMD_SSE
	}

	inline VX_FORCE_INLINE Vec3 Vec4::Cross3(const Vec4& lhs, const Vec4& rhs)
	{
#ifdef VX_USE_SSE
		/// y * z - z * y
		/// z * x - x * z
		/// x * y - y * x
		/// 
		/// 
		/// x, y, z   [3210]lhs / rhs
		/// y, z, x   [3021] shuffle order 
		/// 
		/// vl = lhs * shuffled rhs
		/// vr = rhs * shuffled lhs
		/// 
		/// vl - vr
		/// preserve w all through
		__m128 vl = _mm_shuffle_ps(rhs.mValue, rhs.mValue, _MM_SHUFFLE(3, 0, 2, 1));
		/// shuffled y z x w :- right
		vl = _mm_mul_ps(lhs.mValue, vl);
		/// xy yz zx ww  [l first r second] 
		__m128 vr = _mm_shuffle_ps(lhs.mValue, lhs.mValue, _MM_SHUFFLE(3, 0, 2, 1));
		/// shuffled y z x w :- left
		vr = _mm_mul_ps(rhs.mValue, vr);
		/// xy yz zx ww  [r first l second] (yx zy zx ww) 
		__m128 r = _mm_sub_ps(vl, vr);
		/// result anitsymmetric xy yz zx ww 
		/// required (yz) (zx) (xy)
		/// shuf  ->  y->x, z->y, x->z  3 0 2 1
		return _mm_shuffle_ps(r, r, _MM_SHUFFLE(3, 0, 2, 1));

		/// yz - zy, 
		/// xz - zx
		/// xy - yx
		/// 
		/// x, y, z
		/// 
#else
		Vec3 temp = Vec3(0.0f);
		temp.mFloat[0] = (lhs.Y() * rhs.Z()) - (rhs.Y() * lhs.Z());
		temp.mFloat[1] = (rhs.X() * lhs.Z()) - (lhs.X() * rhs.Z());
		temp.mFloat[2] = (lhs.X() * rhs.Y()) - (rhs.X() * lhs.Y());
		return temp;
#endif // USE_SIMD_SSE
	}
}