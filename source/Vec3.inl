#include "Vec3.h"


inline FORCE_INLINE float Vec3::GetLane(const Vec3& v, int idx)
{
#if USE_SIMD_SSE
	return Get_m128Lane(v.mValue, idx);
#else
	return v.mData32[idx];
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE Vec3 Vec3::Zero()
{
#if USE_SIMD_SSE
	return Vec3(_mm_setzero_ps());
#else
	return Vec3(0.0f);
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE void Vec3::ToZero()
{
#if USE_SIMD_SSE
	mValue = _mm_setzero_ps();
#else
	for (int i = 0; i < 4; i++)
		mData32[i] = 0.0f;
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE Vec3 Vec3::Broadcast(float scalar)
{
#if USE_SIMD_SSE
	return _mm_set_ps1(scalar);
#else
	return Vec3(scalar, scalar, scalar);
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE Vec3 Vec3::LoadAligned(const float* v)
{
#if USE_SIMD_SSE
	return _mm_load_ps(v);
#else
	return Vec3(v[0], v[1], v[2]);
#endif // USE_SIMD_SSE

}

inline FORCE_INLINE Vec3 Vec3::Load(const float* v)
{
#if USE_SIMD_SSE
	return _mm_loadu_ps(v);
#else
	return Vec3(v[0], v[1], v[2]);
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE void Vec3::Store(float* o_v) const
{
#if USE_SIMD_SSE
	_mm_store_ps(o_v, mValue);
#else
#warning not store not support without SIMD
	//return Vec4(v[0], v[1], v[2], v[3]);
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE Vec3 Vec3::operator+(const Vec3& rhs) const
{
#if USE_SIMD_SSE
	return _mm_add_ps(mValue, rhs.mValue);
#else
	return Vec3(mData32[0] + rhs.mData32[0],
		mData32[1] + rhs.mData32[1],
		mData32[2] + rhs.mData32[2]);
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE Vec3& Vec3::operator+=(const Vec3& rhs)
{
#if USE_SIMD_SSE
	mValue = _mm_add_ps(mValue, rhs.mValue);
#else
	for (int i = 0; i < 3; ++i)
		mData32[i] += rhs.mData32[i];
#endif // USE_SIMD_SSE

	return *this;
}

inline FORCE_INLINE Vec3 Vec3::operator-(const Vec3& rhs) const
{
#if USE_SIMD_SSE
	return _mm_sub_ps(this->mValue, rhs.mValue);
#else
	return Vec3(mData32[0] - rhs.mData32[0],
		mData32[1] - rhs.mData32[1],
		mData32[2] - rhs.mData32[2]);
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE Vec3& Vec3::operator-=(const Vec3& rhs)
{
#if USE_SIMD_SSE
	mValue = _mm_sub_ps(this->mValue, rhs.mValue);
#else
	for (int i = 0; i < 3; ++i)
		mData32[i] -= rhs.mData32[i];
#endif // USE_SIMD_SSE

	return *this;
}

inline FORCE_INLINE Vec3 Vec3::operator*(const float scalar) const
{
#if USE_SIMD_SSE
	//broad cast or load scalar
	return _mm_mul_ps(mValue, _mm_set_ps1(scalar));
#else
	return Vec4(mData32[0] * scalar,
		mData32[1] * scalar,
		mData32[2] * scalar);
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE Vec3& Vec3::operator*=(const float scalar)
{
#if USE_SIMD_SSE
	mValue = _mm_mul_ps(mValue, _mm_set_ps1(scalar));
#else
	for (int i = 0; i < 3; ++i)
		mData32[i] *= scalar;
#endif // USE_SIMD_SSE

	return *this;
}

inline FORCE_INLINE Vec3 Vec3::Divide(const float scalar)
{
	Vec3 t = *this;
	const float inv = 1.0f / scalar;
	for (int i = 0; i < 3; ++i)
		t.mData32[i] *= inv;
	return t;
}

inline Vec3 Vec3::operator/(const float scalar) const
{
#if USE_SIMD_SSE
	//return _mm_div_ps(value, _mm_set_ps1(scalar));//<- expensive div per lane
	//return _mm_mul_ps(value, _mm_set_ps1(1.0f /scalar)); <- loss precision due 1/s, most case fastest 

	//one lane division, shuffle x lane across, then mul
	__m128 v = _mm_div_ss(_mm_set_ps1(1.0f), _mm_set_ps1(scalar));
	return _mm_mul_ps(mValue, _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)));
	//return _mm_mul_ps(mValue, _mm_set_ps1(scalar));
#else
	return Vec3(mData32[0] / scalar,
		mData32[1] / scalar,
		mData32[2] / scalar);
#endif // USE_SIMD_SSE
}

inline Vec3& Vec3::operator/=(const float scalar)
{
#if USE_SIMD_SSE
	//return _mm_div_ps(value, _mm_set_ps1(scalar));//<- expensive div per lane
	//return _mm_mul_ps(value, _mm_set_ps1(1.0f /scalar)); <- loss precision due 1/s, most case fastest 

	//one lane division, shuffle x lane across, then mul
	__m128 v = _mm_div_ss(_mm_set_ps1(1.0f), _mm_set_ps1(scalar));
	mValue = _mm_mul_ps(mValue, _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)));
#else
	for (int i = 0; i < 3; ++i)
		mData32[i] /= scalar;
#endif // USE_SIMD_SSE

	return *this;
}

inline FORCE_INLINE float Vec3::MinComponent() const
{
#if USE_SIMD_SSE
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
	return (X() < Y()) ? ((X() < Z()) ? mData32[0] : mData32[2]) : (Y() < Z()) ? mData32[1] : mData32[2];
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE float Vec3::MaxComponent() const
{
#if USE_SIMD_SSE
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
	return (X() > Y()) ? ((X() > Z()) ? mData32[0] : mData32[2]) : (Y() > Z()) ? mData32[1] : mData32[2];
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE int Vec3::MinAxis() const
{
	return (mData32[0] < mData32[1]) ? ((mData32[0] < mData32[2]) ? 0 : 2) : (mData32[1] < mData32[2]) ? 1 : 2;
}

inline FORCE_INLINE int Vec3::MaxAxis() const
{
	return (mData32[0] > mData32[1]) ? ((mData32[0] > mData32[2]) ? 0 : 2) : (mData32[1] > mData32[2]) ? 1 : 2;
}


inline FORCE_INLINE Vec3 Vec3::Min(const Vec3& lhs, const Vec3& rhs)
{
#if USE_SIMD_SSE
	return _mm_min_ps(lhs.mValue, rhs.mValue);
#else
	return Vec3(std::min(lhs.mData32[0], rhs.mData32[0]),
		std::min(lhs.mData32[1], rhs.mData32[1]),
		std::min(lhs.mData32[2], rhs.mData32[2]));
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE Vec3 Vec3::Max(const Vec3& lhs, const Vec3& rhs)
{
#if USE_SIMD_SSE
	return _mm_max_ps(lhs.mValue, rhs.mValue);
#else
	return Vec3(std::max(lhs.mData32[0], rhs.mData32[0]),
		std::max(lhs.mData32[1], rhs.mData32[1]),
		std::max(lhs.mData32[2], rhs.mData32[2]));
#endif // USE_SIMD_SSE

}

inline FORCE_INLINE float Vec3::Dot(const Vec3& rhs) const
{
#if USE_SIMD_SSE
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

	//dot product op first 3 & store 1 (x) then extract 1 (0:x)
	return _mm_cvtss_f32(_mm_dp_ps(mValue, rhs.mValue, 0x71));
#else
	float dot = 0.0f;
	for (int i = 0; i < 3; ++i)
		dot += (mData32[i] * rhs.mData32[i]);
	return dot;
#endif // USE_SIMD_SSE

}

inline FORCE_INLINE float Vec3::Dot(const Vec3& lhs, const Vec3& rhs)
{
#if USE_SIMD_SSE
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

	//dot product op first 3 & store 1 (x) then extract 1 (0:x) 0x71 ->0111 0001
	return _mm_cvtss_f32(_mm_dp_ps(lhs.mValue, rhs.mValue, 0x71));
#else
	float dot = 0.0f;
	for (int i = 0; i < 3; ++i)
		dot += (lhs.mData32[i] * rhs.mData32[i]);
	return dot;
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE float Vec3::LengthSq() const
{
#if USE_SIMD_SSE
	//dot product op first 3 & store 1 (x) then extract 1 (0:x) 0x71 ->0111 0001
	return _mm_cvtss_f32(_mm_dp_ps(mValue, mValue, 0x71));
#else
	float len_sq = 0.0f;
	for (int i = 0; i < 3; ++i)
		len_sq += (mData32[i] * mData32[i]);
	return len_sq;
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE float Vec3::Length() const
{
#if USE_SIMD_SSE
	//dot product op first 3 & store 1 (x) then extract 1 (0:x) 0x71 ->0111 0001
	return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(mValue, mValue, 0x71)));
#else
	float len_sq = 0.0f;
	for (int i = 0; i < 3; ++i)
		len_sq += (mData32[i] * mData32[i]);
	return std::sqrt(len_sq);
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE Vec3 Vec3::Normalised_NOT_SIMD() const
{
	float length_sq = (X() * X() + Y() * Y() + Z() * Z());
	Vec3 result = *this;
	if (length_sq > 1e-6f)
	{
		const float inv = 1.0f / std::sqrt(length_sq);
		for (int i = 0; i < 3; ++i)
			result.mData32[i] *= inv;
	}
	return result;
}

inline FORCE_INLINE Vec3 Vec3::Normalised() const
{
#if USE_SIMD_SSE
	/// 0x7f -> 0111 1111 : op first 3 & store 4 (first)
	///         0zyx dddd 
	__m128 dot = _mm_dp_ps(mValue, mValue, 0x7f);
	__m128 safe_ep = _mm_max_ps(dot, _mm_set_ps1(1e-6f));
	return _mm_div_ps(mValue, _mm_sqrt_ps(safe_ep));
#else
	Vec4 result = *this;
	float length_sq = 0.0f;
	for (int i = 0; i < 3; ++i)
		length_sq += (mData32[i] * mData32[i]);
	if (length_sq > 1e-6f)
	{
		const float inv = 1.0f / std::sqrt(length_sq);
		for (int i = 0; i < 3; ++i)
			result.mData32[i] *= inv;
	}
	return result;
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE Vec3& Vec3::Normalise()
{
#if USE_SIMD_SSE
	/// 0x7f -> 0111 1111 : op first 3 & store 4 (first)
	///         0zyx dddd 
	__m128 dot = _mm_dp_ps(mValue, mValue, 0x7f);
	__m128 safe_ep = _mm_max_ps(dot, _mm_set_ps1(1e-6f));
	mValue = _mm_div_ps(mValue, _mm_sqrt_ps(safe_ep));
#else
	float length_sq = 0.0f;
	for (int i = 0; i < 3; ++i)
		length_sq += (mData32[i] * mData32[i]);
	if (length_sq > 1e-6f)
	{
		const float inv = 1.0f / std::sqrt(length_sq);
		for (int i = 0; i < 3; ++i)
			mData32[i] *= inv;
	}
#endif // USE_SIMD_SSE

	return *this;
}

inline FORCE_INLINE Vec3 Vec3::Inverted() const
{
#if USE_SIMD_SSE
	//return _mm_mul_ps(mValue, _mm_set_ps1(-1.0f)); <-- extra cycle
	return _mm_xor_ps(mValue, _mm_set_ps1(-0.0f)); //<-- 1 cycle (bitwise)
#else
	return Vec3(-mData32[0], -mData32[1], -mData32[2]);
#endif // USE_SIMD_SSE
}

inline FORCE_INLINE Vec3& Vec3::Invert()
{
#if USE_SIMD_SSE
	mValue = _mm_xor_ps(mValue, _mm_set_ps1(-0.0f));
#else
	for (int i = 0; i < 3; ++i)
		mData32[i] = -mData32[i];
#endif // USE_SIMD_SSE

	return *this;
}

inline FORCE_INLINE Vec3 Vec3::Cross_NOT_SIMD(const Vec3& lhs, const Vec3& rhs)
{
	Vec3 temp = Vec3(0.0f);
	temp.mData32[0] = (lhs.Y() * rhs.Z()) - (rhs.Y() * lhs.Z());
	temp.mData32[1] = (rhs.X() * lhs.Z()) - (lhs.X() * rhs.Z());
	temp.mData32[2] = (lhs.X() * rhs.Y()) - (rhs.X() * lhs.Y());
	return temp;
}

inline FORCE_INLINE Vec3 Vec3::Cross(const Vec3& lhs, const Vec3& rhs)
{
#if USE_SIMD_SSE
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
#else
	return Vec3((lhs.Y() * rhs.Z()) - (rhs.Y() * lhs.Z()), 
		(rhs.X() * lhs.Z()) - (lhs.X() * rhs.Z()),
		(lhs.X() * rhs.Y()) - (rhs.X() * lhs.Y());
#endif // USE_SIMD_SSE
}
//
//inline FORCE_INLINE Vec3& Vec4::Cross3(const Vec4& rhs)
//{
//#if USE_SIMD_SSE
//	/// y * z - z * y
//	/// z * x - x * z
//	/// x * y - y * x
//	/// 
//	/// 
//	/// x, y, z   [3210]lhs / rhs
//	/// y, z, x   [3021] shuffle order 
//	/// 
//	/// vl = lhs * shuffled rhs
//	/// vr = rhs * shuffled lhs
//	/// 
//	/// vl - vr
//	/// preserve w all through
//	__m128 vl = _mm_shuffle_ps(rhs.mValue, rhs.mValue, _MM_SHUFFLE(3, 0, 2, 1));
//	/// shuffled y z x w :- right
//	vl = _mm_mul_ps(mValue, vl);
//	/// xy yz zx ww  [l first r second] 
//	mValue = _mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(3, 0, 2, 1));
//	/// shuffled y z x w :- left
//	mValue = _mm_mul_ps(rhs.mValue, mValue);
//	/// xy yz zx ww  [r first l second] (yx zy zx ww) 
//	mValue = _mm_sub_ps(vl, mValue);
//	/// result anitsymmetric xy yz zx ww 
//	/// required (yz) (zx) (xy)
//	/// shuf  ->  y->x, z->y, x->z  3 0 2 1
//	mValue = _mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(3, 0, 2, 1));
//
//	/// yz - zy, 
//	/// xz - zx
//	/// xy - yx
//	/// 
//	/// x, y, z
//	/// 
//	return *this;
//
//#else
//	Vec3 temp = Vec3(0.0f);
//	temp.mData32[0] = (Y() * rhs.Z()) - (rhs.Y() * Z());
//	temp.mData32[1] = (rhs.X() * Z()) - (X() * rhs.Z());
//	temp.mData32[2] = (X() * rhs.Y()) - (rhs.X() * Y());
//	return temp;
//#endif // USE_SIMD_SSE
//
//}
