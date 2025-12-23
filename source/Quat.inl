#include "Quat.h"

namespace vx
{
	VX_INLINE void Quat::Normalise()
	{
		float len = mValue.Length();

		if (len < kEpsilon)
		{
			mValue.SetW(1.0f);
			return;
		}

#ifdef VX_USE_SSE
		mValue /= len; //<- sse path div
#else
		len = 1.0f / len; //<- scalar inv 
		mValue *= len;	//<- scalar mul
#endif // VX_USE_SSE
	}

	inline VX_INLINE Quat Quat::Normalised() const
	{
		Quat q(*this);
		q.Normalise();
		return q;
	}

	inline VX_INLINE bool Quat::IsUnitQuat(float tolerance) const
	{
		return VxAbs(LengthSq() - 1.0f) <= tolerance;
	}

	inline VX_INLINE Quat Quat::Inversed() const
	{
		return Conjugated() / Length();
	}

	inline VX_INLINE Vec3 Quat::Rotate(const Vec3& vec) const
	{
		VX_ASSERT(IsUnitQuat(), "Quaternion is not unit");
		/// v' = q * q(v) * q^-1
		Quat qv(vec, 0.0f);
		Quat v = (*this) * qv * Conjugated();
		return Vec3(v.X(), v.Y(), v.Z());
	}

	inline VX_INLINE Quat Quat::operator*(const Quat& rhs) const
	{
		Quat r(*this);
		r *= rhs;
		return r;
	}

	inline VX_INLINE Quat Quat::operator*=(const Quat& rhs)
	{


#ifdef VX_USE_SSE
		__m128 q1 = mValue.Value();
		__m128 q2 = rhs.mValue.Value();

		//w
		__m128 w1 = mValue.SplatW().Value();
		__m128 w2 = mValue.SplatW().Value();

		__m128 v = _mm_add_ps(
			_mm_mul_ps(w1, q2),
			_mm_mul_ps(w2, q1)
		);

		//cross v1, v2
		__m128 v1_yzx = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(3, 0, 2, 1));
		__m128 v2_zxy = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(3, 1, 0, 2));
		__m128 v1_zxy = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(3, 1, 0, 2));
		__m128 v2_yzx = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(3, 0, 2, 1));

		__m128 cross = _mm_sub_ps(
			_mm_mul_ps(v1_yzx, v2_zxy),
			_mm_mul_ps(v1_zxy, v2_yzx)
		);

		//xyz = w1*v2+w2*v1+cross
		__m128 xyz = _mm_add_ps(v, cross);

		//w = w1*w2 - dot(v1,v2)
		//0x7f -> 0111 1111 : op first 3 & store 4 (first)
		__m128 dot = _mm_dp_ps(q1, q2, 0x7f);
		__m128 w = _mm_sub_ps(_mm_mul_ps(w1, w2), dot);

		//xyz +w
		mValue = _mm_blend_ps(xyz, w, 0b1000);

		return *this;

#else

		float rx = rhs.X();
		float ry = rhs.Y();
		float rz = rhs.Z();
		float rw = rhs.W();

		mValue Vec4(
			W() * rx + X() * rw + Y() * rz - Z() * ry,
			W() * ry - X() * rz + Y() * rw + Z() * rx,
			W() * rz + X() * ry - Y() * rx + Z() * rw,
			W() * rw - X() * rx - Y() * ry - Z() * rz
		);

		return *this;
#endif // VX_USE_SSE
	}

}