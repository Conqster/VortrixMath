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
		/// rotate fast
		/// v' = q * q(v) * q^-1
		///			  = q * q(v.xyz, 0.0f) * q^-1
		///
		/// rodrigues:
		///  x: cross, *: dot
		/// v'= v + 2 * w * (q.xyz x v) + 2 * (q.xyz x (q.xyz x v))
		/// simplify 
		/// 
		/// let t = (q.xyz x v)
		/// v'= v + 2 * w * t + 2 * (q.xyz x t)
		/// 
		/// 
		/// t = 2(q.xyz x v)
		/// t = (q.xyz x v) + (q.xyz x v)
		/// i.e t + t  = (q.xyz x v)
		/// 
		/// v' = v +  w * t + (q.xyz x t)
		/// v' = v + wt + (q.xyz x t)
		/// 
		/// 

		Vec3 q_xyz = Imaginary();
		Vec3 t = q_xyz.Cross(vec);
		t += t;
		return vec + W() * t + q_xyz.Cross(t);
	}

	inline VX_INLINE Vec3 Quat::InverseRotate(const Vec3& vec) const
	{
		VX_ASSERT(IsUnitQuat(), "Quaternion is not unit");
		Vec3 q_xyz = Imaginary();
		q_xyz.FlipSignAssign<-1, -1, -1>();
		Vec3 t = q_xyz.Cross(vec);
		t += t;
		return vec + W() * t + q_xyz.Cross(t);
	}

	inline VX_INLINE Vec3 Quat::RotateSlow(const Vec3& vec) const
	{
		VX_ASSERT(IsUnitQuat(), "Quaternion is not unit");
		/// v' = q * q(v) * q^-1
		Quat qv(vec, 0.0f);
		Quat v = (*this) * qv * Conjugated();
		return v.Imaginary();
	}

	inline VX_INLINE Vec3 Quat::InverseRotateSlow(const Vec3& vec) const
	{
		VX_ASSERT(IsUnitQuat(), "Quaternion is not unit");
		/// v' = q^-1 * q(v) * q
		Quat qv(vec, 0.0f);
		Quat v = Conjugated() * qv * (*this);
		return v.Imaginary();
	}

	inline VX_INLINE Vec3 Quat::RotateAxisX() const
	{
		VX_ASSERT(IsUnitQuat(), "Quaternion is not unit");
		return RotateScaledAxisX(1.0f);

		float x = X(), y = Y(), z = Z(), w = W();

		float tx = x + x;
		float ty = y + y;
		float tz = z + z;

		//float xx = x * tx;
		float yy = ty * y;
		float zz = tz * z;

		float xy = tx * y; //ty * x;
		float zw = tz * w;

		float xz = tx * z;
		float yw = ty * w;


		return Vec3(1.0f - (yy + zz), xy + zw, xz - yw);
	}

	inline VX_INLINE Vec3 Quat::RotateAxisY() const
	{
		VX_ASSERT(IsUnitQuat(), "Quaternion is not unit");
		float x = X(), y = Y(), z = Z(), w = W();

		float tx = x + x;
		float ty = y + y;
		float tz = z + z;

		float xy = tx * y; //ty * x;
		float zw = tz * w;

		float zz = tz * z;
		float xx = tx * x;

		float yz = ty * z;
		float xw = tx * w;

		return Vec3(xy - zw, (1.0f - zz) - xx, yz + xw);
	}

	inline VX_INLINE Vec3 Quat::RotateAxisZ() const
	{
		VX_ASSERT(IsUnitQuat(), "Quaternion is not unit");
		float tx = X() + X();
		float ty = Y() + Y();
		float tz = Z() + Z();

		float xz = tx * Z(); //ty * x;
		float yw = ty * W();

		float yz = ty * Z();
		float xw = tx * W();

		float xx = tx * X();
		float yy = ty * Y();

		return Vec3(xz + yw, yz - xw, (1.0f - xx) - yy);
	}

	inline VX_INLINE Vec3 Quat::RotateScaledAxisX(float scale) const
	{
		VX_ASSERT(IsUnitQuat(), "Quaternion is not unit");
		float x = X(), y = Y(), z = Z(), w = W();

		float tx = x + x;
		float ty = y + y;
		float tz = z + z;

		//float xx = x * tx;
		float yy = ty * y;
		float zz = tz * z;

		float xy = tx * y; //ty * x;
		float zw = tz * w;

		float xz = tx * z;
		float yw = ty * w;

		return Vec3(scale * 1.0f - (yy + zz),
					scale * xy + zw, 
					scale * xz - yw);
	}

	inline VX_INLINE Vec3 Quat::RotateScaledAxisY(float scale) const
	{
		VX_ASSERT(IsUnitQuat(), "Quaternion is not unit");
		float x = X(), y = Y(), z = Z(), w = W();

		float tx = x + x;
		float ty = y + y;
		float tz = z + z;

		float xy = tx * y; //ty * x;
		float zw = tz * w;

		float zz = tz * z;
		float xx = tx * x;

		float yz = ty * z;
		float xw = tx * w;

		return Vec3(scale * (xy - zw), scale * ((1.0f - zz) - xx), scale * (yz + xw));
	}

	inline VX_INLINE Vec3 Quat::RotateScaledAxisZ(float scale) const
	{
		VX_ASSERT(IsUnitQuat(), "Quaternion is not unit");
		float tx = X() + X();
		float ty = Y() + Y();
		float tz = Z() + Z();

		float xz = tx * Z(); //ty * x;
		float yw = ty * W();

		float yz = ty * Z();
		float xw = tx * W();

		float xx = tx * X();
		float yy = ty * Y();

		return Vec3(scale * (xz + yw), scale * (yz - xw), scale * ((1.0f - xx) - yy));
	}

	inline VX_INLINE void Quat::RotateScaledAxes(const Vec3& scale, Vec3& out_x, Vec3& out_y, Vec3& out_z)
	{
		VX_ASSERT(IsUnitQuat(), "Quaternion is not unit");

		//| [1 - 2(yy - zz)] [2(xy - wz)] [2(xz+wy)] |
		//| [2(xy + wz)]   [1-2(xx - zz)] [2(yz - wx)]|
		//| [2(xz - wy)] [2(yz + wz)] [1 - 2(xx - yy)]|
		//||

#ifdef VX_USE_SSE

		__m128 q = mValue.Value();
		__m128 s = scale.Value();

		__m128 two = _mm_add_ps(q, q);
		__m128 t = _mm_mul_ps(two, q);


		/// diag [yy + zz, xx + zz, xx + yy, 0]
		__m128 diag = _mm_add_ps(simd::Swizzle<1, 0, 0, 3>(t), simd::Swizzle<2, 2, 1, 3>(t));
		/// [1-(yy + zz), 1-(xx + zz), 1-(xx + yy)]
		diag = _mm_sub_ps(_mm_set1_ps(1.0f), diag);

		__m128 xyz = _mm_mul_ps(simd::Swizzle<0, 0, 1, 3>(two), simd::Swizzle<1, 2, 2, 3>(q));
		__m128 w_xyz = _mm_mul_ps(simd::Swizzle<3, 3, 3, 3>(two), simd::Swizzle<2, 1, 0, 3>(q));

		
		__m128 r = _mm_add_ps(simd::Swizzle<0, 0, 0, 0>(diag, xyz),
				simd::FlipSign<1, 1, -1, 1>(simd::Swizzle<1, 2, 0, 0>(xyz, w_xyz)));
		out_x = Vec3(_mm_mul_ps(r, simd::Swizzle<0, 0, 0, 0>(s)));

		r = _mm_add_ps(simd::Swizzle<1, 0, 1, 0>(xyz, diag),
				simd::FlipSign<-1, 1, 1, 1>(simd::Swizzle<2, 0, 2, 0>(w_xyz, xyz)));
		out_y = Vec3(_mm_mul_ps(r, simd::Swizzle<1, 1, 1, 1>(s)));

		r = _mm_add_ps(simd::Swizzle<0, 1, 1, 0>(w_xyz, xyz),
				simd::Swizzle<2, 2, 2, 0>(w_xyz, diag));
		out_z = Vec3(_mm_mul_ps(r, simd::Swizzle<2, 2, 2, 2>(s)));
				

#else
		float tx = X() + X();
		float ty = Y() + Y();
		float tz = Z() + Z();

		float xx = tx * X();
		float yy = ty * Y();
		float zz = tz * Z();

		float xy = tx * Y();
		float xz = tx * Z();
		float xw = tx * W();

		float yz = ty * Z();
		float yw = ty * W();
		float zw = tz * W();

		/// basis axes
		out_x = Vec3(scale.X() * (1.0f - (yy + zz)), scale.X() * (xy + zw), scale.X() * (xz - yw));
		out_y = Vec3(scale.Y() * (xy - zw), scale.Y() * ((1.0f - zz) - xx), scale.Y() * (yz + xw));
		out_z = Vec3(scale.Z() * (xz + yw), scale.Z() * (yz - xw), scale.Z() * ((1.0f - xx) - yy));
#endif // VX_USE_SSE
	}

	inline VX_INLINE Mat44 Quat::GetRotationMat44()
	{
#ifdef VX_USE_SSE


		__m128 q = mValue.Value();
		__m128 two = _mm_add_ps(q, q);
		__m128 t = _mm_mul_ps(two, q);


		/// diag [yy + zz, xx + zz, xx + yy, 0]
		__m128 diag = _mm_add_ps(simd::Swizzle<1, 0, 0, 3>(t), simd::Swizzle<2, 2, 1, 3>(t));
		/// [1-(yy + zz), 1-(xx + zz), 1-(xx + yy)]
		diag = _mm_sub_ps(_mm_set1_ps(1.0f), diag);

		__m128 xyz = _mm_mul_ps(simd::Swizzle<0, 0, 1, 3>(two), simd::Swizzle<1, 2, 2, 3>(q));
		__m128 w_xyz = _mm_mul_ps(simd::Swizzle<3, 3, 3, 3>(two), simd::Swizzle<2, 1, 0, 3>(q));


		__m128 c0 = _mm_add_ps(simd::Swizzle<0, 0, 0, 0>(diag, xyz),
					simd::FlipSign<1, 1, -1, 1>(simd::Swizzle<1, 2, 0, 0>(xyz, w_xyz)));

		__m128 c1 = _mm_add_ps(simd::Swizzle<1, 0, 1, 0>(xyz, diag),
				simd::FlipSign<-1, 1, 1, 1>(simd::Swizzle<2, 0, 2, 0>(w_xyz, xyz)));

		__m128 c2 = _mm_add_ps(simd::Swizzle<0, 1, 1, 0>(w_xyz, xyz),
				simd::Swizzle<2, 2, 2, 0>(w_xyz, diag));

		//ensure affine bottom & translate 0, 0,0 1
		__m128 mask = simd::LaneMask<-1, -1, -1, 0>();
	
		return Mat44(Vec4(_mm_and_ps(c0, mask)),
					 Vec4(_mm_and_ps(c1, mask)),
					 Vec4(_mm_and_ps(c2, mask)));

#else

		float tx = X() + X();
		float ty = Y() + Y();
		float tz = Z() + Z();

		float xx = tx * X();
		float yy = ty * Y();
		float zz = tz * Z();

		float xy = tx * Y();
		float xz = tx * Z();
		float xw = tx * W();

		float yz = ty * Z();
		float yw = ty * W();
		float zw = tz * W();

		/// basis axes
		return Mat44(Vec4(1.0f - (yy + zz), xy + zw, xz - yw, 0.0f),
			Vec4(xy - zw, (1.0f - zz) - xx, yz + xw, 0.0f),
			Vec4(xz + yw, yz - xw, (1.0f - xx) - yy, 0.0f));
#endif // VX_USE_SSE
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