#include "Mat44.h"

namespace vx
{
	Mat44::Mat44(const float diagonal)
	{
		//mCol[0] = Vec4(diagonal, 0.0f, 0.0f, 0.0f);
		//mCol[1] = Vec4(0.0f, diagonal, 0.0f, 0.0f);
		//mCol[2] = Vec4(0.0f, 0.0f, diagonal, 0.0f);
		//mCol[3] = Vec4(0.0f, 0.0f, 0.0f, diagonal);
		std::fill_n(mFloats, 16, 0.0f);
		mFloats[0] = mFloats[5] =
		mFloats[10] = mFloats[15] = diagonal;
	}

	inline Mat44::Mat44(const Vec4& col0, const Vec4& col1, const Vec4& col2, const Vec4& col3) :
		mCol {col0, col1, col2, col3}
	{}

	inline Mat44::Mat44(const Vec4 & col0, const Vec4 & col1, const Vec4 & col2) :
		mCol {col0, col1, col2, Vec4(0.0f, 0.0f, 0.0f, 1.0f)}
	{}

	inline VX_INLINE Mat44 Mat44::Identity()
	{
		return Mat44(1.0f);
	}

	inline VX_INLINE Mat44 Mat44::Dummy()
	{
		Mat44 result;
		for (size_t i = 0; i < 16; ++i)
			result.mFloats[i] = static_cast<float>(i);
		return result;
	}

	inline VX_INLINE Mat44 Mat44::Translation(const Vec3& v)
	{
		return Mat44(Vec4(1.0f, 0.0f, 0.0f, 0.0f), 
					 Vec4(0.0f, 1.0f, 0.0f, 0.0f),
					 Vec4(0.0f, 0.0f, 1.0f, 0.0f),
					 Vec4(v, 1.0f));
	}
	inline VX_INLINE Mat44 Mat44::Scale(float scale)
	{
		return Mat44(Vec4(scale, 0.0f, 0.0f, 0.0f),
					Vec4(0.0f, scale, 0.0f, 0.0f),
					Vec4(0.0f, 0.0f, scale, 0.0f),
					Vec4(0.0f, 0.0f, 0.0f, 1.0f));
	}

	inline VX_INLINE Mat44 Mat44::Scale(const Vec3& scale)
	{
		return Mat44(Vec4(scale.X(), 0.0f, 0.0f, 0.0f),
					 Vec4(0.0f, scale.Y(), 0.0f, 0.0f),
					 Vec4(0.0f, 0.0f, scale.Z(), 0.0f),
					 Vec4(0.0f, 0.0f, 0.0f, 1.0f));
	}

	inline VX_INLINE Mat44 Mat44::RotationX(float in_x_rad)
	{
		/// | R/S   R   R  Tx || R/s R  R Tx |
		/// |  R   cos -sin Ty ||  R  c  -s Ty |
		/// |  R   sin  cos Tz ||  R  s   c Tz |
		/// |  0    0    0   1 ||  0  0   0  1 |
		/// 
		float s = VxSin(in_x_rad);
		float c = VxCos(in_x_rad);

		//column major
		return Mat44(Vec4(1.0f, 0.0f, 0.0f, 0.0f),
			Vec4(0.0f, c, s, 0.0f),
			Vec4(0.0f, -s, c, 0.0f),
			Vec4(0.0f, 0.0f, 0.0f, 1.0f));
	}

	inline VX_INLINE Mat44 Mat44::RotationY(float in_y_rad)
	{
		/// |  cos  R  sin  Tx |
		/// |   R  R/S  R   Ty |
		/// | -sin  R  cos Tz |
		/// |   0   0   0   1 |
		/// 
		float s = VxSin(in_y_rad);
		float c = VxCos(in_y_rad);

		return Mat44(Vec4(c, 0.0f, -s, 0.0f),
			Vec4(0.0f, 1.0f, 0.0f, 0.0f),
			Vec4(s, 0.0f, c, 0.0f),
			Vec4(0.0f, 0.0f, 0.0f, 1.0f));
	}

	inline VX_INLINE Mat44 Mat44::RotationZ(float in_z_rad)
	{
		/// | cos -sin   R  Tx |
		/// | sin  cos   R  Ty |
		/// |  R    R   R/S Tz |
		/// |  0    0    0   1 |
		/// 
		float s = VxSin(in_z_rad);
		float c = VxCos(in_z_rad);

		return Mat44(Vec4(c, s,0.0f, 0.0f),
					 Vec4(-s, c, 0.0f, 0.0f),
					 Vec4(0.0f, 0.0f, 1.0f, 0.0f),
					 Vec4(0.0f, 0.0f, 0.0f, 1.0f));
	}

	inline VX_INLINE Mat44 Mat44::Basis(const Vec3& x, const Vec3& y, const Vec3& z)
	{
		return Mat44(Vec4(x, 0.0f),
			Vec4(y, 0.0f),
			Vec4(z, 0.0f),
			Vec4(0.0f, 0.0f, 0.0f, 1.0f));
	}

	inline VX_INLINE Mat44 Mat44::BasisTranslation(const Vec3& x, const Vec3& y, const Vec3& z, const Vec3& t)
	{
		return Mat44(Vec4(x, 0.0f),
					Vec4(y, 0.0f),
					Vec4(z, 0.0f),
					Vec4(t, 1.0f));
	}

	inline VX_INLINE float& Mat44::operator()(int row, int column)
	{
		VX_ASSERT(row < 4, "Row index out of bounds [0, 3]");
		VX_ASSERT(column < 4, "Column index out of bounds [0, 3]");

		return mCol[column][row];
	}

	inline VX_INLINE float Mat44::operator()(int row, int column) const
	{
		VX_ASSERT(row < 4, "Row index out of bounds [0, 3]");
		VX_ASSERT(column < 4, "Column index out of bounds [0, 3]");

		return mCol[column][row];
	}


	inline VX_INLINE Vec4 Mat44::GetColumn(int i) const
	{
		VX_ASSERT(i < 4, "Column index out of bounds [0, 3]");
		return mCol[i];
	}

	inline VX_INLINE Vec3 Mat44::GetColumn3(int i) const
	{
		VX_ASSERT(i < 4, "Column index out of bounds [0, 3]");
		return Vec3(mCol[i]);
	}

	inline VX_INLINE void Mat44::SetColumn(int i, const Vec4& rhs)
	{
		VX_ASSERT(i < 4, "Column index out of bounds [0, 3]");
		mCol[i] = rhs;
	}

	inline VX_INLINE void Mat44::SetColumn3(int i, const Vec3& rhs)
	{
		VX_ASSERT(i < 4, "Column index out of bounds [0, 3]");
		mCol[i] = Vec4(rhs, i == 3 ? 1.0f : 0.0f);
	}


	inline VX_INLINE bool vx::Mat44::operator==(const Mat44& rhs) const
	{
		return mCol[0] == rhs.mCol[0] && 
			   mCol[1] == rhs.mCol[1] && 
			   mCol[2] == rhs.mCol[2] &&
			   mCol[3] == rhs.mCol[3];
	}

	inline VX_INLINE float Mat44::Determinant3x3() const
	{
		/// using geomteric form
		/// scalar triple
		//return GetAxisX().Dot(GetAxisY().Cross(GetAxisZ()));
		// 
		//scalar equivalent Dot(x, Cross(Y, Z)
		return mFloats[0] * (mFloats[5] * mFloats[10] - mFloats[6] * mFloats[9]) -
			mFloats[4] * (mFloats[1] * mFloats[10] - mFloats[2] * mFloats[9]) -
			mFloats[8] * (mFloats[1] * mFloats[6] - mFloats[2] * mFloats[5]);
	}

	inline VX_INLINE int Mat44::GetBasisHandness() const
	{
		const float det = Determinant3x3();
		if (det > kEpsilon) return 1; //Right handed
		if (det < -kEpsilon) return -1; //Left handed
		return 0;					   //Degenerate
	}

	inline VX_INLINE bool Mat44::IsAffine() const
	{
		return mFloats[3] == 0.0f && mFloats[7] == 0.0f && 
			   mFloats[11] == 0.0f && mFloats[15] == 1.0f;
	}

	inline VX_INLINE Vec3 Mat44::Multiply3x3(const Vec3& rhs)
	{
#ifdef VX_USE_SSE
		/// 0x + 4y + 8z + ;0z
		/// 1x + 5y + 9z + ;0z
		/// 2x + 6y + 10z +;0z
		/// 
		/// option 1: grp matrix basis xyz component
		/// x' = XYZx dot rhs
		/// y' = XYZy dot rhs
		/// z' = XYZz dot rhs
		/// 
		/// basis
		/// X: 012 mul rhs splat x
		/// Y: 456 mul rhs splat y
		/// Z: 8910 mul rhs splat z
		/// 
		/// x'y'z' = X + Y + Z
		/// 
		__m128 v = rhs.Value();
		__m128 r = _mm_mul_ps(mCol[0].Value(), _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)));
		r = _mm_add_ps(r, _mm_mul_ps(mCol[1].Value(), _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1))));
		r = _mm_add_ps(r, _mm_mul_ps(mCol[2].Value(), _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2))));
		return Vec3(_mm_shuffle_ps(r, r, _MM_SHUFFLE(2, 2, 1, 0)));
#else
		return Vec3(mFloats[0] * rhs[0] + mFloats[4] * rhs[1] + mFloats[8] * rhs[2],
			mFloats[1] * rhs[0] + mFloats[5] * rhs[1] + mFloats[9] * rhs[2],
			mFloats[2] * rhs[0] + mFloats[6] * rhs[1] + mFloats[10] * rhs[2]);
#endif // VX_USE_SSE
	}



	inline VX_INLINE Mat44 Mat44::PreScaled(const Vec3& scale)
	{
		return Mat44(mCol[0] * scale.X(), mCol[1] * scale.Y(), mCol[2] * scale.Z(), mCol[3]);
	}

	inline VX_INLINE Mat44 Mat44::PostScaled(const Vec3& scale)
	{
		Vec4 s(scale, 1.0f);
		return Mat44(mCol[0] * s, mCol[1] * s, mCol[2] * s, mCol[3] * s);
	}

	inline VX_INLINE Vec4& Mat44::operator[](int column)
	{
		VX_ASSERT(column < 4, "Column index out of bounds [0, 3]");
		return mCol[column];
	}

	inline VX_INLINE Vec4 Mat44::operator[](int column) const
	{
		VX_ASSERT(column < 4, "Column index out of bounds [0, 3]");
		return mCol[column];
	}
}