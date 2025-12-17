#pragma once

#include "Core.h"
#include "Vec4.h"
namespace vx
{

	/*
	* Mat44:- 4x4 matrix class
	* 4x4 Column-major matrix for affine and general tranformation
	* 
	* Usage:
	* - 4x4 operations (general transforms)
	* - 4x3 operations (affine transforms)
	* - 3x3 operations (3x3 upper left of matrix) (rotation / scaling)
	* 
	* Commonly Used for:
	* - Rotation
	* - Translation
	* - Transform composition in rigidbody dynamics.
	* 
	* Coordinate system:
	* - Right-handed
	* - Column-major storage
	* - Column vectors
	* 
	* Mathematical layout (row x column):
	* 
	* | R00 R01 R02 Tx  | | 0 4 8 12 | 
	* | R10 R11 R12 Ty  | | 1 5 9 13 | 
	* | R20 R21 R23 Tz  | | 2 6 10 14 |
	* |  0   0   0   1  | | 3 7 11 15 |
	* 
	* Memory layout column: 
	* | R00 R10 R20 0 | R01 R11 R21 0 | R02 R12 R32 0 | Tx Ty Tz 1 |
	*		[0..3]			[4..7]		   [8..11]		 [12..15]
	*	
	* - Rij: Rotation/scale components
	* - Tx, Ty, Tz: Translation components
	* 
	* 
	* Basis representation:
	* - Columns represent basis vectors and translation
	* 
	* |A.x  B.x  C.x  T.x|
	* |A.y  B.y  C.y  T.y|
	* |A.z  B.z  C.z  T.z|
	* | 0    0    0    1 |
	* 
	* 
	* Vector transform:
	* 
	* v' = M*v
	* 
	* Affine special cases:
	* - Multiply3x3(v): ignores translation
	* - Direction vectors assume w = 0
	* - Position vectors assume w = 1
	* 
	*/
	class alignas(16) Mat44
	{
	public:
		
		VX_INLINE Mat44() = default;
		explicit VX_INLINE Mat44(const float diagonal);
		VX_INLINE Mat44(const Vec4& col0, const Vec4& col1, const Vec4& col2, const Vec4& col3);
		VX_INLINE Mat44(const Vec4& col0, const Vec4& col1, const Vec4& col2);


		static VX_INLINE Mat44 Identity();
		/// dummy matrix that stores
		/// matrix with value in their order in memory
		/// | 0 4 8 12 |
		/// | 1 5 9 13 |
		/// | 2 6 10 14 |
		/// | 3 7 11 15 | 
		static VX_INLINE Mat44 Dummy();
		/// matrix that scales uniformly
		/// | 1 0 0 t.x |
		/// | 0 1 0 t.y |
		/// | 0 0 1 t.z |
		/// | 0 0 0 1 |
		static VX_INLINE Mat44 Translation(const Vec3& v);
		/// matrix that scales uniformly
		/// | s 0 0 0 |
		/// | 0 s 0 0 |
		/// | 0 0 s 0 |
		/// | 0 0 0 1 |
		static VX_INLINE Mat44 Scale(float scale);
		/// matrix that scales (produces a matrix with (inV, 1) on its diagonal)
		/// | s.x  0   0  0 |
		/// |  0  s.y  0  0 |
		/// |  0   0  s.z 0 |
		/// |  0   0   0  1 |
		static VX_INLINE Mat44 Scale(const Vec3& scale);
		/// rotation around X axis PITCH in radian
		/// | R/S   R   R  Tx |
		/// |  R   cos -sin Ty |
		/// |  R   sin  cos Tz |
		/// |  0    0    0   1 |
		/// 
		static VX_INLINE Mat44 RotationX(float in_x_rad);
		/// rotation around Z axis YAW in radian
		/// |  cos  R  sin  Tx |
		/// |   R  R/S  R   Ty |
		/// | -sin  R  cos Tz |
		/// |   0   0   0   1 |
		/// 
		static VX_INLINE Mat44 RotationY(float in_y_rad);
		/// rotation around Z axis Roll in radian
		/// | cos -sin   R  Tx |
		/// | sin  cos   R  Ty |
		/// |  R    R   R/S Tz |
		/// |  0    0    0   1 |
		/// 
		static VX_INLINE Mat44 RotationZ(float in_z_rad);
		static VX_INLINE Mat44 Basis(const Vec3& x, const Vec3& y, const Vec3& z);
		static VX_INLINE Mat44 BasisTranslation(const Vec3& x, const Vec3& y, const Vec3& z, const Vec3& t);
		//static VX_INLINE Mat44 Rotation(const Quat& q);
		//static VX_INLINE Mat44 RotationTranslation(const Quat& q, const Vec3& t);

		//Accessor
		VX_INLINE float& operator()(int row, int column);
		VX_INLINE float operator()(int row, int column) const;

		VX_INLINE Vec4 GetColumn(int i) const;
		VX_INLINE Vec3 GetColumn3(int i) const;
		VX_INLINE void SetColumn(int i, const Vec4& rhs);
		VX_INLINE void SetColumn3(int i, const Vec3& rhs);

		/// @return matrix column 0
		VX_INLINE Vec3 GetAxisX() const { return mCol[0]; }
		/// @return matrix column 1
		VX_INLINE Vec3 GetAxisY() const { return mCol[1]; }
		/// @return matrix column 2
		VX_INLINE Vec3 GetAxisZ() const { return mCol[2]; }
		/// @return matrix column 3
		VX_INLINE Vec3 GetTranslation() const { return mCol[3]; }

		VX_INLINE void SetAxisX(const Vec3& v, const float w = 0.0f) { mCol[0] = Vec4(v, w);}
		VX_INLINE void SetAxisY(const Vec3& v, const float w = 0.0f) { mCol[1] = Vec4(v, w);}
		VX_INLINE void SetAxisZ(const Vec3& v, const float w = 0.0f) { mCol[2] = Vec4(v, w);}
		VX_INLINE void SetTranslation(const Vec3& v, const float w = 1.0f) { mCol[3] = Vec4(v, w); }

		VX_INLINE Vec4 GetDiagonal() const { return Vec4(mFloats[0], mFloats[5], mFloats[10], mFloats[15]); }
		VX_INLINE Vec3 GetDiagonal3() const { return Vec3(mFloats[0], mFloats[5], mFloats[10]); }
		VX_INLINE void SetDiagonal(const Vec4& rhs) { mFloats[0] = rhs.X(); mFloats[5] = rhs.Y(); mFloats[10] = rhs.Z(); mFloats[15] = rhs.W(); }
		VX_INLINE void SetDiagonal3(const Vec3& rhs) { mFloats[0] = rhs.X(); mFloats[5] = rhs.Y(); mFloats[10] = rhs.Z(); mFloats[15] = 1.0f; }


		/// Comparison
		VX_INLINE bool operator == (const Mat44& rhs) const;
		VX_INLINE bool operator != (const Mat44& rhs) const { return !(*this == rhs); }

		VX_INLINE float Determinant3x3() const;
		VX_INLINE int GetBasisHandness() const;
		VX_INLINE bool IsAffine() const;
		VX_INLINE bool IsAffine3x3() const;
		VX_INLINE Mat44 Transposed3x3() const;
		VX_INLINE Mat44 Transposed() const;
		VX_INLINE Mat44 Inverse3x3() const;
		VX_INLINE Mat44 InverseAffine() const;

		VX_INLINE Vec3 Multiply3x3(const Vec3& rhs) const;
		VX_INLINE Vec3 Multiply3x3Transposed(const Vec3& rhs) const;
		VX_INLINE Mat44 Multiply3x3(const Mat44& rhs) const;
		VX_INLINE Mat44 Multiply3x3LeftTransposed(const Mat44& rhs) const;
		VX_INLINE Mat44 Multiply3x3RightTransposed(const Mat44& rhs) const;
		VX_INLINE Mat44 Multiply(const Mat44& rhs) const;
		VX_INLINE Mat44 MultiplyAffine(const Mat44& rhs) const;

		/// post scale matrix == pre scale 
		/// for affine transform, pre/post are equivalenrt
		/// preventing full matrix mutltiplication 
		/// with focus on scale components 
		/// | Sx R  R  Tx |
		/// | R  Sy R  Ty |
		/// | R  R  Sz Tz |
		/// | 0  0  0   1 |
		/// 
		/// return result Sx * s.x, Sy * s.y, Sz * s.z
		VX_INLINE Mat44 PreScaled(const Vec3& scale);
		VX_INLINE Mat44 PostScaled(const Vec3& scale);

		//util
			//Inverse


		friend std::ostream& operator<<(std::ostream& os, const Mat44& m);
	private:
		//Get column by index
		/// Retruns the i-th axis vector column).
		/// i column index [0-3].
		/// @return Vec4 representing the axis/column
		VX_INLINE Vec4& operator[](int column);
		VX_INLINE Vec4 operator[](int column) const;
		union
		{
			Vec4 mCol[4];
			float mFloats[16];
		};
	};
}

#include "Mat44.inl"