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
	* - affine transforms (rotation, translation, scale)
	* - 3x3 operations (3x3 upper left of matrix) (rotation / scaling)
	* 
	* Coordinate system:
	* - Right-handed coordniate system
	* - Column-major storage
	* - Column vectors
	* - Vectors are transformed as: v' = M * v
	* 
	* Mathematical layout (row x column):
	* 
	* | R00 R01 R02 Tx  | | 0 4 8 12 | 
	* | R10 R11 R12 Ty  | | 1 5 9 13 | 
	* | R20 R21 R22 Tz  | | 2 6 10 14 |
	* |  0   0   0   1  | | 3 7 11 15 |
	* 
	* Memory layout column: 
	* | R00 R10 R20 0 | R01 R11 R21 0 | R02 R12 R32 0 | Tx Ty Tz 1 |
	*		[0..3]			[4..7]		   [8..11]		 [12..15]
	*	
	* - Rij: Rotation/scale components
	* - Tx, Ty, Tz: Translation components
	* - Bottom row is [0, 0,0, 1] for affine matrices
	* 
	* Basis representation:
	* - Columns 0-2 represent basis vectors (X, Y, Z)
	* - Column 3 reprensent atranslation
	* 
	* |X.x  Y.x  Z.x  T.x|
	* |X.y  Y.y  Z.y  T.y|
	* |X.z  Y.z  Z.z  T.z|
	* | 0    0    0    1 |
	* 
	* 
	* Vector sematics:
	*  - Direction vectors assume w = 0 (no translation)
	*  - Position vectors assume w = 1 (translation applied)
	*
	* 
	* Affine helpers:
	* - Multiply3x3(): ignores translation
	* - TransformDirection(): direction vectors
	* - Transform(): position vectors
	* 
	* 
	*	NOTE:
	*  Intentionally Used Mat44 as Mat33
	* - Mat44 is 64 bytes and modern cache-line aligned 
	* - Mat33 (48 bytes simded) straddles cache lines in array (would reacquire padding) 
	* - Mat44 already provides all 3x3 rotation functionality 
	*/
	class alignas(16) Mat44
	{
	public:
		
		VX_INLINE Mat44() = default;
		explicit VX_INLINE Mat44(const float diagonal);
		VX_INLINE Mat44(const Vec4& col0, const Vec4& col1, const Vec4& col2, const Vec4& col3);
		VX_INLINE Mat44(const Vec4& col0, const Vec4& col1, const Vec4& col2);
		
		/// | 1 0 0 0 |
		/// | 0 1 0 0 |
		/// | 0 0 1 0 |
		/// | 0 0 0 1 |
		/// @return identity matrix
		static VX_INLINE Mat44 Identity();
		/// debug / utility matrix containing linear indices
		/// Values stored in memeory order (column-major)
		/// | 0 4 8 12 |
		/// | 1 5 9 13 |
		/// | 2 6 10 14 |
		/// | 3 7 11 15 | 
		static VX_INLINE Mat44 Dummy();
		/// Build translation matrix
		/// | 1 0 0 t.x |
		/// | 0 1 0 t.y |
		/// | 0 0 1 t.z |
		/// | 0 0 0 1 |
		static VX_INLINE Mat44 Translation(const Vec3& v);
		/// Build uniform scale matrix
		/// | s 0 0 0 |
		/// | 0 s 0 0 |
		/// | 0 0 s 0 |
		/// | 0 0 0 1 |
		static VX_INLINE Mat44 Scale(float scale);
		/// Build non-uniform scale matrix
		/// | s.x  0   0  0 |
		/// |  0  s.y  0  0 |
		/// |  0   0  s.z 0 |
		/// |  0   0   0  1 |
		static VX_INLINE Mat44 Scale(const Vec3& scale);
		/// Rotation about X axis (pitch), radians
		static VX_INLINE Mat44 RotationX(float angle);
		/// Rotation about Y axis (yaw), radians
		static VX_INLINE Mat44 RotationY(float angle);
		/// Rotation about Z axis (row), radians
		static VX_INLINE Mat44 RotationZ(float angle);
		/// Build matrix from orthonormal basis vectors
		/// Columns represent X, Y, Z axes; translation is zero
		static VX_INLINE Mat44 Basis(const Vec3& x, const Vec3& y, const Vec3& z);
		/// Build matrix from basis vectors and translation
		static VX_INLINE Mat44 BasisTranslation(const Vec3& x, const Vec3& y, const Vec3& z, const Vec3& t);
		
		/// Builds a 4x4 rotation matrix (R - 3x3) from a unit quaternion 
		/// Translation is zero, bottom [0, 0, 0, 1]
		/// | R 0 |
		/// | 0 1 |
		static VX_INLINE Mat44 Rotation(const Quat& q);
		/// Builds a 4x4 Translation rotation matrix (R - 3x3) from a unit quaternion 
		/// Translation is zero, bottom [0, 0, 0, 1]
		/// | R t |
		/// | 0 1 |
		static VX_INLINE Mat44 RotationTranslation(const Quat& q, const Vec3& t);

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
		VX_INLINE bool IsOrthonormal(float tolerance = 1e-4f) const;
		VX_INLINE Mat44 Transposed3x3() const;
		VX_INLINE Mat44 Transposed() const;
		VX_INLINE Mat44 Inverse3x3() const;
		VX_INLINE Mat44 InverseAffine() const;

		VX_INLINE Vec3 Multiply3x3(const Vec3& rhs) const;
		VX_INLINE Vec3 Multiply3x3Transposed(const Vec3& rhs) const;
		VX_INLINE Vec3 MultiplyAffine(const Vec3& rhs) const;
		VX_INLINE Mat44 Multiply3x3(const Mat44& rhs) const;
		VX_INLINE Mat44 Multiply3x3LeftTransposed(const Mat44& rhs) const;
		VX_INLINE Mat44 Multiply3x3RightTransposed(const Mat44& rhs) const;
		VX_INLINE Mat44 Multiply(const Mat44& rhs) const;
		VX_INLINE Mat44 MultiplyAffine(const Mat44& rhs) const;

		VX_INLINE Mat44 Add(const Mat44& rhs) const;
		VX_INLINE Mat44 AddAffine(const Mat44& rhs) const;
		/// Cross Mat44(Mat33) x Vec3
		static VX_INLINE Mat44 SkewSymmetric3x3(const Vec3& rhs);
		VX_INLINE Mat44 operator+(const Mat44& rhs)const = delete;
		VX_INLINE Mat44& operator+=(const Mat44& rhs) = delete;


		/// Transform a position vector by this matrix.
		/// Applies rotation and translation
		VX_INLINE Vec3 Transform(const Vec3& vec)const;
		VX_INLINE static Vec3 Transform(const Mat44& matrix, const Vec3& translate);
		
		/// Tranforms a position vector by inverse of this matrix.
		/// Assume pure rotation and translation
		VX_INLINE Vec3 TransformInverse(const Vec3& vector) const;
		VX_INLINE static Vec3 TransformInverse(const Mat44& matrix, const Vec3& translate);
		
		/// Transform a direction vector by this matrix (rotated direction)
		/// ignore translation
		VX_INLINE Vec3 TransformDirection(const Vec3& vec) const;
		/// Transform a direction vector by the inverse of this matrix (inversely rotated direction).
		/// Assue pure rotation (no scale/shear).
		VX_INLINE Vec3 TransformInverseDirection(const Vec3& vec) const;

		/// Sets the rotation part of matrix from a unit quaternion 
		/// Translation preserved, bottom [0, 0, 0, 1]
		/// | R t |
		/// | 0 1 |
		VX_INLINE void SetRotation(const Quat& q);
		/// Set both rotation and translation components from a quaternion and translation
		/// @param q Quaternion representing rotation
		/// @param pos Translation Vector
		VX_INLINE void SetRotationAndTranslation(const Quat& q, const Vec3& pos);

		VX_INLINE Mat44 GetRotationMat33() const;
		/// Extract the rotation component of matrix as quartenion 
		/// matrix must represnet pure rotation (ortho nor 3x3)
		/// UB, if matix contain scale or shear
		VX_INLINE Quat GetRotationQuat() const;

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
		/// Decomposing matrix into Mat44 (containing rotation & tranlsation) 
		/// out scale (vec3)
		/// see https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
		VX_INLINE Mat44 Decompose(Vec3& o_scale) const;
		VX_INLINE void MakeOrthonormal();

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