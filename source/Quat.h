#pragma once

#include "Core.h"
#include "Vec4.h"
#include "Mat44.h"

namespace vx
{
	/*
	* Quat
	* Quaternion class for representing 3D rotations.
	* 
	* Quaternion are 4D vectors used to hold 
	* 3 degrees of rotational freedom (DOF) in 3D space.
	* if normalised (length = 1), 
	* they repsent vaild rotation frame without ginbal lock.
	* 
	* Layout: 
	* [x, y, z, w]
	* - [x, y, z]: Imaginary components (rotation axis) 
	* - [w]: Real component (rotation magnitude)
	* 
	* Voltrix use right-handed coordinate system and column-major matrix layout.
	* Quaternion multiplcation is non-commutative: q1 * q2 != q2 * q1.
	* 
	* Vec4 as generic container
	*/
	class alignas(16) Quat
	{
	public:
		Quat() = default;
		Quat(float x, float y, float z, float w) : mValue(x, y, z, w){}
		Quat(const Vec3& imaginary, float w) : mValue(imaginary, w){}
		Quat(const Vec4& component) : mValue(component){}

		VX_INLINE float X() const { return mValue.X(); }
		VX_INLINE float Y() const { return mValue.Y(); }
		VX_INLINE float Z() const { return mValue.Z(); }
		VX_INLINE float W() const { return mValue.W(); }

		VX_INLINE Vec3 Imaginary() const { return Vec3(mValue); }
		VX_INLINE Vec4 XYZW() const { return mValue; }

		static VX_INLINE Quat Identity() { return Quat(0.0f, 0.0f, 0.0f, 1.0f); }
		static VX_INLINE Quat FromAxisAngle(const Vec3& axis, float angle);

		VX_INLINE void SetAxisAngle(const Vec3& axis, float angle);
		VX_INLINE void GetAxisAngle(Vec3& axis, float angle);


		VX_INLINE void Normalise();
		VX_INLINE Quat Normalised() const;

		VX_INLINE bool IsUnitQuat(float tolerance = 1e-4f) const;

		VX_INLINE float LengthSq() const { return mValue.LengthSq(); }
		VX_INLINE float Length() const { return mValue.Length(); }

		VX_INLINE Quat Conjugated() const { return Quat(mValue.FlipSign<-1, -1, -1, 1>()); }
		VX_INLINE Quat Inversed() const;
		/// Rotates a vector by this unit quaternion
		/// Quat * Vec3 
		/// v' = q * v * q^-1
		/// 
		/// Rodrigues formula :
		/// v'= v + 2 * w * (q.xyz x v) + 2 * (q.xyz x (q.xyz x v))
		/// v' = v' = v + wt + (q.xyz x t)
		///		t = 2(q.xyz x v)
		/// @param vector representing angular velocity
		/// @note in physics orientation update
		VX_INLINE Vec3 Rotate(const Vec3& vec) const;
		/// Rotate a vector from world space
		/// into quat local space
		/// Quat * Vec3 
		/// v' = q^-1 * v * q
		/// 
		/// Rodrigues formula :
		/// v'= v + 2 * w * (q.xyz x v) + 2 * (q.xyz x (q.xyz x v))
		/// v' = v' = v + wt + (q.xyz x t)
		///		t = 2(q.xyz x v)
		VX_INLINE Vec3 InverseRotate(const Vec3& vec) const;

		VX_INLINE Vec3 RotateSlow(const Vec3& vec) const;
		VX_INLINE Vec3 InverseRotateSlow(const Vec3& vec) const;

		VX_INLINE Vec3 RotateAxisX() const;
		VX_INLINE Vec3 RotateAxisY() const;
		VX_INLINE Vec3 RotateAxisZ() const;

		VX_INLINE Vec3 RotateScaledAxisX(float scale) const;
		VX_INLINE Vec3 RotateScaledAxisY(float scale) const;
		VX_INLINE Vec3 RotateScaledAxisZ(float scale) const;

		VX_INLINE void RotateScaledAxes(const Vec3& scale, Vec3& out_x, Vec3& out_y, Vec3& out_z);
		VX_INLINE Mat44 GetRotationMat44();

		VX_INLINE Quat operator*(const Quat& rhs) const;
		VX_INLINE Quat operator*=(const Quat& rhs);

		VX_INLINE Vec3 operator*(const Vec3& vec) const { return Rotate(vec); }

		VX_INLINE Quat operator/(float rhs) const { return Quat(mValue / rhs); }

	private:
		Vec4 mValue;
	};
}
#include "Quat.inl"