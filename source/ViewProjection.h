#pragma once
#include "Mat44.h"


///Program enforce right hand
#define VX_RENDER_CLIP_SPACE_ZO_BIT (1 << 0)
#define VX_RENDER_CLIP_SPACE_NO_BIT (1 << 1)

#define VX_RENDER_CLIP_SPACE_ZO VX_RENDER_CLIP_SPACE_ZO_BIT
#define VX_RENDER_CLIP_SPACE_NO VX_RENDER_CLIP_SPACE_NO_BIT
#define VX_RENDER_CLIP_SPACE VX_RENDER_CLIP_SPACE_NO

namespace vx
{
	/// Default, this system is right-handed
	/// positive X-axis points right, Vec3(1.0f, 0.0f, 0.0f).
	/// positive Y-axis points up, Vec3(0.0f, 1.0f, 0.0f).
	/// positive Z-axis point "out" of screen towards viewer.
	/// i.e scene (forward) is -Vec3::Forward(), Vec3(0.0f, 0.0f, -1.0f).


	/// Build a right handed look at view matrix
	/// 
	/// @param pos Camera position
	/// @param target Point camera is looking at
	/// @param up vector of camera upward vector (assumed normalised)
	static VX_INLINE Mat44 LookAt(const Vec3& pos,
								  const Vec3& target,
								  const Vec3& up)
	{
		const Vec3 fwd = (target - pos).Normalised();
		const Vec3 rt = Vec3::Cross(fwd, up);
		const Vec3 u = Vec3::Cross(rt, fwd);

		Vec4 x = Vec4(rt.X(), u.X(), -fwd.X(), 0.0f);
		Vec4 y = Vec4(rt.Y(), u.Y(), -fwd.Y(), 0.0f);
		Vec4 z = Vec4(rt.Z(), u.Z(), -fwd.Z(), 0.0f);
		Vec4 t = Vec4(-Vec3::Dot(rt, pos),
					  -Vec3::Dot(u, pos),
					  Vec3::Dot(fwd, pos),
					  1.0f);
		return Mat44(x, y, z, t);
	}


	/// Create a right handed perspective-view
	/// near and far planes correspond to z normalised device 0 and +1 respectively
	/// 
	/// for right handed (Vulkan rendering volume)
	/// 
	/// @param fov Field of view (radians)
	/// @param aspect Aspect ratio of the field of veiw in x direction (ratio of x(width) to y(height).
	/// @param near Distance from view to the near clipping plane
	/// @param far Distance from view to the far clipping plane
	static VX_INLINE Mat44 Perspective_ZO(float fov,
									   float aspect,
									   float zNear,
									   float zFar)
	{
		float h = 1.0f / VxTan(0.5f * fov);
		float w = h / aspect;

		return Mat44(
			Vec4(w, 0.0f, 0.0f, 0.0f),
			Vec4(0.0f, h, 0.0f, 0.0f),
			Vec4(0.0f, 0.0f, zFar / (zNear - zFar), -1.0f),
			Vec4(0.0f, 0.0f, -(zFar * zNear) / (zFar - zNear), 0.0f));
	}

	/// Create a right handed perspective-view
	/// near and far planes correspond to z normalised device -1 and +1 respectively
	/// 
	/// for (OpenGL rendering volume)
	/// 
	/// @param fov Field of view (radians)
	/// @param aspect Aspect ratio of the field of veiw in x direction (ratio of x(width) to y(height).
	/// @param near Distance from view to the near clipping plane
	/// @param far Distance from view to the far clipping plane
	static VX_INLINE Mat44 Perspective_NO(float fov,
										    float aspect,
										    float zNear,
										    float zFar)
	{
		float h = 1.0f / VxTan(0.5f * fov);
		float w = h / aspect;
		// [-1, 1]
		float neg_diff_ratio = -1.0f / (zFar - zNear);

		return Mat44(
			Vec4(w, 0.0f, 0.0f, 0.0f),
			Vec4(0.0f, h, 0.0f, 0.0f),
			Vec4(0.0f, 0.0f, (zFar + zNear) * neg_diff_ratio, -1.0f),
			Vec4(0.0f, 0.0f, (2.0f * zFar * zNear) * neg_diff_ratio, 0.0f));
	}

	/// Create a right handed perspective-view,
	/// Default near and far is NO correspond to z normalised device 0 and +1 respectively
	/// 
	/// @param fov Field of view (radians)
	/// @param aspect Aspect ratio of the field of veiw in x direction (ratio of x(width) to y(height).
	/// @param near Distance from view to the near clipping plane
	/// @param far Distance from view to the far clipping plane
	static VX_INLINE Mat44 Perspective(float fov,
									   float aspect,
									   float zNear,
									   float zFar)
	{
#if VX_RENDER_CLIP_SPACE == VX_RENDER_CLIP_SPACE_ZO
		return Perspective_ZO(fov, aspect, zNear, zFar);
#else VX_RENDER_CLIP_SPACE == VX_RENDER_CLIP_SPACE_NO
		return Perspective_NO(fov, aspect, zNear, zFar);
#endif // VX_RENDER_CLIP_SPACE == VX_RENDER_CLIP_SPACE_ZO

	}

	static VX_INLINE Mat44 Orthographic_ZO(float left,
		float right,
		float bottom,
		float top,
		float zNear,
		float zFar)
	{
		float inv_rt_diff = 1.0f / (right - left);
		float inv_tb_diff = 1.0f / (top - bottom);
		float inv_fn_diff = 1.0f / (zFar - zNear);

		return Mat44(
			Vec4(2.0f * inv_rt_diff, 0.0f, 0.0f, 0.0f),
			Vec4(0.0f, 2.0f * inv_tb_diff, 0.0f, 0.0f),
			Vec4(0.0f, 0.0f, inv_fn_diff, 0.0f),
			Vec4(-(right + left) * inv_rt_diff, -(top + bottom) * inv_tb_diff, -zNear * inv_fn_diff, 1.0f));
	}
	

	static VX_INLINE Mat44 Orthographic_NO(float left,
										   float right,
										   float bottom,
										   float top,
										   float zNear,
										   float zFar)
	{
		float inv_rt_diff = 1.0f / (right - left);
		float inv_tb_diff = 1.0f / (top - bottom);
		float inv_fn_diff = 1.0f / (zFar - zNear);

		return Mat44(
			Vec4(2.0f * inv_rt_diff, 0.0f, 0.0f, 0.0f),
			Vec4(0.0f, 2.0f * inv_tb_diff, 0.0f, 0.0f),
			Vec4(0.0f, 0.0f, 2.0f * inv_fn_diff, 0.0f),
			Vec4(-(right + left) * inv_rt_diff, -(top + bottom) * inv_tb_diff, -(zFar + zNear) * inv_fn_diff, 1.0f));
	}
}