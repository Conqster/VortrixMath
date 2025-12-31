#include "TestCommon.h"
#include "ViewProjection.h"

TEST_SUITE("View Projection Tests")
{

	TEST_CASE("ViewProject")
	{
		float fov = vx::DegToRad(90.0f);
		float aspect = 1.0f;
		float zNear = 1.0f;
		float zFar = 100.0f;

		vx::Mat44 proj = vx::Perspective_ZO(fov, aspect, zNear, zFar);

		vx::Vec4 near_p(0.0f, 0.0f, -zNear, 1.0f);
		vx::Vec4 far_p(0.0f, 0.0f, -zFar, 1.0f);

		vx::Vec4 near_clip = proj.Multiply(near_p);
		vx::Vec4 far_clip = proj.Multiply(far_p);

		CHECK(vx::VxAbs(near_clip.W()) > 1e-5f);
		CHECK(vx::VxAbs(far_clip.W()) > 1e-5f);

		near_clip /= near_clip.W();
		far_clip /= far_clip.W();

		CHECK_APPROX_EQ(near_clip.Z(), 0.0f, 1e-5f);
		CHECK_APPROX_EQ(far_clip.Z(), 1.0f, 1e-5f);



		proj = vx::Perspective_NO(fov, aspect, zNear, zFar);
		near_p = vx::Vec4(0.0f, 0.0f, -zNear, 1.0f);
		far_p = vx::Vec4(0.0f, 0.0f, -zFar, 1.0f);

		near_clip = proj.Multiply(near_p);
		far_clip = proj.Multiply(far_p);

		CHECK(vx::VxAbs(near_clip.W()) > 1e-5f);
		CHECK(vx::VxAbs(far_clip.W()) > 1e-5f);

		near_clip /= near_clip.W();
		far_clip /= far_clip.W();

		CHECK_APPROX_EQ(near_clip.Z(), -1.0f, 1e-5f);
		CHECK_APPROX_EQ(far_clip.Z(), 1.0f, 1e-5f);
	}


	TEST_CASE("Perspective ZO FOV Scaling")
	{
		float fov = vx::DegToRad(90.0f);
		float aspect = 1.0f;
		float zNear = 1.0f;

		vx::Mat44 proj = vx::Perspective_ZO(fov, aspect, zNear, 100.0f);
		float half_width = zNear * vx::VxTan(0.5f * fov);

		//rt edge
		vx::Vec4 p(half_width, 0.0f, -zNear, 1.0f);
		vx::Vec4 clip = proj.Multiply(p);
		clip /= clip.W();

		CHECK_APPROX_EQ(clip.X(), 1.0f);
		CHECK_APPROX_EQ(clip.Y(), 0.0f);
		CHECK_APPROX_EQ(clip.Z(), 0.0f);//[0, 1]

		proj = vx::Perspective_NO(fov, aspect, zNear, 100.0f);
		half_width = zNear * vx::VxTan(0.5f * fov);

		//rt edge
		p = vx::Vec4(half_width, 0.0f, -zNear, 1.0f);
		clip = proj.Multiply(p);
		clip /= clip.W();

		CHECK_APPROX_EQ(clip.X(), 1.0f);
		CHECK_APPROX_EQ(clip.Y(), 0.0f);
		CHECK_APPROX_EQ(clip.Z(), -1.0f); //[-1, 1]
	}

	TEST_CASE("Orthographic Bounds")
	{
		vx::Mat44 ortho = vx::Orthographic_ZO(
			-10.0f, 10.0f,
			-10.0f, 10.0f,
			0.0f, 100.0f
		);

		vx::Vec4 lt(-10.0f, 0.0f, 0.0f, 1.0f);
		vx::Vec4 rt(10.0f, 0.0f, 0.0f, 1.0f);
		vx::Vec4 near_p(0.0f, 0.0f, 0.0f, 1.0f);
		vx::Vec4 far_p(0.0f, 0.0f, 100.0f, 1.0f);

		vx::Vec4 l = ortho.Multiply(lt);
		vx::Vec4 r = ortho.Multiply(rt);
		vx::Vec4 n = ortho.Multiply(near_p);
		vx::Vec4 f = ortho.Multiply(far_p);

		CHECK_APPROX_EQ(l.X(), -1.0f);
		CHECK_APPROX_EQ(r.X(), 1.0f);
		CHECK_APPROX_EQ(n.Z(), 0.0f);
		CHECK_APPROX_EQ(f.Z(), 1.0f);

		//// NO
		ortho = vx::Orthographic_NO(
			-10.0f, 10.0f,
			-10.0f, 10.0f,
			0.0f, 100.0f
		);

		l = ortho.Multiply(lt);
		r = ortho.Multiply(rt);
		n = ortho.Multiply(near_p);
		f = ortho.Multiply(far_p);

		CHECK_APPROX_EQ(l.X(), -1.0f);
		CHECK_APPROX_EQ(r.X(), 1.0f);
		CHECK_APPROX_EQ(n.Z(), -1.0f);
		CHECK_APPROX_EQ(f.Z(), 1.0f);
	}



	TEST_CASE("LookAt")
	{
		vx::Vec4 pos(0.0f);
		vx::Vec3 target = -vx::Vec3::Forward();

		vx::Mat44 view = vx::LookAt(pos, target, vx::Vec3::Up());

		vx::Vec3 origin(0.0f);
		CHECK_APPROX_EQ(view.Transform(origin), origin);

		vx::Vec3 x = view.GetAxisX();
		vx::Vec3 y = view.GetAxisY();
		vx::Vec3 z = view.GetAxisZ();

		CHECK_APPROX_EQ(x.Length(), 1.0f);
		CHECK_APPROX_EQ(y.Length(), 1.0f);
		CHECK_APPROX_EQ(z.Length(), 1.0f);

		CHECK_APPROX_EQ(x.Dot(y), 0.0f);
		CHECK_APPROX_EQ(y.Dot(z), 0.0f);
		CHECK_APPROX_EQ(z.Dot(x), 0.0f);
	}

	TEST_CASE("LookAT Translation")
	{
		vx::Vec3 pos(0.0f, 0.0f, 5.0f);
		vx::Vec3 target(0.0f, 0.0f, 6.0f);

		vx::Mat44 view = vx::LookAt(pos, target, vx::Vec3::Up());

		CHECK_APPROX_EQ(view.Transform(pos), vx::Vec3::Zero());

		view = vx::LookAt(pos, pos - vx::Vec3::Forward(), vx::Vec3::Up());
		vx::Vec3 origin(0.0f);
		vx::Vec3 view_pos = view.Transform(origin);

		CHECK_APPROX_EQ(view_pos, -pos);
	}
}