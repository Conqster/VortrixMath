#include "TestCommon.h"
#include "Quat.h"

TEST_SUITE("Quat Tests")
{

	TEST_CASE("Test Quat construct")
	{
		vx::Quat q = vx::Quat::Identity();

		
		CHECK_APPROX_EQ(q.XYZW(), vx::Vec4(0, 0, 0, 1));

		CHECK_APPROX_EQ(q.X(), 0.0f);
		CHECK_APPROX_EQ(q.Y(), 0.0f);
		CHECK_APPROX_EQ(q.Z(), 0.0f);
		CHECK_APPROX_EQ(q.W(), 1.0f);


		CHECK(q.IsUnitQuat());
	}


	TEST_CASE("Util")
	{
		//identity rotation 
		vx::Quat identity = vx::Quat::Identity();

		vx::Vec3 v(1, 2, 3);
		CHECK_APPROX_EQ(identity.Rotate(v), v);


		vx::Quat q = vx::Quat::FromAxisAngle(vx::Vec3(1, 0, 0), 0.7f);
		vx::Quat nq = q / -1.0f;


		CHECK_APPROX_EQ(q.Rotate(v), nq.Rotate(v));
	}



	TEST_CASE("Normalise and length")
	{
		//identity rotation 
		vx::Quat q(1, 2, 3, 4);
		q.Normalise();

		CHECK(q.IsUnitQuat());
		CHECK_APPROX_EQ(q.Length(), 1.0f);


		vx::Quat a(1, 2, 3, 4);
		vx::Quat b = a.Normalised();

		CHECK_APPROX_EQ(a.Length(), vx::VxSqrt(30.0f));
		CHECK_APPROX_EQ(b.Length(), 1.0f);
	}


	TEST_CASE("Conjugate & inverse")
	{
		//identity rotation 
		vx::Quat q(1, 2, 3, 4);
		vx::Quat c = q.Conjugated();

		CHECK_APPROX_EQ(c.XYZW(), vx::Vec4(-1, -2, -3, 4));


		q = vx::Quat::FromAxisAngle(vx::Vec3(0, 1, 0), 1.0f);
		vx::Quat inv = q.Inversed();

		vx::Quat id = q * inv;

		CHECK_APPROX_EQ(id.XYZW(), vx::Quat::Identity().XYZW());
	}



	TEST_CASE("Axis-Angle")
	{
		vx::Vec3 axis = vx::Vec3(0, 1, 0);
		float angle = 1.234f;

		vx::Quat q = vx::Quat::FromAxisAngle(axis, angle);

		vx::Vec3 out_axis;
		float out_angle;
		q.GetAxisAngle(out_axis, out_angle);

		CHECK_APPROX_EQ(out_angle, angle);
		CHECK_APPROX_EQ(out_axis, axis);

		/// Zero angle
		q = vx::Quat::Identity();
		q.GetAxisAngle(out_axis, out_angle);

		CHECK_APPROX_EQ(out_angle, 0.0f);
		CHECK(out_axis.IsZero());

	}


	TEST_CASE("Quat Multpltt")
	{
		//identity rotation 
		vx::Quat a = vx::Quat::FromAxisAngle(vx::Vec3(1, 0, 0), 0.3f);
		vx::Quat b = vx::Quat::FromAxisAngle(vx::Vec3(0, 1, 0), 0.4f);
		vx::Quat c = vx::Quat::FromAxisAngle(vx::Vec3(0, 0, 1), 0.5f);

		vx::Quat r1 = (a * b) * c;
		vx::Quat r2 = a * (b * c);

		CHECK_APPROX_EQ(r1.Normalised().XYZW(), r2.Normalised().XYZW());

		r1.Normalise();
		r2.Normalise();
		vx::Vec3 v(1, 2, 3);
		//CHECK_APPROX_EQ(r1.Rotate(v), r2.Rotate(v), 1e-4f);
	}

	TEST_CASE("Quat Rotatw")
	{
		//identity rotation 
		vx::Quat q = vx::Quat::FromAxisAngle(vx::Vec3(0, 1, 0), 1.0f);
		vx::Vec3 v(3, 4, 5);

		CHECK_APPROX_EQ(q.Rotate(v), q.RotateSlow(v));
		CHECK_APPROX_EQ(q.InverseRotate(v), q.InverseRotateSlow(v));

		q = vx::Quat::FromAxisAngle(vx::Vec3(0, 0, 1), 0.9f);
		v = vx::Vec3(1, 0, 0);

		vx::Vec3 r = q.Rotate(v);
		vx::Vec3 back = q.InverseRotate(r);

		CHECK_APPROX_EQ(back, v);

	}

	TEST_CASE("Quat Basis rotatw")
	{

		vx::Quat q = vx::Quat::FromAxisAngle(vx::Vec3(0, 0, 1), vx::kVxPi * 0.5f);


		CHECK_APPROX_EQ(q.RotateAxisX(), vx::Vec3(0, 1, 0));
		CHECK_APPROX_EQ(q.RotateAxisY(), vx::Vec3(-1, 0, 0));
		CHECK_APPROX_EQ(q.RotateAxisZ(), vx::Vec3(0, 0, 1));
	}


	TEST_CASE("Quat rotate scaled axes")
	{

		vx::Quat q = vx::Quat::FromAxisAngle(vx::Vec3(0, 1, 0), 0.8f);
		vx::Vec3 scale(2, 3, 4);

		vx::Vec3 x, y, z;
		q.RotateScaledAxes(scale, x, y, z);

		CHECK_APPROX_EQ(q.RotateAxisX() * scale.X(), x);
		CHECK_APPROX_EQ(q.RotateAxisY() * scale.Y(), y);
		CHECK_APPROX_EQ(q.RotateAxisZ() * scale.Z(), z);

		vx::Mat44 M = q.GetRotationMat44().PreScaled(scale);

		CHECK_APPROX_EQ(M.GetAxisX(), x);
		CHECK_APPROX_EQ(M.GetAxisY(), y);
		CHECK_APPROX_EQ(M.GetAxisZ(), z);
	}

	TEST_CASE("stress")
	{

		//for (int i = 0; i < 10000; ++i)
		//{
		//	vx::Vec3 axis = 
		//}
	}
}