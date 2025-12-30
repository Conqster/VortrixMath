#include "TestCommon.h"

TEST_SUITE("Vec4 Tests")
{

	TEST_CASE("Test Vec4 construct")
	{
		vx::Vec4 v(4.0f);


		CHECK_APPROX_EQ(v.X(),4.0f);
		CHECK_APPROX_EQ(v.Y(),4.0f);
		CHECK_APPROX_EQ(v.Z(),4.0f);
		CHECK_APPROX_EQ(v.W(),4.0f);


		vx::Vec4 v1(1, 2, 3, 4);
		CHECK_APPROX_EQ(v1, vx::Vec4(1, 2, 3, 4));



		v1.ToZero();
		CHECK_APPROX_EQ(v1, vx::Vec4(0.0f));

		vx::Vec4 zero = vx::Vec4::Zero();
		CHECK_APPROX_EQ(zero, vx::Vec4(0.0f));
	}


	TEST_CASE("Test basic arithematc")
	{
		vx::Vec4 a = { 1, 2, 3, 4 };
		vx::Vec4 b = { 4, 3, 2, 1 };

		vx::Vec4 c = a + b;
		CHECK_APPROX_EQ(c, vx::Vec4(5.0f));

		c = c - b;
		CHECK_APPROX_EQ(c, vx::Vec4(1.0f, 2.0f, 3.0f, 4.0f));


		a = { 1, 2, 3, 4 };
		a *= 2.0f;
		CHECK_APPROX_EQ(a, vx::Vec4(2.0f, 4.0f, 6.0f, 8.0));

		a = { 2, 4, 6, 8 };
		a = a / 2;
		CHECK_APPROX_EQ(a, vx::Vec4(1.0f, 2.0f, 3.0f, 4.0f));


		b = { 2, 4, 6, 8 };
		b /= 2;
		CHECK_APPROX_EQ(b, vx::Vec4(1.0f, 2.0f, 3.0f, 4.0f));

	}


	TEST_CASE("Equal AND Not Equal")
	{
		vx::Vec4 v(1, 2, 3, 4);

		CHECK(v == vx::Vec4(1, 2, 3, 4));
		CHECK(v != vx::Vec4(2, 2, 3, 4));
	}

	TEST_CASE("Min Max")
	{
		//check min 
		//on x
		CHECK_APPROX_EQ(vx::Vec4(2, 4, 34, 200).MinComponent(), 2.0f);
		//on y
		CHECK_APPROX_EQ(vx::Vec4(200, 2, 4, 34).MinComponent(), 2.0f);
		//on z
		CHECK_APPROX_EQ(vx::Vec4(34, 200, 2, 4).MinComponent(), 2.0f);
		//on w
		CHECK_APPROX_EQ(vx::Vec4(4, 34, 200, 2).MinComponent(), 2.0f);


		//check max
		//on x
		CHECK_APPROX_EQ(vx::Vec4(200, 2, 4, 34).MaxComponent(), 200.0f);
		//on y
		CHECK_APPROX_EQ(vx::Vec4(34, 200, 2, 4).MaxComponent(), 200.0f);
		//on z
		CHECK_APPROX_EQ(vx::Vec4(4, 34, 200, 2).MaxComponent(), 200.0f);
		//on w
		CHECK_APPROX_EQ(vx::Vec4(2, 4, 34, 200).MaxComponent(), 200.0f);


		//max axis
		//on x
		CHECK(vx::Vec4(200, 2, 4, 34).MaxAxis() == 0);
		//on y
		CHECK(vx::Vec4(20, 200, 4, 34).MaxAxis() == 1);
		//on z
		CHECK(vx::Vec4(20, 2, 400, 34).MaxAxis() == 2);
		//on w
		CHECK(vx::Vec4(2, 2, 4, 34).MaxAxis() == 3);

		//min axis
		//on x
		CHECK(vx::Vec4(2, 20, 4, 34).MinAxis() == 0);
		//on y
		CHECK(vx::Vec4(20, 2, 400, 34).MinAxis() == 1);
		//on z
		CHECK(vx::Vec4(20, 200, 4, 34).MinAxis() == 2);
		//on w
		CHECK(vx::Vec4(20, 20, 40, 3).MinAxis() == 3);


		vx::Vec4 a = vx::Vec4(3.0f, 0.0f, 4.0f, 1.0f);
		vx::Vec4 b = vx::Vec4(1.0f, 3.0f, 2.0f, 5.0f);

		CHECK_APPROX_EQ(vx::Vec4::Min(a, b), vx::Vec4(1.0f, 0.0f, 2.0f, 1.0f));
		CHECK_APPROX_EQ(vx::Vec4::Min(a, b), vx::Vec4::Min(b, a));
		CHECK_APPROX_EQ(vx::Vec4::Max(a, b), vx::Vec4(3.0f, 3.0f, 4.0f, 5.0f));
		CHECK_APPROX_EQ(vx::Vec4::Max(a, b), vx::Vec4::Max(b, a));
	}


	TEST_CASE("Vector operations")
	{
		CHECK(vx::Vec4::Dot(vx::Vec4(1, 2, 3, 4), vx::Vec4(1, 2, 3, 4)) ==
			static_cast<float>(1 * 1 + 2 * 2 + 3 * 3 + 4 * 4));
		CHECK(vx::Vec4(1, 2, 3, 4).Dot(vx::Vec4(1, 2, 3, 4)) ==
			static_cast<float>(1 * 1 + 2 * 2 + 3 * 3 + 4 * 4));

		CHECK(vx::Vec4::Dot(vx::Vec4(1, 2, 3, 4), vx::Vec4(5, 6, 7, 8)) ==
			static_cast<float>(1 * 5 + 2 * 6 + 3 * 7 + 4 * 8));
		CHECK(vx::Vec4(1, 2, 3, 4).Dot(vx::Vec4(5, 6, 7, 8)) ==
			static_cast<float>(1 * 5 + 2 * 6 + 3 * 7 + 4 * 8));


		//cross 3 
		float lx = 1.0f;
		float ly = 4.2f;
		float lz = 3.2f;

		float rx = 6.3f;
		float ry = 45.0f;
		float rz = 12.f;

		vx::Vec3 cross(
			(ly * rz) - (ry * lz),
			(rx * lz) - (lx * rz),
			(lx * ry) - (rx * ly)
		);


		bool error = vx::Vec3(3.0f) == vx::Vec3(3.0);
		CHECK_APPROX_EQ(vx::Vec4::Cross3(vx::Vec4(lx, ly, lz), vx::Vec4(rx, ry, rz)), cross);


		CHECK_APPROX_EQ(vx::Vec4::Cross3(vx::Vec4(1, 0, 0), vx::Vec4(0, 1, 0)), vx::Vec3(0, 0, 1));
		CHECK_APPROX_EQ(vx::Vec4::Cross3(vx::Vec4(0, 1, 0), vx::Vec4(1, 0, 0)), vx::Vec3(0, 0, -1));
		CHECK_APPROX_EQ(vx::Vec4::Cross3(vx::Vec4(0, 1, 0), vx::Vec4(0, 0, 1)), vx::Vec3(1, 0, 0));
		CHECK_APPROX_EQ(vx::Vec4::Cross3(vx::Vec4(0, 0, 1), vx::Vec4(0, 1, 0)), vx::Vec3(-1, 0, 0));
		CHECK_APPROX_EQ(vx::Vec4::Cross3(vx::Vec4(0, 0, 1), vx::Vec4(1, 0, 0)), vx::Vec3(0, 1, 0));
		CHECK_APPROX_EQ(vx::Vec4::Cross3(vx::Vec4(1, 0, 0), vx::Vec4(0, 0, 1)), vx::Vec3(0, -1, 0));
	}

	TEST_CASE("vector length")
	{
		float x = 1.0f;
		float y = 2.0f;
		float z = 2.0f;
		float w = 3.0f;
		float scalar_len_sq = static_cast<float>(x * x + y * y + z * z + w * w);
		CHECK_APPROX_EQ(vx::Vec4(x, y, z, w).LengthSq(), scalar_len_sq);
		CHECK_APPROX_EQ(vx::Vec4(x, y, z, w).Length(), std::sqrt(scalar_len_sq));
	}

	TEST_CASE("vector normalise")
	{
		float x = 1.0f;
		float y = 2.0f;
		float z = 2.0f;
		float w = 3.0f;
		float scalar_len = std::sqrt(static_cast<float>(x * x + y * y + z * z + w * w));
		vx::Vec4 scalar_nor = vx::Vec4(x, y, z, w) / scalar_len;
		CHECK_APPROX_EQ(vx::Vec4(x, y, z, w).Normalised(), scalar_nor);
		CHECK_APPROX_EQ(vx::Vec4(x, y, z, w).Normalise(), scalar_nor);
	}



	TEST_CASE("vector invert")
	{
		CHECK_APPROX_EQ(vx::Vec4(1, 2, 3, 4).Inverted(), vx::Vec4(-1, -2, -3, -4));
		CHECK_APPROX_EQ(vx::Vec4(1, 2, 3, 4).Invert(), vx::Vec4(-1, -2, -3, -4));
	}


	TEST_CASE("vector util")
	{
		CHECK_APPROX_EQ(vx::Vec4(-1, -2, -3, -4).Abs(), vx::Vec4(1, 2, 3, 4));
		CHECK_APPROX_EQ(vx::Vec4(-1, 2, -3, 4).Sign(), vx::Vec4(-1,1, -1, 1));
		CHECK_APPROX_EQ(vx::Vec4(-1, -2, -3, -4).Sign(), vx::Vec4(-1));
		CHECK_APPROX_EQ(vx::Vec4(1, 12, 3, 4).Sign(), vx::Vec4(1));


		float x = 1.0f;
		float y = 2.0f;
		float z = 3.0f;
		float w = 4.0f;

		CHECK_APPROX_EQ(vx::Vec4(x, y, z, w).Reciprocal(), vx::Vec4(1.0f / x, 1.0f / y, 1.0f / z, 1.0f/w));

		CHECK_APPROX_EQ(-vx::Vec4(x, y, z, w), vx::Vec4(-x, -y, -z, -w));

		CHECK_APPROX_EQ(vx::Vec4::Broadcast(2.0f), vx::Vec4(2.0f));


		vx::Vec4 xyzw = vx::Vec4(x, y, z, w);
		CHECK_APPROX_EQ(xyzw.SplatX(), vx::Vec4(x));
		CHECK_APPROX_EQ(xyzw.SplatY(), vx::Vec4(y));
		CHECK_APPROX_EQ(xyzw.SplatZ(), vx::Vec4(z));
		CHECK_APPROX_EQ(xyzw.SplatW(), vx::Vec4(w));


		CHECK_APPROX_EQ(xyzw.XYZ(), vx::Vec3(x, y, z));
		CHECK_APPROX_EQ(xyzw.XYZZ(), vx::Vec4(x, y, z, z));
		CHECK_APPROX_EQ(xyzw.XYZ0(), vx::Vec4(x, y, z, 0.0f));
		CHECK_APPROX_EQ(xyzw.XYZ1(), vx::Vec4(x, y, z, 1.0f));

		///Get Lane 
		vx::Vec4 t = vx::Vec4(1, 2, 3, 4);
		CHECK(vx::Vec4::GetLane(t, 0) == t[0]);
		CHECK(vx::Vec4::GetLane(t, 1) == t[1]);
		CHECK(vx::Vec4::GetLane(t, 2) == t[2]);
		CHECK(vx::Vec4::GetLane(t, 3) == t[3]);
	}
}