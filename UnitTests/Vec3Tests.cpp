#include "TestCommon.h"

TEST_SUITE("Vec3 Tests")
{

	TEST_CASE("Test Vec3 construct")
	{
		vx::Vec3 v(4.0f);


		CHECK_APPROX_EQ(v.X(), 4.0f);
		CHECK_APPROX_EQ(v.Y(), 4.0f);
		CHECK_APPROX_EQ(v.Z(), 4.0f);


		vx::Vec3 v1(1, 2, 3);
		CHECK_APPROX_EQ(v1, vx::Vec3(1, 2, 3));



		v1.ToZero();
		CHECK_APPROX_EQ(v1, vx::Vec3(0.0f));

		vx::Vec3 zero = vx::Vec3::Zero();
		CHECK_APPROX_EQ(zero, vx::Vec3(0.0f));

		CHECK(zero.IsZero());


		vx::Vec3 one = vx::Vec3::One();
		CHECK_APPROX_EQ(one, vx::Vec3(1.0f));

		CHECK(!one.IsZero());


		CHECK_APPROX_EQ(vx::Vec3::Up(), vx::Vec3(0.0f, 1.0f, 0.0f));
		CHECK_APPROX_EQ(-vx::Vec3::Up(), vx::Vec3(0.0f, -1.0f, 0.0f)); //mimic down
		CHECK_APPROX_EQ(vx::Vec3::Right(), vx::Vec3(1.0f, 0.0f, 0.0f));
		CHECK_APPROX_EQ(vx::Vec3::Forward(), vx::Vec3(0.0f, 0.0f, 1.0f));
	}


	TEST_CASE("Test basic arithematc")
	{
		vx::Vec3 a = { 1, 2, 3 };
		vx::Vec3 b = { 4, 3, 2 };

		vx::Vec3 c = a + b;
		CHECK_APPROX_EQ(c, vx::Vec3(5.0f));

		c = c - b;
		CHECK_APPROX_EQ(c, vx::Vec3(1.0f, 2.0f, 3.0f));


		a = { 1, 2, 3 };
		a *= 2.0f;
		CHECK_APPROX_EQ(a, vx::Vec3(2.0f, 4.0f, 6.0));

		//rhs
		a = 0.5f * a;
		CHECK_APPROX_EQ(a, vx::Vec3(1.0f, 2.0f, 3.0));

		a = { 2, 4, 6 };
		a = a / 2;
		CHECK_APPROX_EQ(a, vx::Vec3(1.0f, 2.0f, 3.0f));


		b = { 2, 4, 6};
		b /= 2;
		CHECK_APPROX_EQ(b, vx::Vec3(1.0f, 2.0f, 3.0f));

	}


	TEST_CASE("Equal AND Not Equal")
	{
		vx::Vec3 v(1, 2, 3);

		CHECK(v == vx::Vec3(1, 2, 3));
		CHECK(v != vx::Vec3(2, 2, 3));
	}

	TEST_CASE("Min Max")
	{
		//check min 
		//on x
		CHECK_APPROX_EQ(vx::Vec3(2, 4, 34).MinComponent(), 2.0f);
		//on y
		CHECK_APPROX_EQ(vx::Vec3(2, 4, 34).MinComponent(), 2.0f);
		//on z
		CHECK_APPROX_EQ(vx::Vec3(34, 200, 2).MinComponent(), 2.0f);


		//check max
		//on x
		CHECK_APPROX_EQ(vx::Vec3(200, 2, 4).MaxComponent(), 200.0f);
		//on y
		CHECK_APPROX_EQ(vx::Vec3(200, 2, 4).MaxComponent(), 200.0f);
		//on z
		CHECK_APPROX_EQ(vx::Vec3(4, 34, 200).MaxComponent(), 200.0f);


		//max axis
		//on x
		CHECK(vx::Vec3(200, 2, 4).MaxAxis() == 0);
		//on y
		CHECK(vx::Vec3(20, 200, 4).MaxAxis() == 1);
		//on z
		CHECK(vx::Vec3(20, 2, 400).MaxAxis() == 2);

		//min axis
		//on x
		CHECK(vx::Vec3(2, 20, 4).MinAxis() == 0);
		//on y
		CHECK(vx::Vec3(20, 2, 400).MinAxis() == 1);
		//on z
		CHECK(vx::Vec3(20, 200, 4).MinAxis() == 2);


		vx::Vec3 a = vx::Vec3(3.0f, 0.0f, 4.0f);
		vx::Vec3 b = vx::Vec3(1.0f, 3.0f, 2.0f);

		CHECK_APPROX_EQ(vx::Vec3::Min(a, b), vx::Vec3(1.0f, 0.0f, 2.0f));
		CHECK_APPROX_EQ(vx::Vec3::Min(a, b), vx::Vec3::Min(b, a));
		CHECK_APPROX_EQ(vx::Vec3::Max(a, b), vx::Vec3(3.0f, 3.0f, 4.0f));
		CHECK_APPROX_EQ(vx::Vec3::Max(a, b), vx::Vec3::Max(b, a));
	}


	TEST_CASE("Vector operations")
	{
		CHECK(vx::Vec3::Dot(vx::Vec3(1, 2, 3), vx::Vec3(1, 2, 3)) ==
			static_cast<float>(1 * 1 + 2 * 2 + 3 * 3));
		CHECK(vx::Vec3(1, 2, 3).Dot(vx::Vec3(1, 2, 3)) ==
			static_cast<float>(1 * 1 + 2 * 2 + 3 * 3));

		CHECK(vx::Vec3::Dot(vx::Vec3(1, 2, 3), vx::Vec3(5, 6, 7)) ==
			static_cast<float>(1 * 5 + 2 * 6 + 3 * 7));
		CHECK(vx::Vec3(1, 2, 3).Dot(vx::Vec3(5, 6, 7)) ==
			static_cast<float>(1 * 5 + 2 * 6 + 3 * 7));


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
		CHECK_APPROX_EQ(vx::Vec3(lx, ly, lz).Cross(vx::Vec3(rx, ry, rz)), cross);
		CHECK_APPROX_EQ(vx::Vec3::Cross(vx::Vec3(lx, ly, lz), vx::Vec3(rx, ry, rz)), cross);


		CHECK_APPROX_EQ(vx::Vec3(1, 0, 0).Cross(vx::Vec3(0, 1, 0)), vx::Vec3(0, 0, 1));
		CHECK_APPROX_EQ(vx::Vec3(0, 1, 0).Cross(vx::Vec3(1, 0, 0)), vx::Vec3(0, 0, -1));
		CHECK_APPROX_EQ(vx::Vec3(0, 1, 0).Cross(vx::Vec3(0, 0, 1)), vx::Vec3(1, 0, 0));
		CHECK_APPROX_EQ(vx::Vec3(0, 0, 1).Cross(vx::Vec3(0, 1, 0)), vx::Vec3(-1, 0, 0));
		CHECK_APPROX_EQ(vx::Vec3(0, 0, 1).Cross(vx::Vec3(1, 0, 0)), vx::Vec3(0, 1, 0));
		CHECK_APPROX_EQ(vx::Vec3(1, 0, 0).Cross(vx::Vec3(0, 0, 1)), vx::Vec3(0, -1, 0));

		CHECK_APPROX_EQ(vx::Vec3::Cross(vx::Vec3(1, 0, 0), vx::Vec3(0, 1, 0)), vx::Vec3(0, 0, 1));
		CHECK_APPROX_EQ(vx::Vec3::Cross(vx::Vec3(0, 1, 0), vx::Vec3(1, 0, 0)), vx::Vec3(0, 0, -1));
		CHECK_APPROX_EQ(vx::Vec3::Cross(vx::Vec3(0, 1, 0), vx::Vec3(0, 0, 1)), vx::Vec3(1, 0, 0));
		CHECK_APPROX_EQ(vx::Vec3::Cross(vx::Vec3(0, 0, 1), vx::Vec3(0, 1, 0)), vx::Vec3(-1, 0, 0));
		CHECK_APPROX_EQ(vx::Vec3::Cross(vx::Vec3(0, 0, 1), vx::Vec3(1, 0, 0)), vx::Vec3(0, 1, 0));
		CHECK_APPROX_EQ(vx::Vec3::Cross(vx::Vec3(1, 0, 0), vx::Vec3(0, 0, 1)), vx::Vec3(0, -1, 0));
	}

	TEST_CASE("vector length")
	{
		float x = 1.0f;
		float y = 2.0f;
		float z = 2.0f;
		float scalar_len_sq = static_cast<float>(x * x + y * y + z * z);
		CHECK_APPROX_EQ(vx::Vec3(x, y, z).LengthSq(), scalar_len_sq);
		CHECK_APPROX_EQ(vx::Vec3(x, y, z).Length(), std::sqrt(scalar_len_sq));
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
		CHECK_APPROX_EQ(vx::Vec3(1, 2, 3).Inverted(), vx::Vec3(-1, -2, -3));
		CHECK_APPROX_EQ(vx::Vec3(1, 2, 3).Invert(), vx::Vec3(-1, -2, -3));
	}


	TEST_CASE("vector util")
	{
		CHECK_APPROX_EQ(vx::Vec3(-1, -2, -3).Abs(), vx::Vec3(1, 2, 3));
		CHECK_APPROX_EQ(vx::Vec3(-1, 2, -3).Sign(), vx::Vec3(-1, 1, -1));
		CHECK_APPROX_EQ(vx::Vec3(-1, -2, -3).Sign(), vx::Vec3(-1));
		CHECK_APPROX_EQ(vx::Vec3(1, 12, 3).Sign(), vx::Vec3(1));


		float x = 1.0f;
		float y = 2.0f;
		float z = 3.0f;

		CHECK_APPROX_EQ(vx::Vec3(x, y, z).Reciprocal(), vx::Vec3(1.0f / x, 1.0f / y, 1.0f / z));


		CHECK_APPROX_EQ(-vx::Vec3(x, y, z), vx::Vec3(-x, -y, -z));

		CHECK_APPROX_EQ(vx::Vec3::Broadcast(2.0f), vx::Vec3(2.0f));

		CHECK_APPROX_EQ(vx::Vec3(x, y, z).SplatX(), vx::Vec3(x));
		CHECK_APPROX_EQ(vx::Vec3(x, y, z).SplatY(), vx::Vec3(y));
		CHECK_APPROX_EQ(vx::Vec3(x, y, z).SplatZ(), vx::Vec3(z));


		CHECK_APPROX_EQ(vx::Vec3(49, 121, 25).Sqrt(), vx::Vec3(7, 11, 5));
		vx::Vec3 v(49, 121, 25);
		CHECK_APPROX_EQ(v.SqrtAssign(), vx::Vec3(7, 11, 5));
	}
}