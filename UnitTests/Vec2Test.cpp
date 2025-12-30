#include "TestCommon.h"

TEST_SUITE("Vec2 Tests")
{
	TEST_CASE("Test Vec2 construct")
	{
		vx::Vec2 a;
		vx::Vec2 b(1.0f, 2.0f);
		vx::Vec2 c(3.0f);

		CHECK(b.X() == 1.0f);
		CHECK(b.Y() == 2.0f);
		CHECK(c.X() == 3.0f);
		CHECK(c.Y() == 3.0f);

		CHECK(b == vx::Vec2(1.0f, 2.0f));

		CHECK(vx::Vec2::Right() == vx::Vec2(1.0f, 0.0f));
		CHECK(vx::Vec2::Up() == vx::Vec2(0.0f, 1.0f));
	}


	TEST_CASE("Zero Ones")
	{
		vx::Vec2 zero = vx::Vec2::Zero();

		CHECK(zero.IsZero());
		CHECK(zero == vx::Vec2(0.0f));

		vx::Vec2 one = vx::Vec2::One();
		CHECK(!one.IsZero());
	}

	TEST_CASE("bast arithemetic ops")
	{
		vx::Vec2 a(1.0f, 2.0f);
		vx::Vec2 b(3.0f, 4.0f);

		CHECK_APPROX_EQ((a + b), vx::Vec2(4.0f, 6.0f));
		CHECK_APPROX_EQ((b - a), vx::Vec2(2.0f));
		CHECK_APPROX_EQ((a - b), vx::Vec2(-2.0f));
		CHECK_APPROX_EQ((a * 2.0f), vx::Vec2(2.0f, 4.0f));
		CHECK_APPROX_EQ((2.0f * a), vx::Vec2(2.0f, 4.0f));
		CHECK_APPROX_EQ((b / 2.0f), vx::Vec2(1.5f, 2.0f));

		//negate
		CHECK_APPROX_EQ((a - b), -(b-a));
	}


	TEST_CASE("Vector ops")
	{
		vx::Vec2 v0(1.0f, 0.0f);
		vx::Vec2 v1(0.0f, 1.0f);

		CHECK_APPROX_EQ(v0.Dot(v1), 0.0f);
		CHECK_APPROX_EQ(v1.Dot(v1), 1.0f);
	}

	TEST_CASE("Vector length normalise")
	{
		vx::Vec2 v(3.0f, 4.0f);

		CHECK(v.Length() == 5.0f);

		vx::Vec2 n = v.Normalised();
		CHECK_APPROX_EQ(v.Normalise().Length(), 1.0f);
		CHECK(n.IsNormalised());
		CHECK_APPROX_EQ(n.Length(), 1.0f);
	}

	TEST_CASE("Angle")
	{
		vx::Vec2 x(1.0f, 0.0f);
		vx::Vec2 y(0.0f, 1.0f);

		CHECK_APPROX_EQ(x.Angle(y), vx::kVxPi * 0.5f);
		CHECK_APPROX_EQ(x.SignedAngle(y), vx::kVxPi * 0.5f);
		CHECK_APPROX_EQ(x.SignedAngle(y), -y.SignedAngle(x));
	}

	TEST_CASE("Projection")
	{
		vx::Vec2 v(3.0f, 4.0f);
		vx::Vec2 n(1.0f, 0.0f);

		vx::Vec2 p = v.Project(n);
		vx::Vec2 r = v.Reject(n);

		CHECK_APPROX_EQ(p, vx::Vec2(3.0f, 0.0f));
		CHECK_APPROX_EQ(r, vx::Vec2(0.0f, 4.0f));

		//orthogonality 
		CHECK_APPROX_EQ(p.Dot(r), 0.0f);
	}


	TEST_CASE("Reflection")
	{
		vx::Vec2 v(1.0f, -1.0f);
		vx::Vec2 n = vx::Vec2::Up();

		vx::Vec2 r = v.Reflect(n);
		CHECK_APPROX_EQ(r, vx::Vec2(1.0f, 1.0f));
	}

	TEST_CASE("Min Max Clamp")
	{
		vx::Vec2 a(1.0f, 5.0f);
		vx::Vec2 b(3.0f, 2.0f);

		CHECK(vx::Vec2::Min(a, b) == vx::Vec2(1.0f, 2.0f));
		CHECK(vx::Vec2::Max(a, b) == vx::Vec2(3.0f, 5.0f));

		vx::Vec2 v(4.0f, -1.0f);
		v = vx::Vec2::Clamp(v, vx::Vec2(0.0f, 0.0f), vx::Vec2(3.0f, 3.0f));
		CHECK(v == vx::Vec2(3.0f, 0.0f));
	}

	TEST_CASE("Util")
	{
		vx::Vec2 zero(0.0f);
		CHECK(!zero.IsNormalised());

		vx::Vec2 nan(vx::kQuietNaN, 1.0f);
		CHECK(nan.IsNaN());
	}


	TEST_CASE("Lerp")
	{
		vx::Vec2 a(4.5f, 1.4f);
		vx::Vec2 b(233.5f, 953.3f);

		CHECK_APPROX_EQ(vx::Vec2::Lerp(a, b, 0.0f), a);
		CHECK_APPROX_EQ(vx::Vec2::Lerp(a, b, 1.0f), b);

		CHECK_APPROX_EQ(vx::Vec2::Lerp(a, b, 0.5f), (a + b) * 0.5f);


		vx::Vec2 prev = vx::Vec2::Lerp(a, b, 0.0f);
		for (int i = 1; i <= 1.0f; ++i)
		{
			float t = i / 10.0f;
			vx::Vec2 curr = vx::Vec2::Lerp(a, b, t);

			CHECK((curr - prev).Dot(b - a) >= 0.0f);

			prev = curr;
		}


		//zero 
		CHECK_APPROX_EQ(vx::Vec2::Lerp(vx::Vec2::Zero(), b, 0.3f), b * 0.3f);

		//identical inputs
		CHECK_APPROX_EQ(vx::Vec2::Lerp(a, a, 0.7f), a);

		//negative t
		vx::Vec2 test0 = vx::Vec2::Lerp(a, b, -1.0f);
		vx::Vec2 test1 = (a - (b - a));
		CHECK_APPROX_EQ(vx::Vec2::Lerp(a, b, -1.0f), (a - (b - a)), 1e-3f);

		//out of bounds > 
		CHECK_APPROX_EQ(vx::Vec2::Lerp(a, b, 2.0f), (b + (b - a)));

	}
}