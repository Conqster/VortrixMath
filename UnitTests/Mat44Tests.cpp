#include "TestCommon.h"
#include "Mat44.h"


TEST_SUITE("Mat44 Test")
{
	TEST_CASE("Mat44 Identity")
	{
		vx::Mat44 identity = vx::Mat44::Identity();

		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				if (i != j)
					CHECK_APPROX_EQ(identity(i, j), 0.0f);
				else
					CHECK_APPROX_EQ(identity(i, j), 1.0f);
	}


	TEST_CASE("Mat44 construct")
	{
		vx::Mat44 mat1(2.0f);
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				if (i != j)
					CHECK(mat1(i, j) == 0.0f);
				else
					CHECK(mat1(i, j) == 2.0f);


		vx::Mat44 mat2(vx::Vec4(0.0f, 1.0f, 2.0f, 3.0f), vx::Vec4(4.0f, 5.0f, 6.0f, 7.0f),
			vx::Vec4(8.0f, 9.0f, 10.0f, 11.0), vx::Vec4(12.0f, 13.0f, 14.0f, 15.0f));

		CHECK(mat2(0, 0) == 0.0f);
		CHECK(mat2(1, 0) == 1.0f);
		CHECK(mat2(2, 0) == 2.0f);
		CHECK(mat2(3, 0) == 3.0f);

		CHECK(mat2(0, 1) == 4.0f);
		CHECK(mat2(1, 1) == 5.0f);
		CHECK(mat2(2, 1) == 6.0f);
		CHECK(mat2(3, 1) == 7.0f);
	
		CHECK(mat2(0, 2) == 8.0f);
		CHECK(mat2(1, 2) == 9.0f);
		CHECK(mat2(2, 2) == 10.0f);
		CHECK(mat2(3, 2) == 11.0f);

		CHECK(mat2(0, 3) == 12.0f);
		CHECK(mat2(1, 3) == 13.0f);
		CHECK(mat2(2, 3) == 14.0f);
		CHECK(mat2(3, 3) == 15.0f);

		vx::Mat44 mat3(vx::Vec4(0.0f, 1.0f, 2.0f, 3.0f), vx::Vec4(4.0f, 5.0f, 6.0f, 7.0f),
			vx::Vec4(8.0f, 9.0f, 10.0f, 11.0));

		CHECK(mat3(0, 0) == 0.0f);
		CHECK(mat3(1, 0) == 1.0f);
		CHECK(mat3(2, 0) == 2.0f);
		CHECK(mat3(3, 0) == 3.0f);

		CHECK(mat3(0, 1) == 4.0f);
		CHECK(mat3(1, 1) == 5.0f);
		CHECK(mat3(2, 1) == 6.0f);
		CHECK(mat3(3, 1) == 7.0f);

		CHECK(mat3(0, 2) == 8.0f);
		CHECK(mat3(1, 2) == 9.0f);
		CHECK(mat3(2, 2) == 10.0f);
		CHECK(mat3(3, 2) == 11.0f);

		CHECK(mat3(0, 3) == 0.0f);
		CHECK(mat3(1, 3) == 0.0f);
		CHECK(mat3(2, 3) == 0.0f);
		CHECK(mat3(3, 3) == 1.0f);

		vx::Mat44 mat4 = mat3;

		CHECK(mat3 == mat4);
		mat4(1, 3) = 3.0f; //[3][1]
		CHECK(mat3 != mat4);
	}

	TEST_CASE("Basis")
	{
		vx::Vec3 x = vx::Vec3::Right();
		vx::Vec3 y = vx::Vec3::Up();
		vx::Vec3 z = vx::Vec3::Forward();

		CHECK(vx::Mat44::Basis(x, y, z) == vx::Mat44(vx::Vec4(x, 0.0f), 
													 vx::Vec4(y, 0.0f), 
													 vx::Vec4(z, 0.0f)));
	}


	TEST_CASE("Translation")
	{
		vx::Vec3 t(2.0f, 4.6f, 3.0f);

		CHECK(vx::Mat44::Translation(t) == vx::Mat44(vx::Vec4(1.0f, 0.0f, 0.0f, 0.0f),
													 vx::Vec4(0.0f, 1.0f, 0.0f, 0.0f),
													 vx::Vec4(0.0f, 0.0f, 1.0f, 0.0f),
													 vx::Vec4(t, 1.0f)));
	}

	TEST_CASE("BasisTranslation")
	{
		vx::Vec3 x = vx::Vec3::Right();
		vx::Vec3 y = vx::Vec3::Up();
		vx::Vec3 z = vx::Vec3::Forward();
		vx::Vec3 t(2.0f, 4.6f, 3.0f);

		CHECK(vx::Mat44::BasisTranslation(x, y, z, t) == vx::Mat44(vx::Vec4(x, 0.0f),
													 vx::Vec4(y, 0.0f),
													 vx::Vec4(z, 0.0f),
													 vx::Vec4(t, 1.0f)));
	}



	TEST_CASE("Scale")
	{
		CHECK(vx::Mat44::Scale(5.0f) == vx::Mat44(vx::Vec4(5.0f, 0.0f, 0.0f, 0.0f),
										vx::Vec4(0.0f, 5.0f, 0.0f, 0.0f),
										vx::Vec4(0.0f, 0.0f, 5.0f, 0.0f),
										vx::Vec4(0.0f, 0.0f, 0.0f, 1.0f)));


		CHECK(vx::Mat44::Scale(vx::Vec3(2.0f, 4.0f, 6.0f)) == vx::Mat44(vx::Vec4(2.0f, 0.0f, 0.0f, 0.0f),
																		vx::Vec4(0.0f, 4.0f, 0.0f, 0.0f),
																		vx::Vec4(0.0f, 0.0f, 6.0f, 0.0f),
																		vx::Vec4(0.0f, 0.0f, 0.0f, 1.0f)));

		{

			vx::Mat44 M(vx::Vec4(1.0f, 0.0f, 0.0f, 0.0f),
				vx::Vec4(0.0f, 1.0f, 0.0f, 0.0f),
				vx::Vec4(0.0f, 0.0f, 1.0f, 0.0f),
				vx::Vec4(5.0f, 6.0f, 7.0f, 1.0f));

			vx::Vec3 scale(2.0f, 3.0f, 4.0f);

			{
				vx::Mat44 pre_scale_out(M);
				pre_scale_out.SetDiagonal3(scale); //<-- translation remains the same only diagonal of matrix
				//mimic local space scaling
				CHECK(M.PreScaled(scale) == pre_scale_out);
			}

			CHECK(M.PostScaled(scale) == vx::Mat44(vx::Vec4(2.0f, 0.0f, 0.0f, 0.0f),
				vx::Vec4(0.0f, 3.0f, 0.0f, 0.0f),
				vx::Vec4(0.0f, 0.0f, 4.0f, 0.0f),
				vx::Vec4(10.0f, 18.0f, 28.0f, 1.0f)));
		}


		{
			vx::Vec3 translate(5.0f, 0.0f, 0.0f);

			vx::Mat44 M = vx::Mat44::RotationZ(vx::DegToRad(45.0f));
			M.SetTranslation(translate);
			vx::Vec3 scale(2.0f, 3.0f, 4.0f);

			double sin_rad = vx::VxSin(vx::DegToRad(45.0f));
			double cos_rad = vx::VxCos(vx::DegToRad(45.0f));

			{
				vx::Mat44 M_pre = M.PreScaled(scale);
				CHECK(M_pre.GetTranslation() == translate);
				CHECK(M_pre(3, 3) == 1.0f);
				float test10 = M_pre(1, 0);
				float test00 = M_pre(0, 0);
				float test01 = M_pre(0, 1);
				CHECK_APPROX_EQ(M_pre(0, 0), float(cos_rad*scale[0]));
				CHECK_APPROX_EQ(M_pre(1, 0), float(sin_rad * scale[0]));
				CHECK_APPROX_EQ(M_pre(0, 1), -float(sin_rad * scale[1]));
				CHECK_APPROX_EQ(M_pre(1, 1), float(cos_rad * scale[1]));
			}

			{
				vx::Mat44 M_post = M.PostScaled(scale);
				CHECK(M_post.GetTranslation() == translate * scale);
				CHECK(M_post(3, 3) == 1.0f);
				CHECK_APPROX_EQ(M_post(0, 0), 1.414f, 1e-3f);
				CHECK_APPROX_EQ(M_post(1, 0), 2.121f, 1e-3f);
				CHECK_APPROX_EQ(M_post(0, 1), -1.414f, 1e-3f);
				CHECK_APPROX_EQ(M_post(1, 1), 2.121f, 1e-3f);
			}
		}
	}


	TEST_CASE("Matrix multiple operations")
	{
		vx::Mat44 R (vx::Vec4(0.0f, 1.0f, 2.0f, 3.0f), vx::Vec4(4.0f, 5.0f, 6.0f, 7.0f),
			vx::Vec4(8.0f, 9.0f, 10.0f, 11.0), vx::Vec4(12.0f, 13.0f, 14.0f, 15.0f));

		vx::Vec3 v(1.0f, 2.0f, 3.0f);

		CHECK(R.Multiply3x3(v) == vx::Vec3(32.0f, 38.0f, 44.0f));
		CHECK(R.Multiply3x3Transposed(v) == vx::Vec3(8.0f, 32.0f, 56.0f));
	}

	TEST_CASE("Matrix full & 3x3 Mutilpy")
	{
	
		vx::Mat44 A(
			vx::Vec4(1, 2, 3, 4),
			vx::Vec4(5, 6, 7, 8),
			vx::Vec4(9, 10, 11, 12),
			vx::Vec4(13, 14, 15, 16));

		vx::Mat44 B(
			vx::Vec4(17, 18, 19, 20),
			vx::Vec4(21, 22, 23, 24),
			vx::Vec4(25, 26, 27, 28),
			vx::Vec4(29, 30, 31, 32));


		vx::Mat44 C = A.Multiply(B);

		std::cout << A;

		CHECK(C.GetColumn(0) == vx::Vec4(538, 612, 686, 760));
		CHECK(C.GetColumn(1) == vx::Vec4(650, 740, 830, 920));
		CHECK(C.GetColumn(2) == vx::Vec4(762, 868, 974, 1080));
		CHECK(C.GetColumn(3) == vx::Vec4(874, 996, 1118, 1240));



		CHECK(A.Multiply3x3(B) == vx::Mat44(
											vx::Vec4(278, 332, 386, 0),
											vx::Vec4(338, 404, 470, 0),
											vx::Vec4(398, 476, 554, 0),
											vx::Vec4(0, 0, 0, 1)));

		CHECK(A.Multiply3x3LeftTransposed(B) == vx::Mat44(
											vx::Vec4(110, 326, 542, 0),
											vx::Vec4(134, 398, 662, 0),
											vx::Vec4(158, 470, 782, 0),
											vx::Vec4(0, 0, 0, 1)));

		CHECK(A.Multiply3x3(B) != A.Multiply3x3LeftTransposed(B));


		CHECK(A.Multiply3x3RightTransposed(B) == vx::Mat44(
												vx::Vec4(347, 410, 473, 0),
												vx::Vec4(362, 428, 494, 0),
												vx::Vec4(377, 446, 515, 0),
												vx::Vec4(0, 0, 0, 1)));
	}

	TEST_CASE("Matrix Mutilply Affine")
	{


		vx::Mat44 A(
			vx::Vec4(1, 2, 3, 0),
			vx::Vec4(4, 5, 6, 0),
			vx::Vec4(7, 8, 9, 0),
			vx::Vec4(10, 20, 30, 1));

		vx::Mat44 B(
			vx::Vec4(2, 0, 0, 0),
			vx::Vec4(0, 3, 0, 0),
			vx::Vec4(0, 0, 4, 0),
			vx::Vec4(1, 2, 3, 1));


		vx::Mat44 C = A.MultiplyAffine(B);

		/// 3x3 block
		CHECK(C.GetAxisX() == vx::Vec3(2, 4, 6));
		CHECK(C.GetAxisY() == vx::Vec3(12, 15, 18));
		CHECK(C.GetAxisZ() == vx::Vec3(28, 32, 36));

		CHECK(C.GetTranslation() == vx::Vec3(40, 56, 72));

		///affine invariant
		CHECK(C(3, 0) == 0.0f);
		CHECK(C(3, 1) == 0.0f);
		CHECK(C(3, 2) == 0.0f);
		CHECK(C(3, 3) == 1.0f);

		// |4 3 5|
		// |3 2 2|
		// |1 0 1|
	}


	TEST_CASE("Matrix Transposed")
	{


		vx::Mat44 A(
			vx::Vec4(1, 2, 3, 0),
			vx::Vec4(4, 5, 6, 0),
			vx::Vec4(7, 8, 9, 0),
			vx::Vec4(10, 20, 30, 1));

		std::cout << A << "\n";
		std::cout << A.Transposed() << "\n";

		CHECK(A.Transposed() == vx::Mat44(
			vx::Vec4(1, 4, 7, 10),
			vx::Vec4(2, 5, 8, 20),
			vx::Vec4(3, 6, 9, 30),
			vx::Vec4(0, 0, 0, 1)));

		CHECK(A.Transposed3x3() == vx::Mat44(
					vx::Vec4(1, 4, 7, 0),
					vx::Vec4(2, 5, 8, 0),
					vx::Vec4(3, 6, 9, 0),
					vx::Vec4(10, 20, 30, 1)));
	}

	TEST_CASE("Matrix Inverse")
	{

		CHECK(vx::Mat44::Identity().Inverse3x3() == vx::Mat44::Identity());

		vx::Mat44 A(
			vx::Vec4(1, 2, 3, 0),
			vx::Vec4(4, 5, 6, 0),
			vx::Vec4(7, 8, 10, 0),
			vx::Vec4(10, 20, 30, 1));



		//A * A^-1 == Identity
		CHECK(A.Multiply3x3(A.Inverse3x3()) == vx::Mat44::Identity());

		vx::Mat44 S;
		S.SetAxisX(vx::Vec3(2, 0, 0));
		S.SetAxisY(vx::Vec3(0, 4, 0));
		S.SetAxisZ(vx::Vec3(0, 0, 8));

		vx::Mat44 inv = S.Inverse3x3();
		CHECK(inv.GetAxisX() == vx::Vec3(0.5f, 0, 0));
		CHECK(inv.GetAxisY() == vx::Vec3(0, 0.25f, 0));
		CHECK(inv.GetAxisZ() == vx::Vec3(0, 0, 0.125f));


		std::cout << "A: \n" << A;
		std::cout << "inverse affine: \n" << A.InverseAffine();
		std::cout << "inverse 3x3: \n" << A.Inverse3x3();
		CHECK(A.Multiply3x3(A.InverseAffine()) == vx::Mat44::Identity());
	}


	TEST_CASE("Rotation X/Y/Z")
	{
		float half_pi = vx::kVxPi * 0.5f;
		vx::Mat44 R = vx::Mat44::RotationX(half_pi);
		CHECK_APPROX_EQ(R(1, 1), 0); //cos
		CHECK(R(1, 2) == -1);
		CHECK(R(2, 1) == 1);
		CHECK_APPROX_EQ(R(2, 2), 0);
		vx::Vec3 v = vx::Vec3::Up();

		CHECK_APPROX_EQ(R.Multiply3x3(v), vx::Vec3::Forward());

		R = vx::Mat44::RotationY(half_pi);
		CHECK_APPROX_EQ(R(0, 0), 0); //cos
		CHECK(R(2, 0) == -1);
		CHECK(R(0, 2) == 1);
		CHECK_APPROX_EQ(R(2, 2), 0);
		v = vx::Vec3::Right();
		CHECK_APPROX_EQ(R.Multiply3x3(v), -vx::Vec3::Forward());

		R = vx::Mat44::RotationZ(half_pi);
		CHECK_APPROX_EQ(R(0, 0), 0); //cos
		CHECK(R(1, 0) == 1);
		CHECK(R(0, 1) == -1);
		CHECK_APPROX_EQ(R(1, 1), 0);
		CHECK_APPROX_EQ(R.Multiply3x3(v), vx::Vec3::Up());
	}



	TEST_CASE("Column/Axes/Diagonal Accessor")
	{
		vx::Vec4 col0(0.0f, 1.0f, 2.0f, 3.0f);
		vx::Vec4 col1(4.0f, 5.0f, 6.0f, 7.0f);
		vx::Vec4 col2(8.0f, 9.0f, 10.0f, 11.0f);
		vx::Vec4 col3(12.0f, 13.0f, 14.0f, 15.0f);
		vx::Mat44 M(col0, col1, col2, col3);

		//for(int i = 0; i < 4; ++i)
		CHECK(M.GetColumn(0) == col0);
		CHECK(M.GetColumn(1) == col1);
		CHECK(M.GetColumn(2) == col2);
		CHECK(M.GetColumn(3) == col3);

		CHECK(M.GetColumn3(0) == col0.XYZ());
		CHECK(M.GetColumn3(1) == col1.XYZ());
		CHECK(M.GetColumn3(2) == col2.XYZ());
		CHECK(M.GetColumn3(3) == col3.XYZ());

		M.SetColumn3(0, vx::Vec3(20.0f, 30.0f, 40.0f));
		M.SetColumn3(1, vx::Vec3(60.0f, 70.0f, 80.0f));
		M.SetColumn3(2, vx::Vec3(100.0f, 110.0f, 120.0f));
		M.SetColumn3(3, vx::Vec3(140.0f, 150.0f, 160.0f));
		CHECK(M.GetColumn(0) == vx::Vec4(20.0f, 30.0f, 40.0f, 0.0f)); //need fix
		CHECK(M.GetColumn(1) == vx::Vec4(60.0f, 70.0f, 80.0f, 0.0f));
		CHECK(M.GetColumn(2) == vx::Vec4(100.0f, 110.0f, 120.0f, 0.0f));
		CHECK(M.GetColumn(3) == vx::Vec4(140.0f, 150.0f, 160.0f, 1.0f));


		M.SetColumn(0, vx::Vec4(1.0f, 3.2f, 0.0f, 0.0f));
		M.SetColumn(1, vx::Vec4(6.0f, 1.0f, 0.0f, 0.0f));
		M.SetColumn(2, vx::Vec4(0.0f, 23.0f, 1.0f, 0.0f));
		M.SetColumn(3, vx::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
		CHECK(M.GetColumn(0) == vx::Vec4(1.0f, 3.2f, 0.0f, 0.0f));
		CHECK(M.GetColumn(1) == vx::Vec4(6.0f, 1.0f, 0.0f, 0.0f));
		CHECK(M.GetColumn(2) == vx::Vec4(0.0f, 23.0f, 1.0f, 0.0f));
		CHECK(M.GetColumn(3) == vx::Vec4(0.0f, 0.0f, 0.0f, 1.0f));

		M.SetAxisX(vx::Vec3::Right());
		M.SetAxisY(vx::Vec3::Up());
		M.SetAxisZ(vx::Vec3::Forward());
		M.SetTranslation(vx::Vec3(0.0f, 1.0f, 2.0f));
		CHECK(M.GetAxisX() == vx::Vec3::Right());
		CHECK(M.GetAxisY() == vx::Vec3::Up());
		CHECK(M.GetAxisZ() == vx::Vec3::Forward());
		CHECK(M.GetTranslation() == vx::Vec4(0.0f, 1.0f, 2.0f));

		CHECK(M.GetDiagonal() == vx::Vec4(1.0f));
		CHECK(M.GetDiagonal3() == vx::Vec3(1.0f));

		M.SetDiagonal(vx::Vec4(0.0f, 1.0f, 2.0f, 3.0f));
		CHECK(M.GetDiagonal() == vx::Vec4(0.0f, 1.0f, 2.0f, 3.0f));
		M.SetDiagonal3(vx::Vec3(0.0f, 1.0f, 2.0f));
		CHECK(M.GetDiagonal() == vx::Vec4(0.0f, 1.0f, 2.0f, 1.0f));
	}



	TEST_CASE("Matric Handedness")
	{
		vx::Mat44 M = vx::Mat44::Identity();
		CHECK(M.GetBasisHandness() == 1);

		vx::Mat44 M1(vx::Vec4(0.0f, 1.0f, 2.0f, 3.0f), vx::Vec4(4.0f, 5.0f, 6.0f, 7.0f),
			vx::Vec4(8.0f, 9.0f, 10.0f, 11.0), vx::Vec4(12.0f, 13.0f, 14.0f, 15.0f));
		CHECK(M1.GetBasisHandness() == 0);
	}



	TEST_CASE("Utiltity function")
	{
		vx::Mat44 A(
			vx::Vec4(1, 2, 3, 0),
			vx::Vec4(4, 5, 6, 0),
			vx::Vec4(7, 8, 9, 0),
			vx::Vec4(10, 20, 30, 1));

		vx::Mat44 B(
			vx::Vec4(2, 0, 1, 0),
			vx::Vec4(0, 2, 1, 0),
			vx::Vec4(1, 1, 2, 0),
			vx::Vec4(5, 6, 7, 1));

		vx::Vec3 v(1, 2, 3);

		vx::Mat44 sum = A.Add(B);

		CHECK(sum(0, 0) == A(0, 0) + B(0, 0));
		CHECK(sum(1, 1) == A(1, 1) + B(1, 1));
		CHECK(sum(3, 3) == 2.0f);

		vx::Mat44 affine_sum = A.AddAffine(B);
		CHECK(affine_sum(1, 1) == A(1, 1) + B(1, 1));
		CHECK(affine_sum(3, 3) == 1.0f);


		vx::Mat44 skew = vx::Mat44::SkewSymmetric3x3(v);

		CHECK(skew(0, 1) == -v.Z());
		CHECK(skew(0, 2) == v.Y());
		CHECK(skew(1, 0) == v.Z());
		CHECK(skew(1, 2) == -v.X());

		//diagonal 3x3 = 0, affine 1.0
		CHECK(skew.GetDiagonal() == vx::Vec4(0.0f, 0.0f, 0.0f, 1.0f));


		vx::Mat44 valid_full_rank(
			vx::Vec4(1, 0, 0, 0),
			vx::Vec4(0, 2, 0, 0),
			vx::Vec4(0, 0, 3, 0),
			vx::Vec4(5, 6, 7, 1));
	
		vx::Vec3 scale;
		vx::Mat44 rot_translation = valid_full_rank.Decompose(scale);

		//ensure the decomposition produces rot matrix with 1
		CHECK(rot_translation(3, 3) == 1.0f);
		CHECK_APPROX_EQ(rot_translation.GetTranslation(), vx::Vec3(5, 6, 7));

		CHECK(scale.X() > 0);
		CHECK(scale.Y() > 0);
		CHECK(scale.Z() > 0);


		vx::Vec3 result = A.MultiplyAffine(v);

		//| 0 4 8 12 |
		//	| 1 5 9 13 |
		//	| 2 6 10 14 |
		//	| 3 7 11 15 |
		vx::Vec3 expected(
			A(0, 0) * v.X() + A(0, 1) * v.Y() + A(0, 2) * v.Z() + A(0, 3),
			A(1, 0) * v.X() + A(1, 1) * v.Y() + A(1, 2) * v.Z() + A(1, 3),
			A(2, 0) * v.X() + A(2, 1) * v.Y() + A(2, 2) * v.Z() + A(2, 3));

		CHECK(result == expected);


		vx::Mat44 transform(
			vx::Vec4(0.0f, -0.5f, 0.1f, 0.0f),
			vx::Vec4(0.5f, 0.866f, 0.2f, 0.0f),
			vx::Vec4(0.0f, 0.0f, 1.0f, 0.0f),
			vx::Vec4(10.0f, 20.0f, 30.0f, 1.0f));

		transform.MakeOrthonormal();

		vx::Vec3 t = transform.Transform(v);
		vx::Vec3 t_inv = transform.TransformInverse(t);

		vx::Vec3 test = transform.Transposed().MultiplyAffine(t);
		vx::Vec3 test2 = transform.Transposed3x3().MultiplyAffine(t);
		vx::Vec3 test3 = transform.Multiply3x3Transposed(t);
		//Tranform -> TransformInverse should recover original
		CHECK_APPROX_EQ(t_inv, v);

		CHECK(transform.Transform(v) == vx::Mat44::Transform(transform,v));
		CHECK(transform.TransformInverse(t) == vx::Mat44::TransformInverse(transform,t));


		vx::Vec3 dir = transform.TransformDirection(v);
		vx::Vec3 dir_inv = transform.TransformInverseDirection(dir);

		//TranformDir -> TransformInverseDir should recover original
		CHECK_APPROX_EQ(dir_inv, v);

	}


	TEST_CASE("Test Mat44 Quat Rotation")
	{
		vx::Quat q = vx::Quat::FromAxisAngle(vx::Vec3(0.3f, 0.7f, 0.5f).Normalised(), vx::kVxPi * 0.5f);

		vx::Mat44 R = vx::Mat44::Rotation(q);

		CHECK(R.IsOrthonormal());

		vx::Quat q_out = R.GetRotationQuat();

		CHECK_APPROX_EQ(q.Dot(q_out), 1.0f);

		CHECK(R.GetTranslation() == vx::Vec3::Zero());
		CHECK(R.IsAffine()); //ensure affine
	}


	TEST_CASE("Test Mat44 Rotation Translation")
	{
		vx::Quat q = vx::Quat::FromAxisAngle(vx::Vec3(0.3f, 0.7f, 0.5f).Normalised(), vx::kVxPi * 0.5f);
		vx::Vec3 t(1.0f, -2.0f, 3.0f);


		vx::Mat44 Rt = vx::Mat44::RotationTranslation(q, t);

		CHECK(Rt.IsOrthonormal());
		CHECK(Rt.GetTranslation() == t);
		CHECK(Rt.IsAffine()); //ensure affine

		vx::Quat q_out = Rt.GetRotationQuat();

		CHECK_APPROX_EQ(q.Dot(q_out), 1.0f);


		vx::Mat44 M(1.0f);
		M.SetTranslation(t);
		M.SetRotation(q);

		CHECK(M == Rt);
		vx::Mat44 M1(1.0f);
		M1.SetRotationAndTranslation(q, t);
		CHECK(M1 == Rt);


	}
}