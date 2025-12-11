#include "Vec4Scenarios.h"



void Vec4Scenario::Run()
{
	std::cout << "Hello Maths SIMD test bed!!!!" << std::endl;	
	vx::Vec4 vec4;	
	vec4 = vx::Vec4(1.0f, 2.0f, 3.0f, 4.0f);	
	std::cout << "Test vec4: " << vec4 << std::endl;	
	std::cout << "Vec4 scalar 4" << vx::Vec4(4) << "\n";	
	std::cout << "Addition vec4 with (1.0f, 1.0f, 1.0f, 1.0f): " << vec4 + vx::Vec4(1.0f) << "\n";	
	//Vec4 temp = vec4 + V	std::cout << "Max component of 200, 4, 34, 4: " << vx::Vec4(200, 4, 34, 4).MaxComponent() << "\n";
	std::cout << "Max component of 0, 40, 34, 4: " << vx::Vec4(0, 40, 34, 4).MaxComponent() << "\n";
	std::cout << "Max component of 2, 4, 34, 4: " << vx::Vec4(2, 4, 34, 4).MaxComponent() << "\n";
	std::cout << "Max component of 2, 4, 34, 400: " << vx::Vec4(2, 4, 34, 400).MaxComponent() << "\n";
	std::cout << "Dot of Vec4(1.0f, 2.0f, 3.0f, 4.0f) with self (Should be 30): " <<
			vx::Vec4::Dot(vx::Vec4(1.0f, 2.0f, 3.0f, 4.0f), vx::Vec4(1.0f, 2.0f, 3.0f, 4.0f)) << ".\n";
	std::cout << "Dot of Vec4(1.0f, 1.0f, 1.0f, 1.0f) with self (Should be 4): " <<	
		vx::Vec4(1.0f, 1.0f, 1.0f, 1.0f).Dot(vx::Vec4(1.0f, 1.0f, 1.0f, 1.0f)) << ".\n";

	auto experiment_dot_simd = [](const vx::Vec4& v0, const vx::Vec4& v1) {
		std::cout << "\n\n";
		float expected_result = (v0.X() * v1.X()) + (v0.Y() * v1.Y()) + (v0.Z() * v1.Z()) + (v0.W() * v1.W());
		std::cout << "Trying to perform dot product on " << v0 << " & " << v0 <<
			", expected result: " << expected_result << ".\n";
		__m128 r = _mm_mul_ps(v0.Value(), v1.Value());
		std::cout << "Trying to multply their component, result " << vx::Vec4(r) << ".\n";
		float simd_result = FLT_MAX;
		std::cout << "Trying to multply t " << vx::Vec4(r) << ".\n";
		/// 0x71 -> 0111 0001 : op first 3 & store 1 (first)
		/// 0xf1 -> 1111 0001 : op first 4 & store 1 (first)
		/// 
		/// 0x77 -> 0111 0111 : op first 3 & store 3 (first)
		/// 0xff -> 1111 1111 : op first 4 & store 4 (first)
		/// 
		/// 0x7f -> 0111 1111 : op first 4 & store 4 (first)
		/// 
		/// as 0111 0001 
		/// high nibble 0111 [bit 4 - 7] (nibble 1 = 4bits, 0.5bytes) 
		/// low nibble 0001	 [bit 0 - 3]
		/// using with _mm_dp_ps 
		/// high nibble defines, the bits to op on (multply its components) 
		/// 0111 x, y, z, without w 
		/// low nibble defines, the bits to store result
		/// 0001 only x excluding y, z, and w

		//dot product op first 4 & store 1 (x) then extract 1 (0:x)
		simd_result = _mm_cvtss_f32(_mm_dp_ps(v0.Value(), v1.Value(), 0xf1));
		std::cout << "Vec 3 dp multiply [op first 3 & store 1 (first)]: " <<
			vx::Vec4(_mm_dp_ps(v0.Value(), v1.Value(), 0x71));
		std::cout << "\n";
		std::cout << "Vec 4 dp multiply [op first 4 & store 1 (first)]: " << vx::Vec4(_mm_dp_ps(v0.Value(), v1.Value(), 0xf1));
		std::cout << "\n";
		std::cout << "Vec 3 dp multiply [op first 3 & store 3 (first)]: " << vx::Vec4(_mm_dp_ps(v0.Value(), v1.Value(), 0x77));
		std::cout << "\n";
		std::cout << "Vec 4 dp multiply [op first 4 & store 4 (first)]: " << vx::Vec4(_mm_dp_ps(v0.Value(), v1.Value(), 0xff));
		std::cout << "\n";
		std::cout << "Vec 3 dp multiply [op first 3 & store 4 (first)]: " << vx::Vec4(_mm_dp_ps(v0.Value(), v1.Value(), 0x7f));
		std::cout << "\n";		vx::Vec4 a(1, 2, 3, 4);
		vx::Vec4 b(5, 6, 7, 8);
		std::cout << "a: " << a << " & b: " << b << "\n";
		a = vx::Vec4(_mm_shuffle_ps(a.Value(), b.Value(), _MM_SHUFFLE(3, 3, 2, 2)));
	};
		
	vx::Vec4 t0 = vx::Vec4(1, 2, 3, 4);
	vx::Vec4 t1 = vx::Vec4(2, 2, 2, 2);
	experiment_dot_simd(t0, t1);

	std::cout << "=========ZERO===========\n";
	std::cout << t0 << " to zero: ";
	t0.ToZero();
	std::cout << t0 << "\n";
	std::cout << "=========Length===========\n";
	std::cout << "Length Square of " << t1 << ",(expected " <<
		(t1.X() * t1.X() + t1.Y() * t1.Y() + t1.Z() * t1.Z() + t1.W() * t1.W()) << "): " << t1.LengthSq() << "\n";
	std::cout << "Length of " << t1 << ",(expected " <<
		std::sqrt(t1.X() * t1.X() + t1.Y() * t1.Y() + t1.Z() * t1.Z() + t1.W() * t1.W()) << "): " << t1.Length() << "\n";



	std::cout << "=========Normalised===========\n";
	std::cout << "Normalised of " << t1 << ",(expected " << ScalarNormalisation(t1) << "): " <<
		t1.Normalised() << "\n";

	std::cout << "=========Invert===========\n";
	std::cout << "Inverted of " << t1 << ",(expected " <<
		vx::Vec4(-t1.X(), -t1.Y(), -t1.Z(), -t1.W()) << "): " << t1.Inverted() << "\n";
	std::cout << "=========Cross===========\n";
	vx::Vec4 t2(1.0f, 23, 12, 1.0f);
	std::cout << "Cross of " << t1 << "with " << t2 << ",(expected " <<
		Scalar_Vec3Cross(t1, t2) << "): " << vx::Vec4::Cross3(t1, t2) << "\n";
	std::cout << "=========DIV===========\n";
	vx::Vec4 t3(4.0f);
	std::cout << "DIV / of " << t3 << "with " << 2 << ",(expected " << Scalar_Vec4ScalarDivide(t3, 2) << "): " <<	t3/2 <<  "\n";

	std::cout << "Mult *= " << t3 << "with 2 := ";
	t3 *= 2;
	std::cout << t3 << ".\n";

	std::cout << "DIV /= of " << t3 << "with " << 2 << ",(expected " << Scalar_Vec4ScalarDivide(t3, 2) << "): " <<
		(t3 /= 2) << "\n";

	std::cout << "=========GET LANE===========\n";
	t3 = vx::Vec4(2, 4, 6, 8);
	std::cout << t3 << "lanes : ";
	std::cout << "\n\t lane 0: " << vx::Vec4::GetLane(t3, 0) <<
		"\n\t lane 1: " << vx::Vec4::GetLane(t3, 1)	<< "\n\t lane 2: " <<
		vx::Vec4::GetLane(t3, 2) << "\n\t lane 3: " << vx::Vec4::GetLane(t3, 3) << "\n";
	std::cout << t3 << "components x, y, z, w : ";
	std::cout << "\n\t comp x: " << t3.X() << "\n\t comp y: " << t3.Y() <<
		"\n\t comp z: " << t3.Z() << "\n\t comp w: " << t3.W() << "\n";

}

vx::Vec4 Vec4Scenario::ScalarNormalisation(const vx::Vec4& v) const
{
	float length_sq = (v.X() * v.X() + v.Y() * v.Y() + v.Z() * v.Z() + v.W() * v.W());
	vx::Vec4 result = v;
	if (length_sq > 1e-6f)
	{
		const float inv = 1.0f / std::sqrt(length_sq);
		for (int i = 0; i < 4; ++i)
			result[i] *= inv;
	}
	return result;
}

vx::Vec4 Vec4Scenario::Scalar_Vec4ScalarDivide(const vx::Vec4& v, float scalar) const
{
	vx::Vec4 t = v;
	const float inv = 1.0f / scalar;
	for (int i = 0; i < 4; ++i)
		t[i] *= inv;
	return t;
}

vx::Vec3 Vec4Scenario::Scalar_Vec3Cross(const vx::Vec4& lhs, const vx::Vec4& rhs) const
{
	vx::Vec3 temp = vx::Vec3::Zero();
	temp[0] = (lhs.Y() * rhs.Z()) - (rhs.Y() * lhs.Z());
	temp[1] = (rhs.X() * lhs.Z()) - (lhs.X() * rhs.Z());
	temp[2] = (lhs.X() * rhs.Y()) - (rhs.X() * lhs.Y());
	return temp;
}
