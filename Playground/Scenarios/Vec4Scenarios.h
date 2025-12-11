#pragma once
#include "Playground.h"


class Vec4Scenario : public Scenario
{
public:

	void Init() override
	{
		std::cout << "[Vec4 Scenario, Hello SIMD test bed]\n";
	}
	void Run() override;
	void End() override
	{

	}



private:

	vx::Vec4 ScalarNormalisation(const vx::Vec4& v) const;
	vx::Vec4 Scalar_Vec4ScalarDivide(const vx::Vec4& v, float scalar) const;
	vx::Vec3 Scalar_Vec3Cross(const vx::Vec4& lhs, const vx::Vec4& rhs) const;
};
