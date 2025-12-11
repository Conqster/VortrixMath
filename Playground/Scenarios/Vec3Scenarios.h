#pragma once
#include "Playground.h"


class Vec3Scenario : public Scenario
{
public:

	void Init() override
	{
		std::cout << "[Vec3 Scenario]\n";
	}
	void Run() override;
	void End() override
	{

	}
};
