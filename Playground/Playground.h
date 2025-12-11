#pragma once
#include <iostream>
#include "VortrixMaths.h"


class Scenario
{
public:

	Scenario()
	{
		std::cout << "Playground Scenario\n";
	}

	virtual void Init() = 0;
	virtual void Run() = 0;
	virtual void End() = 0;

	// NonCopyable
	Scenario(const Scenario&) = delete;
	void operator=(const Scenario&) = delete;

	~Scenario()
	{

	}
};
