#pragma once
#include <vector>
#include <memory>

#include "Playground.h"

class ScenarioManager
{
public:
	void AddScenario(std::unique_ptr<Scenario> scenario)
	{
		mScenarios.push_back(std::move(scenario));
	}


	void RunAll()
	{
		for (auto& s : mScenarios)
		{
			if (bSpacing)
				std::cout << "\n\n";

			s->Init();
			s->Run();
			s->End();
		}
	}

private:
	std::vector < std::unique_ptr<Scenario>> mScenarios;
	bool bSpacing = true;
};