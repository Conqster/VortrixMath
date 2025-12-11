


#include "Core.h"

#include "Playground.h"

#include "Scenarios/ScenarioManager.h"
#include "Scenarios/Vec3Scenarios.h"
#include "Scenarios/Vec4Scenarios.h"



int main()
{
	ScenarioManager manager;

	//Add scenarios
	manager.AddScenario(std::make_unique<Vec3Scenario>());
	manager.AddScenario(std::make_unique<Vec4Scenario>());

	manager.RunAll();

	return 0;
}


