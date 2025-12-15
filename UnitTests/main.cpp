#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

int main()
{
	doctest::Context ctx;

	ctx.setOption("abort-after", 0);
	ctx.setOption("break-on-failure", false);
	//ctx.setOption("order-by", "name");
	ctx.setOption("verbosity", 2);

	//run test
	int res = ctx.run();


	if (ctx.shouldExit())
		return res;

	return res;
}