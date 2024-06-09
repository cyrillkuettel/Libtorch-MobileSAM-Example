#include "UnitTest.h"


TEST_CASE("", "some-test")
{
	SECTION("Test something")
	{
                SamPredictor predictor = setupPredictor();


                const AppConfig config = {
                    {{228.0f, 102.0f}, {325.0f, 261.0f}},
                    {2.0f, 3.0f},  // top left, bottom right
                    "images/elephants.jpg",
                };


                fs::path sourceDir = fs::path(__FILE__).parent_path();
                fs::path defaultImagePath = sourceDir / config.defaultImagePath;

                REQUIRE(fs::exists(defaultImagePath));
                validateAppConfig(config);

	}

}

