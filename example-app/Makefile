# builds the main library
lib:
	cd build && cmake --build . --config Debug  -j 8

# actually run an image through the model
# shows how to use the library which was built in the step above
example_main: lib
	cd build && cmake --build . --config Debug  -j 8 --target example_main

# builds only the tests
tests: lib
	cd build && cmake --build . --config Debug  -j 8 --target UnitTests

release:
	cd build && cmake --build . --config Release

format:
	/usr/bin/clang-format -i -style=file *.cpp