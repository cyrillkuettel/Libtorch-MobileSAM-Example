#pragma once
#ifndef UNIT_TESTS_H____
#define UNIT_TESTS_H____

#include "../mobile_sam.hpp"
#include "catch.hpp"


SamPredictor setupPredictor();

std::vector<unsigned char> readFileToBuffer(const fs::path& filePath);

#endif // UNIT_TESTS_H____
