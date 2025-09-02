#include <gtest/gtest.h>
#include <iostream>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "=== Unified Test Suite ===" << std::endl;
    std::cout << "Running bottom-up component tests..." << std::endl;
    
    return RUN_ALL_TESTS();
}