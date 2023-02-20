// vulkanTemplate.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
//#define NDEBUG
#include "headers/VulkanTemplateApp.h"
#include <iostream>

int main()
{
    VulkanTemplateApp myApp;

    try {
        myApp.run();
    }
    catch (std::exception ex) {
        std::cout << ex.what() << std::endl;
    }

    //system("pause");
    std::cout << "\n*************************************\n\tEnd of program\n*************************************\n";
    return EXIT_SUCCESS;
}