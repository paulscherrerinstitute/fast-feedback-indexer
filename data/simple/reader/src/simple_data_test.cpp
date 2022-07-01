#include <iostream>
#include "simple_data.h"

int main (int argc, char *argv[])
{
    using namespace simple_data;
    simple_data::stop error;

    if (argc != 2)
        error("expected exactly one argument - the name of the simple data file");

    SimpleData<float> data(argv[1]);

    if (data.spots.size() != 297)
        error("test failed!");
    else
        std::cout << "Test OK.\n";
}
