#include <iostream>
#include <string>
#include "exception.h"

namespace {
    fast_feedback::exception create_copy()
    {
        return fast_feedback::exception(FF_EXCEPTION("12"));
    }

    fast_feedback::exception move_assign_copy()
    {
        fast_feedback::exception ex1{FF_EXCEPTION_OBJ};
        fast_feedback::exception ex2 = FF_EXCEPTION_OBJ;
        ex1 = std::move(create_copy());
        ex2 = ex1;
        return ex2;
    }
}

int main(int, char *argv[])
{
    fast_feedback::exception ex = move_assign_copy() << "34";

    std::string msg{ex.what()};
    std::string fname{ex.file_name};
    std::string failed{"Test failed: "};

    if (ex.line_number == 8) {
        if (fname.find("test_indexer_exception.cpp") != std::string::npos) {
            if (msg == "1234") {
                std::cout << "Test OK.\n";
                return ((EXIT_SUCCESS));
            } else {
                std::cerr << failed << "msg=" << msg << '\n';
            }
        } else {
            std::cerr << failed << "fname=" << fname << '\n';
        }
    } else {
        std::cerr << failed << "line=" << ex.line_number << '\n';
    }

    return ((EXIT_FAILURE));
}
