#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <array>
#include "simple_data.h"
#include "indexer.h"

namespace {

    constexpr struct success_type final {} success;
    constexpr struct failure_type final {} failure;

    template <typename stream>
    [[noreturn]] stream& operator<< (stream& out, [[maybe_unused]] const success_type& data)
    {
        out.flush();
        std::exit((EXIT_SUCCESS));
    }

    template <typename stream>
    [[noreturn]] stream& operator<< (stream& out, [[maybe_unused]] const failure_type& data)
    {
        out.flush();
        std::exit((EXIT_FAILURE));
    }

} // namespace

int main (int argc, char *argv[])
{
    using namespace simple_data;

    try {
        if (argc <= 1)
            throw std::runtime_error("missing file argument");
        SimpleData<float, raise> data(argv[1]);
        std::vector<float> x(data.spots.size() + 3);
        std::vector<float> y(data.spots.size() + 3);
        std::vector<float> z(data.spots.size() + 3);
        unsigned i=0;
        for (const auto& coord : data.unit_cell) {
            x[i] = coord.x;
            y[i] = coord.y;
            z[i] = coord.z;
            i++;
        }
        for (const auto& coord : data.spots) {
            x[i] = coord.x;
            y[i] = coord.y;
            z[i] = coord.z;
            i++;            
        }
        fast_feedback::input<float> in{x.data(), y.data(), z.data(), i-3};
        std::array<float, 3*3> buf;
        fast_feedback::output<float> out{&buf[0], &buf[3], &buf[6], 0u};
        fast_feedback::indexer indexer;
        indexer.index(in, out, fast_feedback::config_runtime<float>{});
    } catch (std::exception& ex) {
        std::cerr << "Test failed: " << ex.what() << '\n' << failure;
    }

    std::cout << "Test OK.\n" << success;
}
