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
            std::cout << "input" << i << ": " << x[i] << ", " << y[i] << ", " << z[i] << '\n';
            i++;
        }
        for (const auto& coord : data.spots) {
            x[i] = coord.x;
            y[i] = coord.y;
            z[i] = coord.z;
            i++;            
        }

        fast_feedback::memory_pin pin_x{x};
        fast_feedback::memory_pin pin_y{y};
        fast_feedback::memory_pin pin_z{z};
        fast_feedback::input<float> in{x.data(), y.data(), z.data(), 1u, i-3u};
        fast_feedback::memory_pin pin_in{fast_feedback::memory_pin::on(in)};

        std::array<float, 3*3> buf;
        fast_feedback::memory_pin pin_buf{buf};
        fast_feedback::output<float> out{&buf[0], &buf[3], &buf[6], 0u};
        fast_feedback::memory_pin pin_out{fast_feedback::memory_pin::on(out)};

        fast_feedback::indexer indexer;
        indexer.index(in, out, fast_feedback::config_runtime<float>{});

        auto success = true;
        for (unsigned i=0; i<3; ++i) {
            if (out.x[i] != in.x[i])
                success = false;
            if (out.y[i] != in.y[i])
                success = false;
            if (out.z[i] != in.z[i])
                success = false;
            std::cout << "output" << i << ": " << out.x[i] << ", " << out.y[i] << ", " << out.z[i] << '\n';
        }

        std::cout << "Test " << ( success ? "OK" : "failed" ) << ".\n";

    } catch (std::exception& ex) {
        std::cerr << "Test failed: " << ex.what() << '\n' << failure;
    }
}
