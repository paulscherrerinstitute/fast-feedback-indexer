// Read in simple data files

#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <Eigen/Core>
#include <Eigen/LU>

namespace {

    // vector coordinates
    template <typename float_type>
    struct Coord final {
        using coordinate_type = float_type;
        float_type x;
        float_type y;
        float_type z;
    };

    // remove white space at left/start
    void ltrim(std::string &s) {
        s.erase(begin(s), std::find_if(cbegin(s), cend(s), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
    }

    // remove white space at right/end
    void rtrim(std::string &s) {
        s.erase(std::find_if(crbegin(s), crend(s), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), end(s));
    }

    // remove leading and trailing white space
    void trim(std::string &s) {
        ltrim(s);
        rtrim(s);
    }

    // input stream for next line in file
    std::istringstream next_line(std::ifstream& fin)
    {
        for (std::string line; std::getline(fin, line);) {
            if (line[0] == '#')
                continue;
            trim(line);
            if (line.empty())
                continue;
            return std::istringstream{line};
        }
        return std::istringstream{};
    }

    // get vector coordinates from file
    template <typename float_type, typename error_function>
    Coord<float_type> read_coordinates(std::istringstream& iss)
    {
        Coord<float_type> c;
        if (! (iss >> c.x >> c.y >> c.z))
            error_function("can't read vec");
        return c;
    }

} // namespace

namespace simple_data {
    // error handler: print error message (nullptr for none) and exit
    [[noreturn]] static void stop(const char* msg)
    {
        if (msg != nullptr)
            std::cerr << "Error: " << msg << '\n';
        std::exit((EXIT_FAILURE));
    }

    // error handler: raise exception
    void static raise(const char* msg)
    {
        throw std::invalid_argument(msg ? msg : "unspecified error");
    }

    // represents a simple data file
    template <typename float_type, typename error_function=stop>
    struct SimpleData final {
        std::array<Coord<float_type>, 3> unit_cell;
        std::vector<Coord<float_type>> spots;

        // read input data from file
        explicit SimpleData(const std::string& input_file_name)
        {
            std::ifstream input_file(input_file_name);
            if (! input_file.is_open())
                error_function("unable to read file");

            {
                std::istringstream line = next_line(input_file);
                unit_cell[0] = read_coordinates<float_type, error_function>(line);
                unit_cell[1] = read_coordinates<float_type, error_function>(line);
                unit_cell[2] = read_coordinates<float_type, error_function>(line);
                Eigen::Matrix<float_type, 3, 3> A;
                A << unit_cell[0].x, unit_cell[1].x, unit_cell[2].x,
                     unit_cell[0].y, unit_cell[1].y, unit_cell[2].y,
                     unit_cell[0].z, unit_cell[1].z, unit_cell[2].z;
                A = A.inverse().transpose();    // transform into reciprocal space
                unit_cell[0].x = A[0,0]; unit_cell[1].x = A[0,1]; unit_cell[2].x = A[0,2];
                unit_cell[0].y = A[1,0]; unit_cell[1].y = A[1,1]; unit_cell[2].y = A[1,2];
                unit_cell[0].z = A[2,0]; unit_cell[1].z = A[2,1]; unit_cell[2].z = A[2,2];
            }

            do {
                std::istringstream line = next_line(input_file);
                if (!line || line.str().empty())
                    break;
                spots.emplace_back(read_coordinates<float_type, error_function>(line));
            } while (true);
        }

        SimpleData() = default;
        ~SimpleData() = default;
        SimpleData(const SimpleData&) = default;
        SimpleData(SimpleData&&) = default;
        SimpleData& operator=(const SimpleData&) = default;
        SimpleData& operator=(SimpleData&&) = default;
    }; // SimpleData

} // namespace simple_data
