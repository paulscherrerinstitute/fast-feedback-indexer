#ifndef FAST_FEEDBACK_EXCEPTION_H
#define FAST_FEEDBACK_EXCEPTION_H

#include <exception>
#include <sstream>

namespace fast_feedback {

    struct exception : public std::exception {

        std::string error_message;
        const char* file_name;
        unsigned line_number;

        inline exception(const std::string& message, const char* file, unsigned line)
            : error_message(message), file_name(file), line_number(line)
        {}

        inline exception(const exception&) = default;
        inline exception(exception&&) = default;

        inline exception& operator=(const exception&) = default;
        inline exception& operator=(exception&&) = default;

        inline virtual ~exception() noexcept
        {}

        inline const char* what() const noexcept override
        {
            return error_message.c_str();
        }

        template<typename T>
        inline exception& operator<<(const T& data)
        {
            std::ostringstream oss;
            oss << data;
            error_message += oss.str();
            return *this;
        }

    };

}

#define FF_EXCEPTION(msg) fast_feedback::exception(msg, __FILE__, __LINE__)
#define FF_EXCEPTION_OBJ fast_feedback::exception{std::string{}, __FILE__, __LINE__}

#endif
