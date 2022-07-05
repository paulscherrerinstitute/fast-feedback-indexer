#include <iostream>
#include "indexer_gpu.h"

namespace gpu {

    template <typename float_type>
    void index (const indexer<float_type>& instance, const input<float_type>& in, output<float_type>& out, const config_runtime<float_type>& conf_rt)
    {
        std::cout << "hello: " << sizeof(float_type) << '\n';
    }

    template void index<float> (const indexer<float>&, const input<float>&, output<float>&, const config_runtime<float>&);

} // namespace gpu
