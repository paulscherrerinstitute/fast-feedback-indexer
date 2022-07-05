#include "indexer.h"
#include "indexer_gpu.h"

namespace fast_feedback {

    template <typename float_type>
    void indexer<float_type>::index(const input<float_type>& in, output<float_type>& out, const config_runtime<float_type>& conf_rt)
    {
        gpu::index(*this, in, out, conf_rt);
    }

    template void indexer<float>::index(const input<float>&, output<float>&, const config_runtime<float>&);

} // namespace fast_feedback
