#include "indexer.h"
#include "indexer_gpu.h"

namespace fast_feedback {

    std::atomic<state_id::type> state_id::next = state_id::null + 1u;

    template <typename float_type>
    void indexer<float_type>::init (indexer<float_type>& instance, const config_persistent<float_type>& conf)
    {
        instance.cpers = conf;
        gpu::init(instance);
    }

    template <typename float_type>
    void indexer<float_type>::drop (indexer<float_type>& instance)
    {
        gpu::drop(instance);
    }

    template <typename float_type>
    void indexer<float_type>::index(const input<float_type>& in, output<float_type>& out, const config_runtime<float_type>& conf_rt)
    {
        gpu::index(*this, in, out, conf_rt);
    }

    template void indexer<float>::init(indexer<float>&, const config_persistent<float>&);
    template void indexer<float>::drop(indexer<float>&);
    template void indexer<float>::index(const input<float>&, output<float>&, const config_runtime<float>&);

} // namespace fast_feedback
