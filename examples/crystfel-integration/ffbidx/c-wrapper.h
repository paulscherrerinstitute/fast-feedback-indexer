#ifdef __cplusplus
extern "C" {
#endif

struct ffbidx_indexer {
    void* ptr;
};

struct ffbidx_settings {
    // for the time being this would be the only parameter with non-default settings
    // let's have all the other parameters hard coded in the C-wrapper and if necessary, later moved here
    unsigned max_spots; // maximum number of input spots
};

int allocate_fast_indexer(struct ffbidx_indexer* idx, struct ffbidx_settings *settings);
void free_fast_indexer(struct ffbidx_indexer idx);
int index_raw(struct ffbidx_indexer idx, float cell[9], float *x, float *y, float *z, unsigned nspots);
int index_refined(struct ffbidx_indexer idx, float cell[9], float *x, float *y, float *z, unsigned nspots);

int fast_feedback_crystfel(struct ffbidx_settings *settings, float cell[9], float *x, float *y, float *z, unsigned nspots);

#ifdef __cplusplus
} // extern "C"
#endif
