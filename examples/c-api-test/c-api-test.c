#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <error.h>
#include "ffbidx/c_api.h"

static int read_float(FILE* f, float* x)
{
    errno = 0;
    int res = fscanf(f, "%f", x);
    if (res == EOF)
        return EOF;
    if (res == 0)
        error(-1, 0, "error: unable to parse float");
    if (res < 0)
        error(-1, errno, "can't read float");
    return res;
}

static void read_vector(FILE* f, float* x, float* y, float* z)
{
    if ((read_float(f, x) <= 0) ||
        (read_float(f, y) <= 0) ||
        (read_float(f, z) <= 0))
    {
        error(-1, 0, "error: unable to read vector");
    }
}

static int read_spot(FILE* f, float* x, float* y, float* z)
{
    int res = read_float(f, x);
    if (res == EOF)
        return EOF;
    if ((res <= 0) ||
        (read_float(f, y) <= 0) ||
        (read_float(f, z) <= 0))
    {
        error(-1, 0, "error: unable to read spot");
    }
    return 0;
}

static void read_input(FILE* f, struct input* in)
{
    for (unsigned i=0; i<3; i++)
        read_vector(f, &in->cell.x[i], &in->cell.y[i], &in->cell.z[i]);
    int n = 0;
    while (
        (n < in->n_spots) &&
        (read_spot(f, &in->spot.x[n], &in->spot.y[n], &in->spot.z[n]) != EOF)
    ) {
        n++;
    }
    in->n_spots = n;
    if (n < 6)
        error(-1, 0, "error: less than 6 spots");
}

int main(int argc, char *argv[])
{
    if (argc != 2)
        error(-1, 0, "usage: %s <filename>\n", argv[0]);

    struct config_persistent cpers;
    struct config_runtime crt;
    struct config_ifssr cifssr;
    set_defaults(&cpers, &crt, &cifssr);
    cpers.max_input_cells = 1;
    const unsigned max_spots = cpers.max_spots;

    struct input in;
    struct output out;
    float spots[3*max_spots];
    float icell[9];
    in.spot.x = &spots[0];
    in.spot.y = &spots[max_spots];
    in.spot.z = &spots[2*max_spots];
    in.n_spots = max_spots;
    in.new_spots = true;
    in.cell.x = &icell[0];
    in.cell.y = &icell[3];
    in.cell.z = &icell[6];
    in.n_cells = 1;
    in.new_cells = true;

    float ocell[9];
    float score;
    out.x = &ocell[0];
    out.y = &ocell[3];
    out.z = &ocell[6];
    out.n_cells = 1;
    out.score = &score;

    const unsigned msg_len = 256;
    char message[msg_len];
    struct error err = {message, msg_len};
    message[0] = 0;

    unsigned cryst[2];

    FILE* f = fopen(argv[1], "r");
    if (! f)
        error(-1, errno, "can't read input file");

    read_input(f, &in);
    fclose(f);

    if (check_config(&cpers, &crt, &cifssr, &err))
        error(-1, 0, "config check failed: %s", message);

    int h = create_indexer(&cpers, &err, NULL);
    if (h < 0)
        error(-1, 0, "can't create indexer: %s", message);

    if (indexer_op(h, &in, &out, &crt, &cifssr))
        error(-1, 0, "can't index: %s", message);

    int res = crystals(h, &in, &out, cifssr.max_distance, cifssr.min_spots, cryst, 2);
    if (res < 0)
        error(-1, 0, "can't find crystals: %s", message);

    if (drop_indexer(h) < 0)
        error(-1, 0, "can't drop indexer");

    for (unsigned i=0; i<3; i++) {
        for (unsigned j=0; j<3; j++)
            printf("%f ", ocell[3*j+i]);
        printf("\n");
    }
    printf("score: %f\n", score);
    printf("crystals:");
    for (int i=0; i<res; i++)
        printf(" %d", cryst[i]);
    printf("\n");
}
