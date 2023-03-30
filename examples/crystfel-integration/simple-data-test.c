#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include "ffbidx/c-wrapper.h"

static int skipline()
{
    int c;
    do {
        if ((c = getc(stdin)) == EOF)
            return EOF;
    } while (c != '\n');
    return 0;
}

static int skipcomment()
{
    int c;
    if ((c = getc(stdin)) == EOF)
        return EOF;
    if (c == '#')
        return skipline();
    return ungetc(c, stdin);
}

static int skipws()
{
    int c;
    do {
        c = getc(stdin);
        if (c == EOF)
            return EOF;
    } while (isspace(c));
    return ungetc(c, stdin);
}

static int getvec(float* x, float* y, float* z)
{
    int c;
    if ((c = scanf("%f", x)) <= 0)
        return EOF;
    if (skipws() == EOF)
        return EOF;
    if ((c = scanf("%f", y)) <= 0)
        return EOF;
    if (skipws() == EOF)
        return EOF;
    if ((c = scanf("%f", z)) <= 0)
        return EOF;
}

int main (int argc, char* argv[])
{
    const unsigned N = 300;
    float data[9 + 3*N];
    float* x = &data[9];
    float* y = &data[9 + N];
    float* z = &data[9 + 2*N];
    struct ffbidx_indexer idx;
    struct ffbidx_settings settings = {200};
    unsigned i=0;
    int c;
    do {
        if (skipws() == EOF)
            break;
        if (skipcomment() == EOF)
            break;
        if (i < 3)
            c = getvec(&data[i], &data[3 + i], &data[6 + i]);
        else
            c = getvec(&x[i - 3], &y[i - 3], &z[i - 3]);
        if (c == EOF)
            break;
        i++;
    } while (1);
    if (getc(stdin) != EOF) {
        printf("input error\n");
        exit(1);
    }
    if (i < 4) {
        printf("format error\n");
        exit(1);
    }
    printf("allocating indexer..\n");
    if (allocate_fast_indexer(&idx, &settings) != 0) {
        printf("indexer allocation failed\n");
        exit(1);
    }
    printf("calling indexer..\n");
    int res;
    if ((res = index_refined(idx, data, x, y, z, i-3)) < 0) {
        printf("indexing error\n");
        exit(1);
    }
    printf("free indexer..\n");
    free_fast_indexer(idx);
    for (i=0; i<9; i+=3) {
        printf("%f %f %f\n", data[i], data[i+1], data[i+2]);
    }
    printf("is_viable = %d\n", res);
}
