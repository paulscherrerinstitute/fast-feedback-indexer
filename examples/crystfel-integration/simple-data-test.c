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
    return 0;
}

int main ()
{
    const unsigned N = 300;
    float data[9 + 3*N];
    float* x = &data[9];
    float* y = &data[9 + N];
    float* z = &data[9 + 2*N];
    struct ffbidx_settings settings = {200u, 32u, 32u*1024u, 6u, .02f};
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
    printf("run indexer..\n");
    int res = fast_feedback_crystfel(&settings, data, x, y, z, i-3);
    if (res < 0) {
        printf("indexer failed\n");
        exit(1);
    }
    for (i=0; i<9; i+=3) {
        printf("%f %f %f\n", data[i], data[i+1], data[i+2]);
    }
    printf("is_viable = %d\n", res);
}
