#include "avgpool_layer.h"
#include "dark_cuda.h"
#include <stdio.h>


avgpool_layer make_avgpool_layer(int const batch, int const w, int const h, int const c)
{
    fprintf(stderr, "avg                          %4d x%4d x%4d ->   %4d\n",  w, h, c, c);
    int const output_size = c * batch;
    float * const output = calloc(output_size, sizeof(float));
    float * const delta = calloc(output_size, sizeof(float));
    return (avgpool_layer){
        .type=AVGPOOL,
        .batch=batch,
        .h=h,
        .w=w,
        .c=c,
        .out_w=1,
        .out_h=1,
        .out_c=c,
        .outputs=c,
        .inputs=h * w * c,
        .output=output,
        .delta=delta,
        .forward=forward_avgpool_layer,
        .backward=backward_avgpool_layer,
#ifdef GPU
        .forward_gpu=forward_avgpool_layer_gpu,
        .backward_gpu=backward_avgpool_layer_gpu,
        .output_gpu=cuda_make_array(output, output_size),
        .delta_gpu=cuda_make_array(delta, output_size)
#endif
    };
}


void resize_avgpool_layer(avgpool_layer * const l, int const w, int const h)
{
    l->w = w;
    l->h = h;
    l->inputs = h * w * l->c;
}


void forward_avgpool_layer(avgpool_layer const l, network_state state)
{
    for (int b = 0; b < l.batch; ++b) {
        for (int k = 0; k < l.c; ++k) {
            int const out_index = k + b * l.c;
            l.output[out_index] = 0;
            for (int i = 0; i < l.h * l.w; ++i) {
                int const in_index = i + l.h * l.w * (k + b * l.c);
                l.output[out_index] += state.input[in_index];
            }
            l.output[out_index] /= l.h * l.w;
        }
    }
}


void backward_avgpool_layer(avgpool_layer const l, network_state state)
{
    for (int b = 0; b < l.batch; ++b) {
        for (int k = 0; k < l.c; ++k) {
            int const out_index = k + b * l.c;
            for (int i = 0; i < l.h * l.w; ++i){
                int const in_index = i + l.h * l.w * (k + b * l.c);
                state.delta[in_index] += l.delta[out_index] / (l.h * l.w);
            }
        }
    }
}
