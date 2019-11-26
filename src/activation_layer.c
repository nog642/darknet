#include "activation_layer.h"
#include "utils.h"
#include "dark_cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


layer make_activation_layer(int const batch, int const inputs, ACTIVATION const activation)
{
    layer const l = (layer){
        .type=ACTIVE,

        .batch=batch,
        .inputs=inputs,
        .outputs=inputs,

        .delta=calloc(batch * inputs, sizeof(float)),
        .output=calloc(batch * inputs, sizeof(float)),

        .forward=forward_activation_layer,
        .backward=backward_activation_layer,
#ifdef GPU
        .forward_gpu=forward_activation_layer_gpu,
        .backward_gpu=backward_activation_layer_gpu,

        .output_gpu=cuda_make_array(l.output, batch * inputs),
        .delta_gpu=cuda_make_array(l.delta, batch * inputs),
#endif
        .activation=activation
    };
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}


void forward_activation_layer(layer l, network_state state)
{
    copy_cpu(l.outputs * l.batch, state.input, 1, l.output, 1);
    activate_array(l.output, l.outputs * l.batch, l.activation);
}


void backward_activation_layer(layer l, network_state state)
{
    gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);
    copy_cpu(l.outputs * l.batch, l.delta, 1, state.delta, 1);
}


#ifdef GPU

void forward_activation_layer_gpu(layer l, network_state state)
{
    copy_ongpu(l.outputs * l.batch, state.input, 1, l.output_gpu, 1);
    activate_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation);
}


void backward_activation_layer_gpu(layer l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation, l.delta_gpu);
    copy_ongpu(l.outputs * l.batch, l.delta_gpu, 1, state.delta, 1);
}

#endif
