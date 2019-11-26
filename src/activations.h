#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "darknet.h"
#include "dark_cuda.h"
#include "math.h"

//typedef enum{
//    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU, SWISH, MISH
//}ACTIVATION;


#ifdef __cplusplus
extern "C" {
#endif


ACTIVATION get_activation(char const * s);


char * get_activation_string(ACTIVATION a);

float activate(float x, ACTIVATION a);

float gradient(float x, ACTIVATION a);

void gradient_array(float const * x, int n, ACTIVATION a, float * delta);

void gradient_array_swish(float const * x, int n, float const * sigmoid, float * delta);

void gradient_array_mish(int n, float const * activation_input, float * delta);

void activate_array(float * x, int n, ACTIVATION a);

void activate_array_swish(float const * x, int n, float * output_sigmoid, float * output);

void activate_array_mish(float const * x, int n, float * activation_input, float * output);

#ifdef GPU

void activate_array_ongpu(float * x, int n, ACTIVATION a);

void activate_array_swish_ongpu(float * x, int n, float * output_sigmoid_gpu, float * output_gpu);

void activate_array_mish_ongpu(float * x, int n, float * activation_input_gpu, float * output_gpu);

void gradient_array_ongpu(float * x, int n, ACTIVATION a, float * delta);

void gradient_array_swish_ongpu(float * x, int n, float * sigmoid_gpu, float * delta);

void gradient_array_mish_ongpu(int n, float * activation_input_gpu, float * delta);

#endif  // GPU


static inline float logistic_activate(float const x)
{
    return 1.f / (1.f + expf(-x));
}


static inline float leaky_activate(float const x)
{
    return (x > 0) ? x : .1f * x;
}


static inline float tanh_activate(float const x)
{
    return (expf(2 * x) - 1) / (expf(2 * x) + 1);
}


static inline float logistic_gradient(float const x)
{
    return (1 - x) * x;
}


#ifdef __cplusplus
}
#endif

#endif
