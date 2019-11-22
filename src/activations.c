#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


char* get_activation_string(const ACTIVATION a)
{
    switch (a) {
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case SELU:
            return "selu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}


ACTIVATION get_activation(const char* const s)
{
    if (strcmp(s, "logistic") == 0) {
        return LOGISTIC;
    }
    if (strcmp(s, "swish") == 0) {
        return SWISH;
    }
    if (strcmp(s, "mish") == 0) {
        return MISH;
    }
    if (strcmp(s, "loggy") == 0) {
        return LOGGY;
    }
    if (strcmp(s, "relu") == 0) {
        return RELU;
    }
    if (strcmp(s, "elu") == 0) {
        return ELU;
    }
    if (strcmp(s, "selu") == 0) {
        return SELU;
    }
    if (strcmp(s, "relie") == 0) {
        return RELIE;
    }
    if (strcmp(s, "plse") == 0) {
        return PLSE;
    }
    if (strcmp(s, "hardtan") == 0) {
        return HARDTAN;
    }
    if (strcmp(s, "lhtan") == 0) {
        return LHTAN;
    }
    if (strcmp(s, "linear") == 0) {
        return LINEAR;
    }
    if (strcmp(s, "ramp") == 0) {
        return RAMP;
    }
    if (strcmp(s, "leaky") == 0) {
        return LEAKY;
    }
    if (strcmp(s, "tanh") == 0) {
        return TANH;
    }
    if (strcmp(s, "stair") == 0) {
        return STAIR;
    }
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}


float activate(const float x, const ACTIVATION a)
{
    switch (a) {
        case LINEAR:
            return x;
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return 2.f / (1.f + expf(-x)) - 1;
        case RELU:
            return x * (x > 0);
        case ELU:
            return (x >= 0) * x + (x < 0) * (expf(x) - 1);
        case SELU:
            return (x >= 0) * 1.0507f * x + (x < 0) * 1.0507f * 1.6732f * (expf(x) - 1);
        case RELIE:
            return (x > 0) ? x : .01f * x;
        case RAMP:
            return x * (x > 0) + .1f * x;
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            if (x < -4) {
                return .01f * (x + 4);
            }
            if (x > 4) {
                return .01f * (x - 4) + 1;
            }
            return .125f * x + .5f;
        case STAIR:;
            const int n = floorf(x);
            if (n % 2 == 0) {
                return floorf(x / 2.f);
            }
            return x - n + floorf(x / 2.f);
        case HARDTAN:
            if (x < -1) {
                return -1;
            }
            if (x > 1) {
                return 1;
            }
            return x;
        case LHTAN:
            if (x < 0) {
                return .001f * x;
            }
            if (x > 1) {
                return .001f * (x - 1) + 1;
            }
            return x;
        default:
            return 0;
    }
}


void activate_array(float* const x, const int n, const ACTIVATION a)
{
    switch (a) {
        case LINEAR:
            break;
        case LEAKY:
            #pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                x[i] = leaky_activate(x[i]);
            }
            break;
        case LOGISTIC:
            #pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                x[i] = logistic_activate(x[i]);
            }
            break;
        default:
            for (int i = 0; i < n; ++i) {
                x[i] = activate(x[i], a);
            }
    }
}


void activate_array_swish(const float* const x, const int n, float* const output_sigmoid, float* const output)
{
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        const float x_val = x[i];
        const float sigmoid = logistic_activate(x_val);
        output_sigmoid[i] = sigmoid;
        output[i] = x_val * sigmoid;
    }
}


// https://github.com/digantamisra98/Mish
void activate_array_mish(const float* const x, const int n, float* const activation_input, float* const output)
{
    const float MISH_THRESHOLD = 20;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        const float x_val = x[i];
        activation_input[i] = x_val;  // store value before activation
        // output[i] = x_val * tanh_activate(log(1 + expf(x_val)));
        if (x_val < MISH_THRESHOLD) {
            output[i] = x_val * tanh_activate(log(expf(x_val)));
        } else {
            output[i] = x_val * tanh_activate(x_val);
        }
    }
}


float gradient(float x, const ACTIVATION a)
{
    switch (a) {
        case LINEAR:
            return 1;
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:;
            const float y = (x + 1.f) / 2.f;
            return 2 * (1 - y) * y;
        case RELU:
            return x > 0;
        case ELU:
            return (x >= 0) + (x < 0) * (x + 1);
        case SELU:
            return (x >= 0) * 1.0507f + (x < 0) * (x + 1.0507f * 1.6732f);
        case RELIE:
            return (x > 0) ? 1 : .01f;
        case RAMP:
            return (x > 0) + .1f;
        case LEAKY:
            return (x > 0) ? 1 : .1f;
        case TANH:
            return 1 - x * x;
        case PLSE:
            return (x < 0 || x > 1) ? .01f : .125f;
        case STAIR:
            if (floor(x) == x) {
                return 0;
            }
            return 1;
        case HARDTAN:
            if (x > -1 && x < 1) {
                return 1;
            }
            return 0;
        case LHTAN:
            if (x > 0 && x < 1) {
                return 1;
            }
            return .001f;
        default:
            return 0;
    }
}


void gradient_array(const float* const x, const int n, const ACTIVATION a, float* const delta)
{
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        delta[i] *= gradient(x[i], a);
    }
}


// https://github.com/BVLC/caffe/blob/04ab089db018a292ae48d51732dd6c66766b36b6/src/caffe/layers/swish_layer.cpp#L54-L56
void gradient_array_swish(const float* const x, const int n, const float* const sigmoid, float* const delta)
{
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        const float swish = x[i];
        delta[i] *= swish + sigmoid[i] * (1 - swish);
    }
}


// https://github.com/digantamisra98/Mish
void gradient_array_mish(const int n, const float* const activation_input, float* const delta)
{
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        const float MISH_THRESHOLD = 20.0f;

        // implementation from TensorFlow: https://github.com/tensorflow/addons/commit/093cdfa85d334cbe19a37624c33198f3140109ed
        // implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
        const float inp = activation_input[i];
        const float sp = (inp < MISH_THRESHOLD) ? log1p(exp(inp)) : inp;
        const float grad_sp = 1 - exp(-sp);
        const float tsp = tanh(sp);
        const float grad_tsp = (1 - tsp * tsp) * grad_sp;
        const float grad = inp * grad_tsp + tsp;
        delta[i] *= grad;

        // float x = activation_input[i];
        // float d = 2 * expf(x) + expf(2 * x) + 2;
        // float w = 4 * (x + 1) + 4 * expf(2 * x) + expf(3 * x) + expf(x) * (4 * x + 6);
        // float derivative = expf(x) * w / (d * d);
        // delta[i] *= derivative;
    }
}
