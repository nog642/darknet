#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


char * get_activation_string(ACTIVATION const a)
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
            return "relu";
    }
}


ACTIVATION get_activation(char const * const s)
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
    if (strcmp(s, "normalize_channels") == 0) {
        return NORM_CHAN;
    }
    if (strcmp(s, "normalize_channels_softmax") == 0) {
        return NORM_CHAN_SOFTMAX;
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


float activate(float const x, ACTIVATION const a)
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
            int const n = floorf(x);
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


void activate_array(float * const x, int const n, ACTIVATION const a)
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


void activate_array_swish(float const * const x, int const n, float * const output_sigmoid, float * const output)
{
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float const x_val = x[i];
        float const sigmoid = logistic_activate(x_val);
        output_sigmoid[i] = sigmoid;
        output[i] = x_val * sigmoid;
    }
}


// https://github.com/digantamisra98/Mish
void activate_array_mish(float const * const x, int const n, float * const activation_input, float * const output)
{
    float const MISH_THRESHOLD = 20;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float const x_val = x[i];
        activation_input[i] = x_val;  // store value before activation
        output[i] = x_val * tanh_activate(softplus_activate(x_val, MISH_THRESHOLD));
     }
}


void activate_array_normalize_channels(float const * const x, int const n,
                                       int const batch, int const channels,
                                       int const wh_step, float * const output)
{
    int const size = n / channels;

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        int const wh_i = i % wh_step;
        int const b = i / wh_step;

        float const eps = 0.0001;
        if (i < size) {
            float sum = eps;
            for (int k = 0; k < channels; ++k) {
                float const val = x[wh_i + k * wh_step + b * wh_step * channels];
                if (val > 0) {
                    sum += val;
                }
            }
            for (int k = 0; k < channels; ++k) {
                float val = x[wh_i + k * wh_step + b * wh_step * channels];
                if (val > 0) {
                    val /= sum;
                } else {
                    val = 0;
                }
                output[wh_i + k * wh_step + b * wh_step * channels] = val;
            }
        }
    }
}


void activate_array_normalize_channels_softmax(float const * const x,
                                               int const n, int const batch,
                                               int const channels,
                                               int const wh_step,
                                               float * const output)
{
    int const size = n / channels;

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        int const wh_i = i % wh_step;
        int const b = i / wh_step;

        float const eps = 0.0001;
        if (i < size) {
            float sum = eps;
            for (int k = 0; k < channels; ++k) {
                float const val = x[wh_i + k * wh_step + b * wh_step * channels];
                sum += expf(val);
            }
            for (int k = 0; k < channels; ++k) {
                float val = x[wh_i + k * wh_step + b * wh_step * channels];
                val = expf(val) / sum;
                output[wh_i + k * wh_step + b * wh_step * channels] = val;
            }
        }
    }
}


void gradient_array_normalize_channels_softmax(float const * const x,
                                               int const n, int const batch,
                                               int const channels,
                                               int const wh_step,
                                               float * const delta)
{
    int const size = n / channels;

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        int const wh_i = i % wh_step;
        int const b = i / wh_step;

        if (i < size) {
            float grad = 0;
            for (int k = 0; k < channels; ++k) {
                int const index = wh_i + k * wh_step + b * wh_step * channels;
                float const out = x[index];
                float const d = delta[index];
                grad += out * d;
            }
            for (int k = 0; k < channels; ++k) {
                int const index = wh_i + k * wh_step + b * wh_step * channels;
                float d = delta[index];
                d *= grad;
                delta[index] = d;
            }
        }
    }
}

void gradient_array_normalize_channels(float *x, const int n, int batch, int channels, int wh_step, float *delta)
{
    int size = n / channels;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        int wh_i = i % wh_step;
        int b = i / wh_step;

        if (i < size) {
            float grad = 0;
            int k;
            for (k = 0; k < channels; ++k) {
                const int index = wh_i + k * wh_step + b*wh_step*channels;
                float out = x[index];
                float d = delta[index];
                grad += out*d;
            }
            for (k = 0; k < channels; ++k) {
                const int index = wh_i + k * wh_step + b*wh_step*channels;
                if (x[index] > 0) {
                    float d = delta[index];
                    d = d * grad;
                    delta[index] = d;
                }
            }
        }
    }
}


float gradient(float const x, ACTIVATION const a)
{
    switch (a) {
        case LINEAR:
            return 1;
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:;
            float const y = (x + 1.f) / 2.f;
            return 2 * (1 - y) * y;
        case RELU:
            return relu_gradient(x);
        case NORM_CHAN:
            //return relu_gradient(x);
        case NORM_CHAN_SOFTMAX:
            printf(" Error: should be used custom NORM_CHAN or NORM_CHAN_SOFTMAX-function for gradient \n");
            exit(0);
            return 0;
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


void gradient_array(float const * const x, int const n, ACTIVATION const a, float * const delta)
{
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        delta[i] *= gradient(x[i], a);
    }
}


// https://github.com/BVLC/caffe/blob/04ab089db018a292ae48d51732dd6c66766b36b6/src/caffe/layers/swish_layer.cpp#L54-L56
void gradient_array_swish(float const * const x, int const n, float const * const sigmoid, float * const delta)
{
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float const swish = x[i];
        delta[i] *= swish + sigmoid[i] * (1 - swish);
    }
}


// https://github.com/digantamisra98/Mish
void gradient_array_mish(int const n, float const * const activation_input, float * const delta)
{
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float const MISH_THRESHOLD = 20.0f;

        // implementation from TensorFlow: https://github.com/tensorflow/addons/commit/093cdfa85d334cbe19a37624c33198f3140109ed
        // implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
        float const inp = activation_input[i];
        float const sp = softplus_activate(inp, MISH_THRESHOLD);
        float const grad_sp = 1 - exp(-sp);
        float const tsp = tanh(sp);
        float const grad_tsp = (1 - tsp * tsp) * grad_sp;
        float const grad = inp * grad_tsp + tsp;
        delta[i] *= grad;

        // float x = activation_input[i];
        // float d = 2 * expf(x) + expf(2 * x) + 2;
        // float w = 4 * (x + 1) + 4 * expf(2 * x) + expf(3 * x) + expf(x) * (4 * x + 6);
        // float derivative = expf(x) * w / (d * d);
        // delta[i] *= derivative;
    }
}
