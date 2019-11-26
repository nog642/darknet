#include "blas.h"

#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void reorg_cpu(float * x, int out_w, int out_h, int out_c, int batch, int stride, int forward, float * out)
{
    int in_c = out_c / (stride * stride);

    // printf("\n out_c = %d, out_w = %d, out_h = %d, stride = %d, forward = %d \n", out_c, out_w, out_h, stride, forward);
    // printf("  in_c = %d,  in_w = %d,  in_h = %d \n", in_c, out_w * stride, out_h * stride);

    for (int b = 0; b < batch; ++b) {
        for (int k = 0; k < out_c; ++k) {
            for (int j = 0; j < out_h; ++j) {
                for (int i = 0; i < out_w; ++i) {
                    int in_index = i + out_w * (j + out_h * (k + out_c * b));
                    int c2 = k % in_c;
                    int offset = k / in_c;
                    int w2 = i * stride + offset % stride;
                    int h2 = j * stride + offset / stride;
                    int out_index = w2 + out_w * stride * (h2 + out_h * stride * (c2 + in_c * b));
                    if (forward) {
                        out[out_index] = x[in_index];  // used by default for forward (i.e. forward = 0)
                    } else {
                        out[in_index] = x[out_index];
                    }
                }
            }
        }
    }
}


void flatten(float * const x, int const size, int const layers, int const batch, int const forward)
{
    float * const swap = calloc(size * layers * batch, sizeof(float));
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < layers; ++c) {
            for (int i = 0; i < size; ++i) {
                int const i1 = b * layers * size + c * size + i;
                int const i2 = b * layers * size + i * layers + c;
                if (forward) {
                    swap[i2] = x[i1];
                } else {
                    swap[i1] = x[i2];
                }
            }
        }
    }
    memcpy(x, swap, size * layers * batch * sizeof(float));
    free(swap);
}


void weighted_sum_cpu(float const * const a, float const * const b, float const * const s, int const n, float * const c)
{
    for (int i = 0; i < n; ++i) {
        c[i] = s[i] * a[i] + (1 - s[i]) * (b ? b[i] : 0);
    }
}


void weighted_delta_cpu(float const * const a, float const * const b, float const * const s, float * const da,
                        float * const db, float * const ds, int const n, float const * const dc)
{
    for (int i = 0; i < n; ++i) {
        if (da) {
            da[i] += dc[i] * s[i];
        }
        if (db) {
            db[i] += dc[i] * (1 - s[i]);
        }
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}


void shortcut_cpu(int const batch, int const w1, int const h1, int const c1, float const * const add, int const w2,
                  int const h2, int const c2, float * const out)
{
    int stride = w1 / w2;
    int sample = w2 / w1;
    assert(stride == h1 / h2);
    assert(sample == h2 / h1);
    if (stride < 1) {
        stride = 1;
    }
    if (sample < 1) {
        sample = 1;
    }
    int const minw = (w1 < w2) ? w1 : w2;
    int const minh = (h1 < h2) ? h1 : h2;
    int const minc = (c1 < c2) ? c1 : c2;

    for (int b = 0; b < batch; ++b) {
        for (int k = 0; k < minc; ++k) {
            for (int j = 0; j < minh; ++j) {
                for (int i = 0; i < minw; ++i) {
                    int const out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
                    int const add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));
                    out[out_index] += add[add_index];
                }
            }
        }
    }
}


void mean_cpu(float const * const x, int const batch, int const filters, int const spatial, float * const mean)
{
    float const scale = 1. / (batch * spatial);
    for (int i = 0; i < filters; ++i){
        mean[i] = 0;
        for (int j = 0; j < batch; ++j){
            for (int k = 0; k < spatial; ++k){
                int const index = j * filters * spatial + i * spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}


void variance_cpu(float const * const x, float const * const mean, int const batch, int const filters,
                  int const spatial, float * const variance)
{
    float const scale = 1. / (batch * spatial - 1);
    for (int i = 0; i < filters; ++i) {
        variance[i] = 0;
        for (int j = 0; j < batch; ++j) {
            for (int k = 0; k < spatial; ++k) {
                int const index = j * filters * spatial + i * spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}


void normalize_cpu(float * const x, float const * const mean, float const * const variance, int const batch,
                   int const filters, int const spatial)
{
    for (int b = 0; b < batch; ++b) {
        for (int f = 0; f < filters; ++f) {
            for (int i = 0; i < spatial; ++i) {
                int const index = b * filters * spatial + f * spatial + i;
                x[index] = (x[index] - mean[f]) / (sqrt(variance[f]) + .000001f);
            }
        }
    }
}


void const_cpu(int const N, float const ALPHA, float * const X, int const INCX)
{
    for (int i = 0; i < N; ++i) {
        X[i * INCX] = ALPHA;
    }
}


void mul_cpu(int const N, float const * const X, int const INCX, float * const Y, int const INCY)
{
    for (int i = 0; i < N; ++i) {
        Y[i * INCY] *= X[i * INCX];
    }
}


void pow_cpu(int const N, float const ALPHA, float const * const X, int const INCX, float * const Y, int const INCY)
{
    for (int i = 0; i < N; ++i) {
        Y[i * INCY] = pow(X[i * INCX], ALPHA);
    }
}


void axpy_cpu(int const N, float const ALPHA, float const * const X, int const INCX, float * const Y, int const INCY)
{
    for (int i = 0; i < N; ++i) {
        Y[i * INCY] += ALPHA * X[i * INCX];
    }
}


void scal_cpu(int const N, float const ALPHA, float * const X, int const INCX)
{
    for (int i = 0; i < N; ++i) {
        X[i * INCX] *= ALPHA;
    }
}


void scal_add_cpu(int const N, float const ALPHA, float const BETA, float * const X, int const INCX)
{
    for (int i = 0; i < N; ++i) {
        X[i * INCX] = X[i * INCX] * ALPHA + BETA;
    }
}


void fill_cpu(int const N, float const ALPHA, float * const X, int const INCX)
{
    if (INCX == 1 && ALPHA == 0) {
        memset(X, 0, N * sizeof(float));
    } else {
        for (int i = 0; i < N; ++i) {
            X[i * INCX] = ALPHA;
        }
    }
}


void deinter_cpu(int const NX, float * const X, int const NY, float * const Y, int const B, float const * const OUT)
{
    int index = 0;
    for (int j = 0; j < B; ++j) {
        for (int i = 0; i < NX; ++i) {
            if (X != NULL) {
                X[j * NX + i] += OUT[index];
            }
            ++index;
        }
        for (int i = 0; i < NY; ++i) {
            if (Y != NULL) {
                Y[j * NY + i] += OUT[index];
            }
            ++index;
        }
    }
}


void inter_cpu(int const NX, float const * const X, int const NY, float const * const Y, int const B, float * const OUT)
{
    int index = 0;
    for (int j = 0; j < B; ++j) {
        for (int i = 0; i < NX; ++i) {
            OUT[index++] = X[j * NX + i];
        }
        for (int i = 0; i < NY; ++i) {
            OUT[index++] = Y[j * NY + i];
        }
    }
}


void copy_cpu(int const N, float const * const X, int const INCX, float * const Y, int const INCY)
{
    for (int i = 0; i < N; ++i) {
        Y[i * INCY] = X[i * INCX];
    }
}


void mult_add_into_cpu(int const N, float const * const X, float const * const Y, float * const Z)
{
    for (int i = 0; i < N; ++i) {
        Z[i] += X[i] * Y[i];
    }
}


void smooth_l1_cpu(int const n, float const * const pred, float const * const truth, float * const delta,
                   float * const error)
{
    for (int i = 0; i < n; ++i) {
        float const diff = truth[i] - pred[i];
        float const abs_val = fabs(diff);
        if (abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        } else {
            error[i] = 2 * abs_val - 1;
            delta[i] = (diff > 0) ? 1 : -1;
        }
    }
}


void l1_cpu(int const n, float const * const pred, float const * const truth, float * const delta, float * const error)
{
    for (int i = 0; i < n; ++i) {
        float const diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}


void softmax_x_ent_cpu(int const n, float const * const pred, float const * const truth, float * const delta,
                       float * const error)
{
    for (int i = 0; i < n; ++i) {
        float const t = truth[i];
        float const p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t - p;
    }
}


void logistic_x_ent_cpu(int const n, float const * const pred, float const * const truth, float * const delta,
                        float * const error)
{
    for (int i = 0; i < n; ++i) {
        float const t = truth[i];
        float const p = pred[i];
        error[i] = -t * log(p) - (1 - t) * log(1-p);
        delta[i] = t - p;
    }
}


void l2_cpu(int const n, float const * const pred, float const * const truth, float * const delta, float * const error)
{
    for (int i = 0; i < n; ++i) {
        float const diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}


float dot_cpu(int const N, float const * const X, int const INCX, float const * const Y, int const INCY)
{
    float dot = 0;
    for (int i = 0; i < N; ++i) {
        dot += X[i * INCX] * Y[i * INCY];
    }
    return dot;
}


void softmax(float const * const input, int const n, float const temp, float * const output, int const stride)
{
    float largest = -FLT_MAX;
    for (int i = 0; i < n; ++i) {
        if (input[i * stride] > largest) {
            largest = input[i * stride];
        }
    }

    float sum = 0;
    for (int i = 0; i < n; ++i) {
        float const e = exp(input[i * stride]/temp - largest/temp);
        sum += e;
        output[i * stride] = e;
    }

    for (int i = 0; i < n; ++i) {
        output[i * stride] /= sum;
    }
}


void softmax_cpu(float const * const input, int const n, int const batch, int const batch_offset, int const groups,
                 int const group_offset, int const stride, float const temp, float * const output)
{
    for (int b = 0; b < batch; ++b) {
        for (int g = 0; g < groups; ++g) {
            softmax(input + b * batch_offset + g * group_offset, n, temp, output + b * batch_offset + g * group_offset, stride);
        }
    }
}


void upsample_cpu(float * const in, int const w, int const h, int const c, int const batch, int const stride,
                  int const forward, float const scale, float * const out)
{
    for (int b = 0; b < batch; ++b) {
        for (int k = 0; k < c; ++k) {
            for (int j = 0; j < h * stride; ++j) {
                for (int i = 0; i < w * stride; ++i) {
                    int const in_index = b * w * h * c + k * w * h + (j / stride) * w + i / stride;
                    int const out_index = b * w * h * c * stride*stride + k * w * h * stride * stride + j * w * stride + i;
                    if (forward) {
                        out[out_index] = scale * in[in_index];
                    } else {
                        in[in_index] += scale * out[out_index];
                    }
                }
            }
        }
    }
}


void constrain_cpu(int const size, float const ALPHA, float * const X)
{
    for (int i = 0; i < size; ++i) {
        X[i] = fminf(ALPHA, fmaxf(-ALPHA, X[i]));
    }
}


void fix_nan_and_inf_cpu(float * const input, size_t const size)
{
    for (int i = 0; i < size; ++i) {
        float const val = input[i];
        if (isnan(val) || isinf(val)){
            input[i] = 1.0f / i;  // pseudo random value
        }
    }
}
