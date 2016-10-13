#include <math.h>

void relu(float *x, float *y, int n) {
    for (int i = 0; i < n; i++) y[i] = x[i] > 0 ? x[i] : 0;
}

void relu_grad(float *x, float *gx, float *y, float *gy, int n) {
    for (int i = 0; i < n; i++) gx[i] += gy[i] * (x[i] > 0 ? x[i] : 0);
}

void sigmoid(float *x, float *y, int n) {
    for (int i = 0; i < n; i++) y[i] = 1 / (1 + exp(-x[i]));
}

void sigmoid_grad(float *x, float *gx, float *y, float *gy, int n) {
    for (int i = 0; i < n; i++) gx[i] += gy[i] * y[i] * (1 - y[i]);
}

void tanh(float *x, float *y, int n) {
    for (int i = 0; i < n; i++) y[i] = tanh(x[i]);
}

void tanh_grad(float *x, float *gx, float *y, float *gy, int n) {
    for (int i = 0; i < n; i++) gx[i] += gy[i] * (1 - y[i] * y[i]);
}
