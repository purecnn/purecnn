// This code uses source of caffe from "https://github.com/BVLC/caffe".
// Prerequisite: BLAS via ATLAS, MKL, or Open BLAS

#include <vector>
#include <cstring>
#include <cblas.h>

template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <typename Dtype>
void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  caffe_set(height * width * channels, Dtype(0), data_im);
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

template <typename Dtype>
void conv2d_forward(const int* xsize, const Dtype* x, const int* wsize, const Dtype* w,
    const Dtype* bias, const int* padding, const int* stride, Dtype* y, Dtype* work) {

  const int ysize[4] = {
      (xsize[0] + 2 * padding[0] - wsize[0]) / stride[0] + 1,
      (xsize[1] + 2 * padding[1] - wsize[1]) / stride[1] + 1,
      wsize[3],
      xsize[3]};
  std::vector<Dtype> bias_multiplier((Dtype)1., ysize[0] * ysize[1]);

  for (int i = 0; i < xsize[3]; ++i) {
    const Dtype* input = &(x[i * xsize[0] * xsize[1] * xsize[2]]);
    Dtype* output = &(y[i * ysize[0] * ysize[1] * ysize[2]]);
    im2col_cpu(input, xsize[2], xsize[1], xsize[0], wsize[1], wsize[0], padding[1],
        padding[0], stride[1], stride[0], (int)1, (int)1, work);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, ysize[2], ysize[0] * ysize[1],
        wsize[0] * wsize[1] * wsize[2], (Dtype)1., w, work, (Dtype)0., output);

    if (NULL != bias) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, ysize[2], ysize[0] * ysize[1],
          1, (Dtype)1., bias, &(bias_multiplier[0]), (Dtype)1., output);
    }
  }
}

template <typename Dtype>
void conv2d_backward(const int* xsize, const Dtype* x, Dtype* gx, const int* wsize,
    const Dtype* w, Dtype* gw, const Dtype* bias, Dtype* gbias, const Dtype* gy,
    const int* padding, const int* stride, Dtype* work) {

  const int ysize[4] = {
      (xsize[0] + 2 * padding[0] - wsize[0]) / stride[0] + 1,
      (xsize[1] + 2 * padding[1] - wsize[1]) / stride[1] + 1,
      wsize[3],
      xsize[3]};
  std::vector<Dtype> bias_multiplier((Dtype)1., ysize[0] * ysize[1]);

  for (int i = 0; i < xsize[3]; ++i) {
    const Dtype* input = &(x[i * xsize[0] * xsize[1] * xsize[2]]);
    const Dtype* outdiff = &(gy[i * ysize[0] * ysize[1] * ysize[2]]);
    if (NULL != gbias) {
      caffe_cpu_gemv<Dtype>(CblasNoTrans, ysize[2], ysize[0] * ysize[1], 1.,
          outdiff, &(bias_multiplier[0]), 1., gbias);
    }
    if (NULL != gw) {
      im2col_cpu(input, xsize[2], xsize[1], xsize[0], wsize[1], wsize[0], padding[1],
          padding[0], stride[1], stride[0], (int)1, (int)1, work);
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, ysize[2],
          wsize[0] * wsize[1] * wsize[2], ysize[0] * ysize[1], (Dtype)1., outdiff, work,
          (Dtype)1., gw);
    }
    if (NULL != gx) {
      Dtype* indiff = &(gx[i * xsize[0] * xsize[1] * xsize[2]]);
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, wsize[0] * wsize[1] * wsize[2],
          ysize[0] * ysize[1], ysize[2], (Dtype)1., w, outdiff, (Dtype)0., work);
      col2im_cpu(work, xsize[2], xsize[1], xsize[0], wsize[1], wsize[0],
          padding[1], padding[0], stride[1], stride[0], (int)1, (int)1, indiff);
    }
  }
}

extern "C" {
  void conv2d_forward_f32(const int* xsize, const float* x, const int* wsize, const float* w,
      const float* bias, const int* padding, const int* stride, float* y, float* work) {
    conv2d_forward<float>(xsize, x, wsize, w, bias, padding, stride, y, work);
  }

  void conv2d_forward_f64(const int* xsize, const double* x, const int* wsize, const double* w,
      const double* bias, const int* padding, const int* stride, double* y, double* work) {
    conv2d_forward<double>(xsize, x, wsize, w, bias, padding, stride, y, work);
  }

  void conv2d_backward_f32(const int* xsize, const float* x, float* gx, const int* wsize,
      const float* w, float* gw, const float* bias, float* gbias, const float* gy,
      const int* padding, const int* stride, float* work) {
    conv2d_backward<float>(xsize, x, gx, wsize, w, gw, bias, gbias, gy, padding, stride,
        work);
  }

  void conv2d_backward_f64(const int* xsize, const double* x, double* gx, const int* wsize,
      const double* w, double* gw, const double* bias, double* gbias, const double* gy,
      const int* padding, const int* stride, double* work) {
    conv2d_backward<double>(xsize, x, gx, wsize, w, gw, bias, gbias, gy, padding, stride,
        work);
  }
}
