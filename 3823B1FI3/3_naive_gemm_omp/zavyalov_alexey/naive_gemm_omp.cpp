#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    std::vector<float> c(n * n);

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            int i_mult_n = i * n;
            int k_mult_n = k * n;
            float a_element = a[i_mult_n + k];

#pragma omp simd
            for (int j = 0; j < n; j ++) {
                c[i_mult_n + j] += a_element * b[k_mult_n + j];
            }
        }
    }

    return c;
}