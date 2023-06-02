#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <immintrin.h>


// The following function returns the sum of the numbers of __m256 variable.
// It is written by ChatGPT. 
float hsum(__m256 v){
v = _mm256_hadd_ps(v, v); // v = [v3+v2, v1+v0, v7+v6, v5+v4, v3+v2, v1+v0, v7+v6, v5+v4]
v = _mm256_hadd_ps(v, v); // v = [v1+v0+v3+v2, v1+v0+v3+v2, v5+v4+v7+v6, v5+v4+v7+v6, v1+v0+v3+v2, v1+v0+v3+v2, v5+v4+v7+v6, v5+v4+v7+v6]

// Extract the lower 128 bits
__m128 vlow = _mm256_castps256_ps128(v); // vlow = [v1+v0+v3+v2, v1+v0+v3+v2, v1+v0+v3+v2, v1+v0+v3+v2]

// Extract the higher 128 bits
__m128 vhigh = _mm256_extractf128_ps(v, 1); // vhigh = [v5+v4+v7+v6, v5+v4+v7+v6, v5+v4+v7+v6, v5+v4+v7+v6]

// Add the lower and higher 128 bits
__m128 vsum = _mm_add_ps(vlow, vhigh); // vsum = [sum, sum, sum, sum]

// Extract the first element of the result
float sum = _mm_cvtss_f32(vsum); // sum = sum of the eight floats
return sum;  
}


int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  for(int i=0; i<N; i++) {
    __m256 xi_vec = _mm256_set1_ps(x[i]);
    __m256 yi_vec = _mm256_set1_ps(y[i]);
    __m256 m_vec = _mm256_load_ps(m);
    __m256 xj_vec = _mm256_load_ps(x);
    __m256 yj_vec = _mm256_load_ps(y);
    __m256 rx_vec = _mm256_sub_ps(xi_vec, xj_vec);
    __m256 ry_vec = _mm256_sub_ps(yi_vec, yj_vec);
    __m256 rsq_vec = _mm256_add_ps(_mm256_mul_ps(rx_vec, rx_vec), _mm256_mul_ps(ry_vec, ry_vec));
    __m256 rinv_vec = _mm256_rsqrt_ps(rsq_vec);
    __m256 rinv_cubic_vec = _mm256_mul_ps(_mm256_mul_ps(rinv_vec,rinv_vec),rinv_vec);
    __m256 mask = _mm256_cmp_ps(rsq_vec, _mm256_set1_ps(1e-10), _CMP_GT_OQ);
    __m256 fxdiff_vec = _mm256_mul_ps(rx_vec, _mm256_mul_ps(m_vec, rinv_cubic_vec));
    __m256 fydiff_vec = _mm256_mul_ps(ry_vec, _mm256_mul_ps(m_vec, rinv_cubic_vec));
    fx[i] -= hsum(_mm256_blendv_ps(_mm256_setzero_ps(), fxdiff_vec, mask));
    fy[i] -= hsum(_mm256_blendv_ps(_mm256_setzero_ps(), fydiff_vec, mask));
    //for(int j=0; j<N; j++) {
        //// using masking for if
      //if(i != j) {
        //float rx = x[i] - x[j];
        //float ry = y[i] - y[j]; //use _mm256_sub_ps
        //float r = std::sqrt(rx * rx + ry * ry); // calculate 1/r using _mm256_rsqrt_ps
        //if(i==j+1)printf("fx_diff_i,j = %f, i = %d, j = %d \n", rx * m[j] / (r*r*r), i, j);
        //fx[i] -= rx * m[j] / (r * r * r);
        //fy[i] -= ry * m[j] / (r * r * r);
      //}
    //}
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
