#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void cnt_keys(int *key, int *bucket, int n, int range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    atomicAdd(&bucket[key[idx]], 1);
}

int main() {
  int n = 500;
  int range = 50;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *d_key, *d_bucket;
  std::vector<int> bucket(range, 0);
  cudaMallocManaged(&d_key, n * sizeof(int));
  cudaMemcpy(d_key, key.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMallocManaged(&d_bucket, range * sizeof(int));
  cudaMemcpy(d_bucket, bucket.data(), range * sizeof(int), cudaMemcpyHostToDevice);

  int M = 128;
  cnt_keys<<<(n + M- 1)/M, M>>>(d_key, d_bucket, n, range);
  cudaMemcpy(bucket.data(), d_bucket, range * sizeof(int), cudaMemcpyDeviceToHost);

  int j = 0;
  for (int i = 0; i < range; i++) {
      for (; bucket[i] > 0; bucket[i]--) {
          key[j++] = i;
      }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(d_key);
  cudaFree(d_bucket);
}
