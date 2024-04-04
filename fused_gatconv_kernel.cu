#include "computeUtil.h"
#include <cuda.h>
// #include <torch/types.h>

#include <stdio.h>
#include <unistd.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

// #define MAX(a, b) ((a < b) ? (b) : (a))
#define LeakyRelu(x, negative_slope) ((x > 0) ? (x) : ((x)*negative_slope))
using namespace std;


__global__ void fused_forward_kernel(int m, int nnz, int h, int f,
  float attn_drop, const float *attn_row,
  const float *attn_col, const int *row_ptr,
  const int *col_ind, const float *in_feat,
  const float negative_slope,
  float *edge_max, float *edge_sum,
  float *edge_mask, float *out_feat,
  unsigned long long seed, const int* row_limits) {

  int rid = blockIdx.x;
  int hid = blockIdx.y; 
  int lb = row_ptr[rid];
  int hb = row_ptr[rid + 1];
  int ptr = lb + threadIdx.x; 
  int loop = (hb - lb + 31) / 32; 
  extern __shared__ float val_sh[];
  float *attn_val_sh = val_sh;
  int *cid_sh = (int *)&val_sh[32];

  // attn_row维度：(N, n_heads)
  float attn_row_val = attn_row[rid * h + hid]; 
  // float attn_row_val=0;

  float weightMax = -1e38;
  // computing weightMax
  for (int j = 0; j < loop; j++) {
    int pid = ptr + (j << 5);  
    float weight = -1e38;
    if (pid < hb) {
      int cid = col_ind[pid]; 
      float attn_col_val = attn_col[cid * h + hid]; 
      weight = attn_row_val + attn_col_val; 
      weight = LeakyRelu(weight, negative_slope); 
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1) {
      float tmp = __shfl_xor_sync(0xffffffff, weight, stride, 32); 
      weight = MAX(tmp, weight);
    }
    weightMax = MAX(weight, weightMax);
  }


  if (threadIdx.x == 0)
    edge_max[rid * h + hid] = weightMax;

  float expAll = 0;
  for (int j = 0; j < loop; j++) {
    int pid = ptr + (j << 5);
    float exptmp = 0;
    if (pid < hb) {
      int cid = col_ind[pid];
      float attn_col_val = attn_col[cid * h + hid];
      float weight = attn_row_val + attn_col_val;
      weight = LeakyRelu(weight, negative_slope);
      exptmp = exp(weight - weightMax); 
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1) {
      float tmp = __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
      exptmp += tmp;
    }
    expAll += exptmp;
  }


  if (threadIdx.x == 0)
    edge_sum[rid * h + hid] = expAll;

  // if(threadIdx.x==0) printf("rid=%d expAll=%f\n", rid, expAll);

  // 处理特征向量
  int fid = threadIdx.y * 32 + threadIdx.x;
  // for (int fid = threadIdx.x; fid < (f + 31) / 32 * 32; fid += 32)
  {
    float acc = 0;
    for (int j = 0; j < loop; j++) { 
      int pid = ptr + (j << 5);
      float weight = 0;
      int cid = 0;
      if (pid < hb && edge_mask[pid * h + hid] > attn_drop)
      // if (pid < hb)
      {
        cid = col_ind[pid]; 
        float attn_col_val = attn_col[cid * h + hid];
        weight = attn_row_val + attn_col_val;
        weight = LeakyRelu(weight, negative_slope);
        weight = exp(weight - weightMax) / expAll;
      }
      // if(blockIdx.x==2) printf("tid=%d  weight=%f\n", threadIdx.x, weight );
      attn_val_sh[threadIdx.x] = weight / (1.0 - attn_drop); 
      // attn_val_sh[threadIdx.x] = weight; 
      cid_sh[threadIdx.x] = cid;
      __syncwarp();
      int jj = lb + (j << 5); 
      for (int kk = 0; kk < 32 && jj + kk < hb; kk++) { 
        int cid = cid_sh[kk];
        float val = attn_val_sh[kk];
        acc += val * in_feat[cid * h * f + hid * f + fid]; 

        // if(blockIdx.x==0 && blockIdx.y==0 && j==0){
        //   printf("block(%d, %d) warp(%d) tid(%d, %d): cid=%d, hid=%d, fid=%d\n", 
        //   blockIdx.x, blockIdx.y, j, threadIdx.x, threadIdx.y, cid, hid, fid);
        // }
      }
      __syncwarp();
    }
    // if(rid>=0&&fid==0)  printf("rid==%d fid==%d: acc=%f\n", rid, fid, acc);
    if (fid < f)
    out_feat[rid * h * f + hid * f + fid] = acc; 
  }
}

