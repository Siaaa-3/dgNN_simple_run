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


__global__ void fused_forward_kernel_test(int m, int nnz, int h, int f,
  float attn_drop, const float *attn_row,
  const float *attn_col, const int *row_ptr,
  const int *col_ind, const float *in_feat,
  const float negative_slope,
  float *edge_max, float *edge_sum,
  float *edge_mask, float *out_feat,
  unsigned long long seed, 
  const int* row_limits, int& nlimits,
  const int nthreads, const int m_per_block) {

  if(DEBUG_MODE==1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】code into fused_forward_kernel_test\n");

  int tid = threadIdx.x; 
  int hid = blockIdx.y; 
  
  // if(blockIdx.x >= nlimits) return;
  // if((blockIdx.x != 0 && (row_limits[blockIdx.x] == row_limits[blockIdx.x-1])) 
  //   || (blockIdx.x != (nlimits - 1) && (row_limits[blockIdx.x] == row_limits[blockIdx.x+1]))){
  //   printf("badRow=%d ", row_limits[blockIdx.x]);
  //   return;
  // }

  int left = row_limits[blockIdx.x]; // 第几行
  int right = row_limits[blockIdx.x+1];
  int lb = row_ptr[left]; //第几个元素
  int hb = row_ptr[right];

  int ptr = lb + threadIdx.x; //当前block内的线程处理全局第几个元素

  int rid = -1;
  while(left<right){
    int mid = (left+right)/2;
    if(row_ptr[mid+1]<=(lb+tid)){
      left = mid+1;
    }else{
      right = mid;
    }
  }
  if(ptr<hb) {
    rid = left;
    left = row_limits[blockIdx.x];
    // if(left!=right) printf("ptr:%d  left=%d, right=%d\n", ptr, left, right);
    // if(blockIdx.x==4 && tid<4) printf("tid=%d  rid=%d\n", tid, rid);
  }
  // if(blockIdx.x<4 && threadIdx.x > 57) printf("bloxk.x=%d  thread.x = %d: rid=%d\n", blockIdx.x, threadIdx.x, rid);
  float attn_row_val = 0;
  if(ptr < hb) attn_row_val = attn_row[rid * h + hid]; 

  if(DEBUG_MODE==1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 after compute rid\n");

  extern __shared__ int sh[];
  int* shared_row = sh;
  float* weightMax = (float*)&sh[nthreads]; // nthreads;
  float* expAll = (float*)&sh[nthreads + m_per_block]; // nthreads + m_per_block

  if(DEBUG_MODE==1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 after declare shared_memory\n");

  // if(ptr > hb) return;

  shared_row[tid] = rid;
  // if(rid==5&&tid==0){
  //   for(int i=0;i<32;i++){
  //     if(shared_row[i]!=-1) printf("%d ", shared_row[i]);
  //   }
  // }

  int cid = -1;
  float attn_col_val = 0;
  float weight = 0.0f;
  if (ptr < hb) {
    cid = col_ind[ptr]; 
    attn_col_val = attn_col[cid * h + hid]; 
    weight = attn_row_val + attn_col_val; 
    weight = LeakyRelu(weight, negative_slope); 
  }
  if(DEBUG_MODE==1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 after get weight=relu(w1+w2)\n");


  __syncwarp();
  float w = weight;
  for (int stride = 1; stride < 32; stride <<= 1) {
    float tmp = __shfl_down_sync(0xffffffff, w, stride, 32); 
    if((tid+stride) < nthreads){
      if(shared_row[tid + stride] == rid) {
        w = MAX(tmp, w);
      }      
    }
  }
  if(DEBUG_MODE==1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 after compute weightMax\n");


  if( rid>=0 && tid==(row_ptr[rid]-lb)) {
    weightMax[rid-left] = w; 
  }
  if(DEBUG_MODE==1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 after compute weightMax2\n");

  // if (threadIdx.x == 0 && rid>0)
  //   edge_max[rid * h + hid] = weightMax[rid];

  
  float exptmp = 0;
  if (ptr < hb && rid>=0) {
    exptmp = exp(weight - weightMax[rid-left]); 
  }
  if(DEBUG_MODE==1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 after compute exptmp\n");

  __syncwarp();
  for (int stride = 1; stride < 32; stride <<= 1) {
    float tmp = __shfl_down_sync(0xffffffff, exptmp, stride, 32); 
    if(tid+stride < nthreads){
      if(shared_row[tid+stride]==rid) {
        exptmp += tmp;
      }      
    }
  }

  if(DEBUG_MODE==1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 after compute expAll\n");

  if(ptr < hb && rid >= 0 && rid <= right && tid==(row_ptr[rid]-lb)) {
    expAll[rid-left] = exptmp; 
  }
  
  // if (threadIdx.x == 0)
  //   edge_sum[rid * h + hid] = expAll[rid];

  
  if (ptr < hb && edge_mask[ptr * h + hid] > attn_drop && rid>=0)
  // if (ptr < hb)
  {
    weight = exp(weight - weightMax[rid-left]) / expAll[rid-left];
    weight /= (1.0-attn_drop);
  }

  for(int fid=0;fid<f;fid++){
    float tmp_feat = 0;
    if(ptr<hb){
      tmp_feat = weight * in_feat[cid * h * f + hid * f + fid];
    }
    if(DEBUG_MODE==1) if(blockIdx.x==0&&threadIdx.x==0&&fid==f-1) printf("【1】 after compute tmp_feat\n");

    __syncwarp();
    for (int stride = 1; stride < 32; stride <<= 1) {
      float tmp = __shfl_down_sync(0xffffffff, tmp_feat, stride, 32); 
      if(tid+stride<nthreads){
        if(shared_row[tid+stride] == rid) {
          tmp_feat += tmp;
        }        
      }

    }

    if(ptr < hb && rid >= 0 && tid == (row_ptr[rid]-lb)) {
      out_feat[rid * h * f + hid * f + fid] = tmp_feat; 
    }
  }
  if(DEBUG_MODE==1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 Kernel end: after write into out_feat\n");
  
}

__global__ void compute_row_limits_kernel(int nblocks, int NNZ_PER_BLOCK, int m,
  int *csr_row_ptr, int* row_limits){
  if(DEBUG_MODE==1) if(blockIdx.x==0&&threadIdx.x==0) printf("【0b】 into compute row_limits kernel\n");


  int gid = blockDim.x * blockIdx.x + threadIdx.x; // 第几个block

  if(gid >= nblocks){
    return;
  }

  int s0 = NNZ_PER_BLOCK * gid; // 该行前面有不超过 s0 个的非零元
  // 最终效果为：可能有部分block处理的nnz个数大于NNZ_PER_BLOCK，但总体来说，每个block平均处理NNZ_PER_BLOCK个元素

  int left  = 0;
  int right = m;
  int mid   = (left + right) / 2;
  while((csr_row_ptr[left]) < s0 && left < mid && right > mid)
  {
    if((csr_row_ptr[mid]) <= s0)
    {
        left = mid;
    }
    else
    {
        right = mid;
    }
    mid = (left + right) / 2;
  }

  row_limits[gid] = left;

  if(gid == nblocks - 1)
  {
      row_limits[gid + 1] = m;
  }
  if(DEBUG_MODE==1) if(blockIdx.x==0&&threadIdx.x==0) printf("【0b】 end of compute row_limits kernel\n");

  // if(DEBUG_MODE==1){
  //   if(blockIdx.x==0&&threadIdx.x==0) {
  //     printf("\nrow_limits[%d]:\n", nblocks);
  //     for(int i=0; i<nblocks+1; i++) printf("%d ", row_limits[i]);
  //     printf("\n\n");
  //   }    
  // }


}