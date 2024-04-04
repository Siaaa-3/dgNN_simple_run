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
  const int nthreads, const int m_per_block
) {
  if(DEBUG_MODE>=1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 into kernel\n");
  
  int tid = threadIdx.x; 
  int hid = blockIdx.y; 

  int left = row_limits[blockIdx.x]; // 第几行
  int right = row_limits[blockIdx.x + 1];
  int lb = row_ptr[left]; //第几个元素
  int hb = row_ptr[right];

  int ptr = lb + tid; //当前block内的线程处理全局第几个元素

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
    right = row_limits[blockIdx.x + 1];
  }

  float attn_row_val = 0;
  if(ptr < hb) attn_row_val = attn_row[rid * h + hid]; 

  extern __shared__ int sh[];
  int* shared_row = sh;
  float* shared_weight = (float*)&sh[nthreads]; 

  if(ptr < hb) shared_row[tid] = rid;

  int cid = -1;
  float attn_col_val = 0;
  float weight = 0.0f;
  if (ptr < hb) {
    cid = col_ind[ptr]; 
    attn_col_val = attn_col[cid * h + hid]; 
    weight = attn_row_val + attn_col_val; 
    weight = LeakyRelu(weight, negative_slope); 
    shared_weight[tid] = weight;
  }

  __syncthreads();
  for(int stride = 1; stride < nthreads; stride <<= 1){
    float w = 0.0f;
    if(lb + tid + stride < hb){
      w = MAX(shared_weight[tid + stride], shared_weight[tid]);
    }
    __syncthreads();
    if(shared_row[tid + stride] == rid){
      shared_weight[tid] = w;
    }
    __syncthreads();
  }

  float weightMax = -1e38;
  if(ptr < hb) weightMax = shared_weight[row_ptr[rid] - lb];
  __syncthreads();

  if (threadIdx.x == 0 && rid>0)
    edge_max[rid * h + hid] = weightMax;

  float expw = 0.0f;
  if (ptr < hb) {
    expw = exp(weight - weightMax); 
    shared_weight[tid] = expw;
  }
  __syncthreads(); 
  for(int stride = 1; stride < nthreads; stride <<= 1){
    float w = 0.0f;
    if(lb + tid + stride < hb){
        w = shared_weight[tid] + shared_weight[tid + stride];
    }
    __syncthreads();
    if(shared_row[tid + stride] == rid){
      shared_weight[tid] = w;
    }
    __syncthreads();
  }

  float expAll = 0.0f;
  if(ptr < hb) expAll = shared_weight[row_ptr[rid] - lb];
  __syncthreads();
  
  if (threadIdx.x == 0)
    edge_sum[rid * h + hid] = expAll;

  float softmaxw = 0.0f;
  if (ptr < hb && edge_mask[ptr * h + hid] > attn_drop)
  {
    softmaxw = expw / expAll;
    softmaxw /= (1.0 - attn_drop);
  }
  for(int fid = 0; fid < f; fid++){
    float tmp_feat = 0;
    if(ptr < hb){
      tmp_feat = softmaxw * in_feat[cid * h * f + hid * f + fid];
      shared_weight[tid] = tmp_feat;
    }
    __syncthreads();

    for(int stride = 1; stride < nthreads; stride <<= 1){
      float w = 0.0f;
      if(lb + tid + stride < hb){
        w = shared_weight[tid] + shared_weight[tid + stride];
      }
      __syncthreads();
      if(shared_row[tid + stride] == rid){
        shared_weight[tid] = w;
        }
      __syncthreads();

    }
    if(ptr < hb && tid == (row_ptr[rid] - lb)) {
      out_feat[rid * h * f + hid * f + fid] = shared_weight[tid]; 
    }
    __syncthreads();
  } 
}

__global__ void compute_row_limits_kernel(const int nblocks, const int NNZ_PER_BLOCK, const int m,
  const int* csr_row_ptr, int* row_limits){
  if(DEBUG_MODE>=1) if(blockIdx.x==0&&threadIdx.x==0) printf("【0b】 into compute row_limits kernel\n");
  int gid = blockDim.x * blockIdx.x + threadIdx.x; // 第几个block

  if(gid >= nblocks){
    return;
  }

  int s0 = NNZ_PER_BLOCK * gid;

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
  if(DEBUG_MODE>=1) if(blockIdx.x==0&&threadIdx.x==0) printf("【0b】 end of compute row_limits kernel\n");

  if(DEBUG_MODE>=4){
    if(blockIdx.x==0&&threadIdx.x==0) {

      printf("\ncsr_row_ptr[%d]:\n", nblocks);
      for(int i=30150; i<30250; i++) {
        if(i%10==0) printf("\n【%d】", i);
        printf("%d ", csr_row_ptr[i]);
      }
    }    
  }
}

// __global__ void fused_forward_kernel_test(int m, int nnz, int h, int f,
//   float attn_drop, const float *attn_row,
//   const float *attn_col, const int *row_ptr,
//   const int *col_ind, const float *in_feat,
//   const float negative_slope,
//   float *edge_max, float *edge_sum,
//   float *edge_mask, float *out_feat,
//   unsigned long long seed, 
//   const int* row_limits, int& nlimits,
//   const int nthreads, const int m_per_block) {

//   if(DEBUG_MODE>=1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】code into fused_forward_kernel_test\n");
  
//   int tid = threadIdx.x; 
//   int hid = blockIdx.y; 

//   int left = row_limits[blockIdx.x]; // 第几行
//   int right = row_limits[blockIdx.x+1];
//   int lb = row_ptr[left]; //第几个元素
//   int hb = row_ptr[right];

//   // if(blockIdx.x==5 && tid==0) printf("left=%d(行), right=%d(行), lb=%d(nnz), rb=%d(nnz)\n", left, right, lb, hb); 

//   int ptr = lb + threadIdx.x; //当前block内的线程处理全局第几个元素

//   int rid = -1;
//   while(left<right){
//     int mid = (left+right)/2;
//     if(row_ptr[mid+1]<=(lb+tid)){
//       left = mid+1;
//     }else{
//       right = mid;
//     }
//   }
//   if(ptr<hb) {
//     rid = left;
//     left = row_limits[blockIdx.x];
//     // if(left!=right) printf("ptr:%d  left=%d, right=%d\n", ptr, left, right);
//     // if(blockIdx.x==4 && tid<4) printf("tid=%d  rid=%d\n", tid, rid);
//   }
//   // if(blockIdx.x<4 && threadIdx.x > 57) printf("bloxk.x=%d  thread.x = %d: rid=%d\n", blockIdx.x, threadIdx.x, rid);

//   float attn_row_val = 0;
//   if(ptr < hb) attn_row_val = attn_row[rid * h + hid]; 
//   if(DEBUG_MODE>=0) if(blockIdx.x==0&&threadIdx.x==0) {printf("\ntestRid=%d, 查看过程中间各结果\n",testRid );}
//   if(DEBUG_MODE>=0)if(rid==testRid) printf("left=%d(行), right=%d(行), lb=%d(nnz), rb=%d(nnz)\n", left, right, lb, hb); 
//   if(DEBUG_MODE>=0) if(rid==testRid) printf("attn_row_val=%f\n", attn_row_val);

//   if(DEBUG_MODE>=1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 after compute rid\n");

//   extern __shared__ int sh[];
//   int* shared_row = sh;
//   float* weightMax = (float*)&sh[nthreads]; 
//   float* expAll = (float*)&sh[nthreads + m_per_block]; 

//   if(DEBUG_MODE>=1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 after declare shared_memory\n");

//   // if(ptr > hb) return;

//   shared_row[tid] = rid;
//   // if(rid==5&&tid==0){
//   //   for(int i=0;i<32;i++){
//   //     if(shared_row[i]!=-1) printf("%d ", shared_row[i]);
//   //   }
//   // }

//   int cid = -1;
//   float attn_col_val = 0;
//   float weight = 0.0f;
//   if (ptr < hb) {
//     cid = col_ind[ptr]; 
//     attn_col_val = attn_col[cid * h + hid]; 
//     if(DEBUG_MODE>=0)if(rid==testRid) printf("tid=%d: attn_col_val[%d]=%f\n",tid, cid, attn_col_val);
//     weight = attn_row_val + attn_col_val; 
//     weight = LeakyRelu(weight, negative_slope); 
//   }
//   if(DEBUG_MODE>=1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 after get weight=relu(w1+w2)\n");


//   __syncwarp();
//   float w = weight;
//   for (int stride = 1; stride < 32; stride <<= 1) {
//     float tmp = __shfl_down_sync(0xffffffff, w, stride, 32); 
//     if(DEBUG_MODE>=0)if(rid==testRid) printf("tid=%d stride=%d: w=%f, tmpW=%f, shared_row[tid+stride]=%d  ==?  rid=%d\n",
//       tid, stride, w, tmp, shared_row[tid+stride], rid);
//     if((tid+stride) < nthreads){
//       if(shared_row[tid + stride] == rid) {
//         w = MAX(tmp, w);
//       }      
//     }
//   }
//   if(DEBUG_MODE>=1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 after compute weightMax\n");


//   if( rid>=0 && tid==(row_ptr[rid]-lb)) {
//     weightMax[rid-left] = w; 
//   }
//   if(DEBUG_MODE>=0) if(rid==testRid) printf("tid=%d: weightMax=%f\n",tid, weightMax[rid-left]);
//   if(DEBUG_MODE>=1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 after compute weightMax2\n");

//   // if (threadIdx.x == 0 && rid>0)
//   //   edge_max[rid * h + hid] = weightMax[rid];

  
//   float exptmp = 0;
//   if (ptr < hb && rid>=0) {
//     exptmp = exp(weight - weightMax[rid-left]); 
//   }
//   if(DEBUG_MODE>=1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 after compute exptmp\n");

//   __syncwarp();
//   for (int stride = 1; stride < 32; stride <<= 1) {
//     float tmp = __shfl_down_sync(0xffffffff, exptmp, stride, 32); 
//     if(tid+stride < nthreads){
//       if(shared_row[tid+stride]==rid) {
//         exptmp += tmp;
//       }      
//     }
//   }

//   if(DEBUG_MODE>=1) if(blockIdx.x==0&&threadIdx.x==0) printf("【1】 after compute expAll\n");

//   if(ptr < hb && rid >= 0 && rid <= right && tid==(row_ptr[rid]-lb)) {
//     expAll[rid-left] = exptmp; 
//   }
//   if(DEBUG_MODE>=0)if(rid==testRid) printf("expAll=%f\n", expAll[rid-left]);
  
//   // if (threadIdx.x == 0)
//   //   edge_sum[rid * h + hid] = expAll[rid];

  
//   if (ptr < hb && edge_mask[ptr * h + hid] > attn_drop && rid>=0)
//   // if (ptr < hb)
//   {
//     weight = exp(weight - weightMax[rid-left]) / expAll[rid-left];
//     weight /= (1.0-attn_drop);
//   }

//   for(int fid=0;fid<f;fid++){
//     float tmp_feat = 0;
//     if(ptr<hb){
//       tmp_feat = weight * in_feat[cid * h * f + hid * f + fid];
//       if(DEBUG_MODE>=0)if(rid==testRid&&fid==0) printf("cid=%d【acc=w*in_feat】[%f=%f*%f] \n",cid, tmp_feat, weight, in_feat[cid * h * f + hid * f + fid]);
//     }
//     if(DEBUG_MODE>=1) if(blockIdx.x==0&&threadIdx.x==0&&fid==f-1) printf("【1】 after compute tmp_feat\n");

//     __syncwarp();
//     for (int stride = 1; stride < 32; stride <<= 1) {
//       float tmp = __shfl_down_sync(0xffffffff, tmp_feat, stride, 32); 
//       if(tid+stride<nthreads){
//         if(shared_row[tid+stride] == rid) {
//           tmp_feat += tmp;
//         }        
//       }
//     }


//     if(ptr < hb && rid >= 0 && tid == (row_ptr[rid]-lb)) {
//       out_feat[rid * h * f + hid * f + fid] = tmp_feat; 
//     }
//   }  
// }