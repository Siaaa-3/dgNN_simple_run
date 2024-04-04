#include "computeUtil.h"
#include <cuda.h>
#include <stdio.h>
#include <unistd.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

// #define MAX(a, b) ((a < b) ? (b) : (a))
#define LeakyRelu(x, negative_slope) ((x > 0) ? (x) : ((x)*negative_slope))
using namespace std;



__device__ void appendIntToString(char* base, int num, int& index) {
  if (num == 0) {
      base[index++] = '0';
      return;
  }

  int start = index;
  while (num > 0) {
      int digit = num % 10;
      base[index++] = '0' + digit;
      num /= 10;
  }

  // 反转数字部分
  int end = index - 1;
  while (start < end) {
      char temp = base[start];
      base[start] = base[end];
      base[end] = temp;
      start++;
      end--;
  }
}

__device__ void appendFloatToString(char* base, float num, int& index, int decimalPlaces) {
  if (num < 0) {
      base[index++] = '-';
      num = -num;
  }

  // 处理整数部分
  int intPart = static_cast<int>(num);
  appendIntToString(base, intPart, index);

  base[index++] = '.'; // 小数点

  // 处理小数部分
  float fractionalPart = num - intPart;
  for (int i = 0; i < decimalPlaces; ++i) {
      fractionalPart *= 10;
      int digit = static_cast<int>(fractionalPart) % 10;
      base[index++] = '0' + digit;
  }
}

__device__ void appendString(char* base, const char* str, int& index) {
  while (*str != '\0') {
      base[index++] = *str++;
  }
}


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
  , char *strData, int strLength, int debug_nblocks
) {
  
  int tid = threadIdx.x; 
  int hid = blockIdx.y; 

  int left = row_limits[blockIdx.x]; // 第几行
  int right = row_limits[blockIdx.x + 1];
  int lb = row_ptr[left]; //第几个元素
  int hb = row_ptr[right];

  int ptr = lb + tid; //当前block内的线程处理全局第几个元素

  int rid = -1;
  while(left<right){
    int mid = (left + right) / 2;
    if(row_ptr[mid+1]<=(lb + tid)){
      left = mid + 1;
    }else{
      right = mid;
    }
  }
  if(ptr<hb) {
    rid = left;
    left = row_limits[blockIdx.x];
    right = row_limits[blockIdx.x + 1];
  }


  // 直接cpu逻辑计算结果
  float cpu_res[4] = {0.0f};
  float cpu_wmax = 0.0f;
  float cpu_expall = 0.0f;
  float cpu_w[10] = {0.0f};
  float cpu_exp[10] = {0.0f};
  float cpu_softmaxw[10] = {0.0f};
  float cpu_tmp_feat[10] = {0.0f};
  float cpu_in_feat[10] = {0.0f};
  if(ptr < hb){
    hid = hid;
    rid = rid;
    float expAll = 0, weightMax = -1e38;
    float attn_row_val = attn_row[rid * h + hid];
  
    for(int i = row_ptr[rid]; i < row_ptr[rid+1]; ++i){
      int cid = col_ind[i];

      float w = attn_row_val + attn_col[cid * h + hid];
      w = LeakyRelu(w, negative_slope);
      
      weightMax = MAX(w, weightMax);
    }
    cpu_wmax = weightMax;
  
    int i = 0;
    for(int ind = row_ptr[rid]; ind < nnz && i < 10; ind++){
      cpu_w[i] = LeakyRelu(attn_row_val + attn_col[col_ind[ind] * h + hid], negative_slope);
      cpu_exp[i] = exp(cpu_w[i] - weightMax);
      i++;
    }
    // for(int i = 0; i < 10; i++){
    //   cpu_softmaxw[i] = cpu_exp[i] / cpu_wmax;
    // }
    
    for(int i = row_ptr[rid]; i < row_ptr[rid+1]; ++i){
      int cid = col_ind[i];
      float w = LeakyRelu(attn_row_val + attn_col[cid * h + hid], negative_slope);
      float exptmp = exp(w - weightMax);
      // if(i - row_ptr[rid] < 10){
      //   cpu_w[i - row_ptr[rid]] = w;
      //   cpu_exp[i - row_ptr[rid]] = exptmp;
      // }
      expAll += exptmp;
    }
    cpu_expall = expAll;
  
    for(int fid = 0; fid < f; ++fid){
      float acc = 0;
      for(int i = row_ptr[rid]; i < row_ptr[rid+1]; ++i){
        int cid = col_ind[i];
        float w = attn_row_val + attn_col[cid * h + hid];
        w = LeakyRelu(w, negative_slope);
        
        w = exp(w - weightMax)/expAll;
        // w = w / (1 - attn_drop);

        cpu_softmaxw[i-row_ptr[rid]] = w;
        if(fid == 0) {
          cpu_tmp_feat[i-row_ptr[rid]] = w * in_feat[cid * h * f + hid * f + fid];
          cpu_in_feat[i-row_ptr[rid]] = in_feat[cid * h * f + hid * f + fid];
        }

        // if(mask>attn_drop) mask=0.5 attn_drop=0.1
        acc += w * in_feat[cid * h * f + hid * f + fid];
      }
      cpu_res[fid] = acc;
    }     
  }
  
  






  float attn_row_val = 0.0f;
  if(ptr < hb) attn_row_val = attn_row[rid * h + hid]; 
  if(blockIdx.x == 0) printf("attn_row_val=%f ", w);

  extern __shared__ int sh[];
  int* shared_row = sh;
  volatile float* shared_weight = (float*)&sh[nthreads]; 


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

  float w_stride[6];
  float tmp_w_stride[6];
  int row_stride[6] = {-1};
  float w_max_stride[6] = {0};

  int i = 0;
  __syncthreads();
  for(int stride = 1; stride < nthreads; stride <<= 1){
    if(ptr < hb) {
      w_stride[i] = shared_weight[tid];
    }
    if(lb + tid + stride < hb) {
      tmp_w_stride[i] = shared_weight[tid + stride];
      row_stride[i] = shared_row[tid + stride];
    }
    i++;

    float w = 0.0f;
    if(lb + tid + stride < hb){
        w = MAX(shared_weight[tid + stride], shared_weight[tid]);
    }
    __syncthreads();
    if(shared_row[tid + stride] == rid){
      shared_weight[tid] = w;
    }
    __syncthreads();
    if(ptr < hb) w_max_stride[i] = shared_weight[row_ptr[rid] - lb];
  }

  float weightMax = -1e38;
  if(ptr < hb) weightMax = shared_weight[row_ptr[rid] - lb];
  int max_index = 0;
  if(ptr < hb) max_index = row_ptr[rid] - lb;
  float test_max = 0.0f;
  if(ptr < hb) test_max = shared_weight[row_ptr[rid] - lb];
  __syncthreads();


  if (threadIdx.x == 0 && rid>0)
    edge_max[rid * h + hid] = weightMax;

  // memset(shared_weight, 0.0f, sizeof(float) * nthreads);
  float expw = 0.0f;
  if (ptr < hb) {
    expw = exp(weight - weightMax); 
    shared_weight[tid] = expw;
  }

  float exp_stride[6];
  float tmp_exp_stride[6];

  i = 0;
  __syncthreads(); 
  for(int stride = 1; stride < nthreads; stride <<= 1){
    exp_stride[i] = shared_weight[tid];
    if(lb + tid + stride < hb) {
      tmp_exp_stride[i] = shared_weight[tid + stride];
    }
    i++;

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
    // softmaxw /= (1.0-attn_drop);
  }

  float out_feat_stride[6] = {0.0f};
  float tmp_out_feat_stride[6] = {0.0f};
  int ind_stride[6] = {0};
  float final_feat = 0.0f;
  i = 0;
  for(int fid = 0; fid < f; fid++){
    float tmp_feat = 0;
    if(ptr < hb){
      tmp_feat = softmaxw * in_feat[cid * h * f + hid * f + fid];
      shared_weight[tid] = tmp_feat;
    }
    __syncthreads();

    for(int stride = 1; stride < nthreads; stride <<= 1){
      if(ptr < hb && fid == 0) {
        out_feat_stride[i] = shared_weight[tid];
        ind_stride[i] = stride;
      }
      if(lb + tid + stride < hb && fid == 0) tmp_out_feat_stride[i] = shared_weight[tid + stride];
     i++;
      
      float w = 0.0f;
      if(lb + tid + stride < hb){
        // if(shared_row[tid + stride] == rid){
          w = shared_weight[tid] + shared_weight[tid + stride];
        // }
      }
      __syncthreads();
      if(shared_row[tid + stride] == rid) shared_weight[tid] = w;
      __syncthreads();

    }
    if(ptr < hb && tid == (row_ptr[rid] - lb)) {
      if(fid == 0) final_feat = shared_weight[tid];
      out_feat[rid * h * f + hid * f + fid] = shared_weight[tid]; 
    }
    __syncthreads();
    
    int index = 0;
    if(fid == 0 && ptr < hb && (shared_weight[row_ptr[rid]-lb] - cpu_res[fid] > 1e-4) && (blockIdx.x < debug_nblocks)){
      index = 0;
      char* currentStr = &strData[blockIdx.x * nthreads * strLength + tid * strLength];

      // bid left right lb hb
      appendString(currentStr, "bid=", index);
      appendIntToString(currentStr, blockIdx.x, index);
      appendString(currentStr, " left=", index);
      appendIntToString(currentStr, left, index);
      appendString(currentStr, " right=", index);
      appendIntToString(currentStr, right, index);
      appendString(currentStr, " lb=", index);
      appendIntToString(currentStr, lb, index);
      appendString(currentStr, " hb=", index);
      appendIntToString(currentStr, hb, index);
      appendString(currentStr, " ptr=", index);
      appendIntToString(currentStr, ptr, index);
      appendString(currentStr, "\n", index);

      appendString(currentStr, "rid=", index);
      appendIntToString(currentStr, rid, index);
      appendString(currentStr, "\n", index);

      appendString(currentStr, "tid=", index);
      appendIntToString(currentStr, tid, index);      
      appendString(currentStr, " cid=", index);
      appendIntToString(currentStr, cid, index);
      appendString(currentStr, " cval=", index);
      appendFloatToString(currentStr, attn_col_val, index, 5);
      appendString(currentStr, "\n\n", index);

      appendString(currentStr, "cpu_softmaxw[10]\n", index);
      for(int i = 0; i < 10; i++){
        appendFloatToString(currentStr, cpu_softmaxw[i], index, 5);      
        appendString(currentStr, " ", index);
      }
      appendString(currentStr, "\n", index);
      appendString(currentStr, "cpu_in_feat[10]\n", index);
      for(int i = 0; i < 10; i++){
        appendFloatToString(currentStr, cpu_in_feat[i], index, 5);      
        appendString(currentStr, " ", index);
      }
      appendString(currentStr, "\n", index);
      appendString(currentStr, "cpu_tmp_feat[10]\n", index);
      for(int i = 0; i < 10; i++){
        appendFloatToString(currentStr, cpu_tmp_feat[i], index, 5);      
        appendString(currentStr, " ", index);
      }
      appendString(currentStr, "\n\n", index);
      // 
      appendString(currentStr, "feat_stride[6]\n", index);
      appendString(currentStr, "stride feat tmp_feat\n", index);
      for(int i = 0; i < 6; i++){
        appendIntToString(currentStr, ind_stride[i], index);   
        appendString(currentStr, " ", index);
        appendFloatToString(currentStr, out_feat_stride[i], index, 5);   
        appendString(currentStr, " ", index);
        appendFloatToString(currentStr, tmp_out_feat_stride[i], index, 5);   
        appendString(currentStr, "\n", index);
      }
      appendString(currentStr, "\n", index);


      appendString(currentStr, "final_feat\n", index);
      for(int i = 0; i < 4; i++){
        appendFloatToString(currentStr, final_feat, index, 5);      
        appendString(currentStr, " ", index);
      }
      appendString(currentStr, "\n\n", index);


      appendString(currentStr, "tmpf=softmax(w)*infeat", index);
      appendString(currentStr, "【", index);
      appendFloatToString(currentStr, tmp_feat, index, 5);   
      appendString(currentStr, "=", index);
      appendFloatToString(currentStr, softmaxw, index, 5);    
      appendString(currentStr, "*", index);
      appendFloatToString(currentStr, in_feat[cid * h * f + hid * f + fid], index, 5);    
      appendString(currentStr, "】", index);
      appendString(currentStr, "\n\n", index);

      // fid cpu_res outfeat
      appendString(currentStr, "fid=", index);
      appendIntToString(currentStr, fid, index);
      appendString(currentStr, " outfeat=", index);
      appendFloatToString(currentStr, shared_weight[row_ptr[rid]-lb], index, 5);
      appendString(currentStr, " cpu_res=", index);
      appendFloatToString(currentStr, cpu_res[fid], index, 5);   
      appendString(currentStr, "(tid=", index);
      appendIntToString(currentStr, row_ptr[rid]-lb, index);
      appendString(currentStr, ")", index);
      appendString(currentStr, "\n\n", index);
      currentStr[index] = '\0';
    }
  }  
}

__global__ void compute_row_limits_kernel(const int nblocks, const int NNZ_PER_BLOCK, const int m,
  const int* csr_row_ptr, int* row_limits){
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

}
