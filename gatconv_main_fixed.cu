#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include "computeUtil.h"


#include "fused_gatconv_kernel.cu"
#include "fused_gatconv_kernel_test.cu"

#include "utils.cpp"



int main() {

    cudaSetDevice(0);

    // printDevices();
    
    int m, nnz; 
    int h = 1, f = 4;
    float attn_drop = 0.1, negative_slope = 0.01;
    unsigned long long seed = 123456789;
    // int nthreads = 8; // 最小max_nnzPerRow, 最大1024
    int overload_threshold = 1024; // 一个block最多处理多少非零元
    int max_nnzPerRow = 0;

    std::string file = "m352w_nnz1919w";
    std::string mtxMode;
    std::string base = "./matrix/" + file;
    std::vector<int> row_ptr_host, col_ind_host, row_overload;
    std::string filename = base + "/csr_"+ file +".mtx";
    if(DEBUG_MODE==1)  std::cout << "filename:" << filename <<endl;

    if (readCSR(filename, m, nnz, row_ptr_host, col_ind_host, mtxMode, row_overload, overload_threshold, max_nnzPerRow) == 0) {
        std::cerr << "Read CSR matrix success." << std::endl;
    } else {
        std::cerr << "Failed to read the matrix from file." << std::endl;
    }

    if(DEBUG_MODE==1)  {
        std::cout << "mtxMode=" << mtxMode << endl;
        std::cout << "m=" << m << " nnz=" << nnz << endl;
        std::cout << "max_nnzPerRow=" << max_nnzPerRow << endl;
    }

    if(DEBUG_MODE==1)  printf("【0】 after read csr\n");

    
    std::string tail = std::to_string(m)+"_h"+std::to_string(h);

    std::string  attnrow_path = base + "/attn_row_m"+ tail +".txt";
    const char* file_attn_row = attnrow_path.c_str();

    std::string  attncol_path = base + "/attn_col_m"+ tail +".txt";
    const char* file_attn_col = attncol_path.c_str();

    std::string  infeat_path = base + "/in_feat_m"+ tail +"_f"+std::to_string(f)+".txt";
    const char* file_in_feat = infeat_path.c_str();

    std::vector<float> attn_row_host, attn_col_host, in_feat_host;
    readDenseMtx(file_attn_row, attn_row_host);
    readDenseMtx(file_attn_col, attn_col_host);
    readDenseMtx(file_in_feat, in_feat_host);
    if(DEBUG_MODE==1)  printf("【0】 after read dense mtx\n");

    float *edge_mask_host = (float*)malloc(sizeof(float) * nnz * h);
    for (int i = 0; i < nnz * h; ++i) {
        edge_mask_host[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    float *attn_row, *attn_col, *in_feat, *edge_max, *edge_sum, *edge_mask, *out_feat;
    cudaMalloc(&attn_row, sizeof(float) * m * h);
    cudaMalloc(&attn_col, sizeof(float) * m * h);
    cudaMalloc(&in_feat, sizeof(float) * m * h * f);
    cudaMalloc(&edge_max, sizeof(float) * m * h);
    cudaMalloc(&edge_sum, sizeof(float) * m * h);
    cudaMalloc(&edge_mask, sizeof(float) * nnz * h);
    cudaMalloc(&out_feat, sizeof(float) * m * h * f);

    cudaMemcpy(attn_row, attn_row_host.data(), attn_row_host.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(attn_col, attn_col_host.data(), attn_col_host.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_feat, in_feat_host.data(), in_feat_host.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_mask, edge_mask_host, sizeof(float) * nnz * h, cudaMemcpyHostToDevice);
    if(DEBUG_MODE==1) printf("【0】 after copy attns and in_feat\n");

    int *row_ptr, *col_ind, *row_limits;
    cudaMalloc(&row_ptr, row_ptr_host.size() * sizeof(int));

    cudaMalloc(&col_ind, sizeof(int) * nnz);
    cudaMemcpy(row_ptr, row_ptr_host.data(), row_ptr_host.size() * sizeof(int), cudaMemcpyHostToDevice);
    // cudaError_t err = cudaMemcpy(row_ptr, row_ptr_host.data(), row_ptr_host.size() * sizeof(int), cudaMemcpyHostToDevice);
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error copying to device: " << cudaGetErrorString(err) << std::endl;
    // }
    cudaMemcpy(col_ind, col_ind_host.data(), col_ind_host.size() * sizeof(int), cudaMemcpyHostToDevice);

    if(DEBUG_MODE==1)  printf("【0】 after copy row_ptr and col_ind\n");

    int NNZ_PER_BLOCK = getNnzPerBlock(max_nnzPerRow, overload_threshold);
    int nblocks = nnz / NNZ_PER_BLOCK + 2;
    if(DEBUG_MODE==1)  {
        std::cout << "max_nnzPerRow=" << max_nnzPerRow << endl;
        std::cout << "NNZ_PER_BLOCK=" << NNZ_PER_BLOCK << endl;
        std::cout << "nblocks=" << nblocks << endl;
    }
    dim3 blocks0(nblocks);
    dim3 threads0(NNZ_PER_BLOCK);


    int nlimits = nblocks + 1;
    cudaMalloc(&row_limits, nlimits * sizeof(int));
    compute_row_limits_kernel<<<blocks0, threads0>>>(nblocks, NNZ_PER_BLOCK, m, row_ptr, row_limits);
    cudaDeviceSynchronize(); 
    if(DEBUG_MODE==1)  printf("【0】 after compute row limits, should into 【1】\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 每个block共享内存 49152 bytes， 可容纳 12288 个int/float
    // int nthreads = NNZ_PER_BLOCK; // 最小max_nnzPerRow, 最大1024
    // int m_per_block = 2048; // 最小2*, 最大 12288-nthreads
    // dim3 blocks(nblocks, h);
    // dim3 threads(nthreads);
    // fused_forward_kernel_test<<<blocks, threads, nthreads * (sizeof(int)) + m_per_block * 2 * sizeof(float)>>>(
    //     m, nnz, h, f, attn_drop, attn_row, attn_col, row_ptr, col_ind,
    //     in_feat, negative_slope, edge_max, edge_sum, edge_mask, out_feat, seed,
    //     row_limits, nlimits, nthreads, m_per_block
    // );
    

    dim3 blocks(m, h);
    dim3 threads(32, (f + 31) / 32);
    fused_forward_kernel<<<blocks, threads, 32 * (sizeof(int)+sizeof(float))>>>(
        m, nnz, h, f, attn_drop, attn_row, attn_col, row_ptr, col_ind,
        in_feat, negative_slope, edge_max, edge_sum, edge_mask, out_feat, seed,
        row_limits);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if(DEBUG_MODE==1)  printf("【0】 out of fused_forward_kernel_test()\n");


    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\nKernel execution time: %f milliseconds\n", milliseconds);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    // 等待CUDA操作完成
    cudaDeviceSynchronize();
    if(DEBUG_MODE==1)  printf("【0】 all kernels execute end()\n");

    float *out_feat_host = (float*)malloc(sizeof(float) * m * h * f);
    cudaMemcpy(out_feat_host, out_feat, sizeof(float) * m * h * f, cudaMemcpyDeviceToHost);

    printf("out_feat[m,h,f]: (m=%d, h=%d, f=%d) \n", m, h, f);
    for(int j=0;j<16&&j<m;j++){
        printf("m=%d  ", j);
        for(int i=0;i<h*f&&i<8;i++){
            printf("%.3f ", out_feat_host[i+j*h*f]);
        }
        printf("\n");
    }

    free(out_feat_host);

    cudaFree(attn_row);
    cudaFree(attn_col);
    cudaFree(row_ptr);
    cudaFree(row_limits);
    cudaFree(col_ind);
    cudaFree(in_feat);
    cudaFree(edge_max);
    cudaFree(edge_sum);
    cudaFree(edge_mask);
    cudaFree(out_feat);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// nvcc -g -G gatconv_main_fixed.cu -o test
//  cuda-memcheck ./test 2>&1 | tee test_output.log