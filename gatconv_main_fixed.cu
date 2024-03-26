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
    int overload_threshold = 1024; // 一个block最多处理多少非零元
    int max_nnzPerRow = 0;

    std::string file = "m8w_nnz20w";
    std::string program = "test"; //test dgnn
    if(program=="test")printf("run test\n");
    else if(program=="dgnn") printf("run dgNN\n");
    std::string mtxMode;
    std::string base = "./matrix/" + file;
    std::vector<int> row_ptr_host, col_ind_host, row_overload;
    std::string filename = base + "/csr_"+ file +".mtx";
    if(DEBUG_MODE>=0)  std::cout << "filename:" << filename <<endl;

    if (readCSR(filename, m, nnz, row_ptr_host, col_ind_host, mtxMode, row_overload, overload_threshold, max_nnzPerRow) == 0) {
        std::cerr << "Read CSR matrix success." << std::endl;
    } else {
        std::cerr << "Failed to read the matrix from file." << std::endl;
    }

    // printf("nnz of row_ptr[77539] = %d\n", row_ptr_host[77540] - row_ptr_host[77539]);

    if(DEBUG_MODE>=0)  {
        std::cout << "mtxMode=" << mtxMode << endl;
        std::cout << "m=" << m << " nnz=" << nnz << endl;
        std::cout << "max_nnzPerRow=" << max_nnzPerRow << endl;
    }

    if(DEBUG_MODE>=1)  printf("【0】 after read csr\n");

    
    std::string tail = std::to_string(m)+"_h"+std::to_string(h);
    std::string  attnrow_path = base + "/attn_row_m"+ tail +".txt";
    std::string  attncol_path = base + "/attn_col_m"+ tail +".txt";
    std::string  infeat_path = base + "/in_feat_m"+ tail +"_f"+std::to_string(f)+".txt";

    std::vector<float> attn_row_host, attn_col_host, in_feat_host;
    readDenseMtx(attnrow_path, attn_row_host);
    readDenseMtx(attncol_path, attn_col_host);
    readDenseMtx(infeat_path, in_feat_host);
    if(DEBUG_MODE>=1)  printf("【0】 after read dense mtx\n");


    float *edge_mask_host = (float*)malloc(sizeof(float) * nnz * h);
    for (int i = 0; i < nnz * h; ++i) {
        // edge_mask_host[i] = static_cast<float>(rand()) / RAND_MAX;
        edge_mask_host[i] = 0.5f;
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
    if(DEBUG_MODE>=1) printf("【0】 after copy attns and in_feat\n");

    int *row_ptr, *col_ind, *row_limits;
    cudaMalloc(&row_ptr, row_ptr_host.size() * sizeof(int));

    cudaMalloc(&col_ind, sizeof(int) * nnz);
    cudaMemcpy(row_ptr, row_ptr_host.data(), row_ptr_host.size() * sizeof(int), cudaMemcpyHostToDevice);
    // cudaError_t err = cudaMemcpy(row_ptr, row_ptr_host.data(), row_ptr_host.size() * sizeof(int), cudaMemcpyHostToDevice);
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error copying to device: " << cudaGetErrorString(err) << std::endl;
    // }
    cudaMemcpy(col_ind, col_ind_host.data(), col_ind_host.size() * sizeof(int), cudaMemcpyHostToDevice);

    if(DEBUG_MODE>=1)  printf("【0】 after copy row_ptr and col_ind\n");
 
    int NNZ_PER_BLOCK = 0;
    int nblocks = 0;
    getLimitsParams(max_nnzPerRow, overload_threshold, nnz, NNZ_PER_BLOCK, nblocks);

    if(DEBUG_MODE>=0)  {
        std::cout << "\n计算rowlimits的参数:" << std::endl; 
        std::cout << "max_nnzPerRow=" << max_nnzPerRow << "" << endl;
        std::cout << "NNZ_PER_BLOCK=" << NNZ_PER_BLOCK << "" << endl;
        std::cout << "nblocks=" << nblocks << "" <<endl;
    }
    
    dim3 blocks0(3);
    dim3 threads0(1024);

    int nlimits = nblocks + 1;
    cudaMalloc(&row_limits, nlimits * sizeof(int));
    compute_row_limits_kernel<<<blocks0, threads0>>>(nblocks, NNZ_PER_BLOCK, m, row_ptr, row_limits);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
    cudaDeviceSynchronize(); 

    int nthreads = NNZ_PER_BLOCK;
    int m_per_block = NNZ_PER_BLOCK;
    int *row_limits_host = (int*)malloc(nlimits * sizeof(int));
    cudaMemcpy(row_limits_host, row_limits, nlimits * sizeof(int), cudaMemcpyDeviceToHost);
    getForwardParams(row_limits_host, nlimits, row_ptr_host, nthreads, m_per_block);
    if(DEBUG_MODE >= 0) {
        printf("\n\n共享内存参数:\n");
        printf("nthreads=%d, m_per_block=%d\n", nthreads, m_per_block);
    }

    if(DEBUG_MODE>=1)  printf("【0】 after compute row limits、nthreads、m_per_block, should into 【1】\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    if(program=="test"){
        dim3 blocks(nblocks, h);
        dim3 threads(nthreads);
        // 总共使用了共享内存存 nthreas + 2 * m_per_block 个数
        fused_forward_kernel_test<<<blocks, threads, nthreads * (sizeof(int)) + m_per_block * 2 * sizeof(float)>>>(
            m, nnz, h, f, attn_drop, attn_row, attn_col, row_ptr, col_ind,
            in_feat, negative_slope, edge_max, edge_sum, edge_mask, out_feat, seed,
            row_limits, nlimits, nthreads, m_per_block
        );        
    }
    // dgnn
    else {
        dim3 blocks(m, h);
        dim3 threads(32, (f + 31) / 32);
        fused_forward_kernel<<<blocks, threads, 32 * (sizeof(int)+sizeof(float))>>>(
            m, nnz, h, f, attn_drop, attn_row, attn_col, row_ptr, col_ind,
            in_feat, negative_slope, edge_max, edge_sum, edge_mask, out_feat, seed,
            row_limits);        
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if(DEBUG_MODE>=1)  printf("【0】 out of fused_forward_kernel_test()\n");


    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\nKernel execution time: %f milliseconds\n", milliseconds);

    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    // 等待CUDA操作完成
    cudaDeviceSynchronize();
    if(DEBUG_MODE>=1)  printf("【0】 all kernels execute end()\n");

    float *out_feat_host = (float*)malloc(sizeof(float) * m * h * f);
    cudaMemcpy(out_feat_host, out_feat, sizeof(float) * m * h * f, cudaMemcpyDeviceToHost);

    if(DEBUG_MODE >= 2){
        printf("out_feat[m,h,f]: (m=%d, h=%d, f=%d) \n", m, h, f);
        filename = base + "/result_" + program + "_h"+ std::to_string(h) + "_f" + std::to_string(f) +".txt";
        printf("see more in %s\n", filename.c_str());

        for(int j=0;j<16&&j<m;j++){
            printf("m=%d  ", j);
            for(int i=0;i<h*f&&i<8;i++){
                printf("%.3f ", out_feat_host[i+j*h*f]);
            }
            printf("\n");
        }
        
        if(DEBUG_MODE >= 3) saveToFile(filename, out_feat_host, m, h, f);
    }


    free(row_limits_host);
    free(out_feat_host);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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

    return 0;
}

// nvcc -g -G -o test gatconv_main_fixed.cu
//  cuda-memcheck ./test 2>&1 | tee test_output.log