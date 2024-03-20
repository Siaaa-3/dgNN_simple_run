#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <ctime>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

void printCSR(int m, int nnz, int* row_ptr, int* col_ind) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < m; ++col) {
            bool found = false;
            for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
                if (col_ind[idx] == col) {
                    found = true;
                    break;
                }
            }
            printf("%d ", found ? 1 : 0);
        }
        printf("\n");
    }
}

void printDevices(){
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device Number: " << i << std::endl;
        std::cout << "  Device name: " << prop.name << std::endl;
        std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
        std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
        std::cout << "  Peak Memory Bandwidth (GB/s): " <<
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl;
        printf("  Maximum shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max thread dimensions (x, y, z): ("
                << prop.maxThreadsDim[0] << ", "
                << prop.maxThreadsDim[1] << ", "
                << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid dimensions (x, y, z): ("
                << prop.maxGridSize[0] << ", "
                << prop.maxGridSize[1] << ", "
                << prop.maxGridSize[2] << ")" << std::endl;
    }
    return ;
}

int readCSR(const std::string& filename, int& numRows, int& numEntries,
    std::vector<int>& row_ptr, std::vector<int>& col_indices, std::string& mtxMode) 
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return -1;
    }

    std::string line;
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string word;
        int wordCount = 0;

        // 遍历这一行的每个单词
        while (iss >> word) {
            wordCount++;
            if (wordCount == 4) {
                mtxMode = word;
                break;
            }
        }
    }
    
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // int numRows, numCols, numEntries;
    int numCols;
    if (sscanf(line.c_str(), "%d %d %d", &numRows, &numCols, &numEntries) == 3) {
        std::vector<int> row_indices; 

        row_ptr.clear();
        col_indices.clear();
        row_ptr.resize(numRows + 1, 0);

        for (int i = 0; i < numEntries; ++i) {
            int row, col;
            float data;
            file >> row >> col;
            if(mtxMode=="real") {
                file >> data;
            }else if(mtxMode=="complex"){
                file >> data >> data;
            }
            row_indices.push_back(row - 1);
            col_indices.push_back(col - 1);
            row_ptr[row]++;
    
            // if(i<32) std::cout<<"i="<<i<<" row["<<row<<"]"<<"="<<row_ptr[row]<<endl;
        }

        // std::cout << "\n\nrow_ptr2 contains: ";
        // for(int i = 0; i<32&&i < row_ptr.size(); ++i) {
        //     std::cout << row_ptr[i] << " ";
        // }

        file.close();
        
        for (int i = 0; i < numRows; i++) {
            row_ptr[i + 1] += row_ptr[i];
        }

        // std::cout << "\n\nrow_ptr_host3 contains: ";
        // for(int i = 0; i<32&&i < row_ptr.size(); ++i) {
        //     std::cout << row_ptr[i] << " ";
        // }

        return 0;
    } else {
        return -1; 
    }
}

int readCSR(const std::string& filename, int& numRows, int& numEntries,
     std::vector<int>& row_ptr, std::vector<int>& col_indices, std::string& mtxMode,
     std::vector<int>& row_overload, const int& overload_threshold, int& max_nnzPerRow) 
{

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return -1;
    }

    std::string line;
    // 读取第一行
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string word;
        int wordCount = 0;

        // 遍历这一行的每个单词
        while (iss >> word) {
            wordCount++;
            if (wordCount == 4) {
                mtxMode = word;
                break;
            }
        }
    }

    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // int numRows, numCols, numEntries;
    int numCols;
    if (sscanf(line.c_str(), "%d %d %d", &numRows, &numCols, &numEntries) == 3) {
        std::vector<int> row_indices; 

        row_ptr.clear();
        col_indices.clear();
        row_ptr.resize(numRows + 1, 0);

        for (int i = 0; i < numEntries; ++i) {
            int row, col;
            float data;
            file >> row >> col;
            if(mtxMode=="real") {
                file >> data;
            }else if(mtxMode=="complex"){
                file >> data >> data;
            }
            row_indices.push_back(row - 1);
            col_indices.push_back(col - 1);
            row_ptr[row]++;
    
            // if(i<32) std::cout<<"i="<<i<<" row["<<row<<"]"<<"="<<row_ptr[row]<<endl;
        }

        // std::cout << "\n\nrow_ptr2 contains: ";
        // for(int i = 0; i<32&&i < row_ptr.size(); ++i) {
        //     std::cout << row_ptr[i] << " ";
        // }

        file.close();

        max_nnzPerRow = 0;
        for(int i = 0; i < numRows; i++) {
            if(row_ptr[i] > overload_threshold) {
                row_overload.push_back(i);
            }
            if(row_ptr[i] > max_nnzPerRow) {
                max_nnzPerRow = row_ptr[i];
            }
        }

        if(row_overload.size()==0){
            printf("The CSR matrix does not have more than %d rows with non-zero elements.\n", overload_threshold);
        }else{
            std::cout << "row_overload contains: ";
            for(int i = 0; i < row_overload.size(); ++i) {
                std::cout << row_overload[i] << "(" << row_ptr[row_overload[i]] << ")  ";
            }
            std::cout << std::endl;


        }
        std::cout << std::endl;
        
        for (int i = 0; i < numRows; i++) {
            row_ptr[i + 1] += row_ptr[i];
        }

        // std::cout << "\n\nrow_ptr_host3 contains: ";
        // for(int i = 0; i<32&&i < row_ptr.size(); ++i) {
        //     std::cout << row_ptr[i] << " ";
        // }

        return 0;
    } else {
        return -1; 
    }
}

int readCSR(const std::string& filename, int& numRows, std::string& mtxMode) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return -1;
    }

    std::string line;
    // 读取第一行
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string word;
        int wordCount = 0;

        // 遍历这一行的每个单词
        while (iss >> word) {
            wordCount++;
            if (wordCount == 4) {
                mtxMode = word;
                break;
            }
        }
    }

    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // int numRows, numCols, numEntries;
    int numCols, numEntries;
    if (sscanf(line.c_str(), "%d %d %d", &numRows, &numCols, &numEntries) == 3) {
        return 0;
    } else {
        return -1; 
    }
}


void generateDenseMtx(int m, int h, int f, const char* filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件 " << filename << std::endl;
        return;
    }

    std::srand(std::time(0));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < h; ++j) {
            for(int k = 0; k < f; ++k ){
                float randomValue = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                file << randomValue;
                if (j * h + k < h * f) file << "  "; 
                if(k == (f - 1)) file << "   ";
            }
        }
        file << "\n"; 
    }

    file.close();
}

void readDenseMtx(const char* filename, std::vector<float>& values) {

    FILE* file = fopen(filename, "r");
    if (file == nullptr) {
        std::cerr << "无法打开文件 " << filename << std::endl;
        return ; // 返回空向量
    }

    // std::vector<float> values;
    float value;

    // 使用 fscanf 循环读取浮点数，直到文件末尾
    while (fscanf(file, "%f", &value) == 1) {
        values.push_back(value);
    }

    // 关闭文件
    fclose(file);

    return ;
}


void computeRowlimitsCPU(int nblocks, int NNZ_PER_BLOCK, int m, std::vector<int>& row_ptr, int* row_limits){
    row_limits[0] = 0;
    int xblock = 1;
    for(int i = 0; i < m && xblock < nblocks; ++i){
        if(row_ptr[i+1] - row_limits[xblock-1] > xblock * NNZ_PER_BLOCK){
            row_limits[xblock] = i;
            xblock++;
        }
    }
    row_limits[nblocks] = m;
}

int getNnzPerBlock(int max_nnzPerRow, int overload_threshold){
    if (max_nnzPerRow <= 0) {
        std::cout << "计算 NNZ_PER_BLOCK 失败" << std::endl;
        return -1;
    }
    if(max_nnzPerRow > overload_threshold) {
        return overload_threshold;
    }
    // 大于 max_nnzPerRow 的最小的 2次幂
    if ((max_nnzPerRow & (max_nnzPerRow - 1)) == 0) {
        return max_nnzPerRow;
    }
    int power = 1;
    while (max_nnzPerRow > 0) {
        max_nnzPerRow >>= 1;
        power <<= 1;
    }
    return power;
}

// int main()
// {
//     // nvcc -o test utils.cpp

//     bool newFeatMtx = true;
//     if(newFeatMtx){
//         int m;
//         int h = 1;
//         int f = 4;

//         // attn_row[m, h]
//         std::string file = "m352w_nnz1919w";
//         std::string base = "./matrix/" + file + "/";
//         std::string filename = base + "csr_" + file + ".mtx";
//         std::string mtxMode;
//         readCSR(filename, m, mtxMode);
//         std::printf("filename=%s  m=%d  mtxMode=%s  \n", filename.c_str(), m, mtxMode.c_str());
//         std::string tail = std::to_string(m)+"_h"+std::to_string(h);

//         std::string  attnrow_path = base + "attn_row_m"+ tail +".txt";
//         const char* file_attn_row = attnrow_path.c_str();
//         generateDenseMtx(m, h, 1, file_attn_row);

//         std::string  attncol_path = base + "attn_col_m"+ tail +".txt";
//         const char* file_attn_col = attncol_path.c_str();
//         generateDenseMtx(m, h, 1, file_attn_col);

//         std::string  infeat_path = base + "in_feat_m"+ tail +"_f"+std::to_string(f)+".txt";
//         const char* file_in_feat = infeat_path.c_str();
//         generateDenseMtx(m, h, f, file_in_feat);        
//     }


//     bool rowLimitsCPU = false;
//     if(rowLimitsCPU){
//         int nblocks = 10; 
//         int NNZ_PER_BLOCK = 5; 
//         int m = 39;
//         int nnz = 131;
//         int* row_limits = (int*)malloc(sizeof(int)*(nblocks+1));

//         std::string file = "m39_nnz131";
//         std::string base = "./matrix/" + file;
//         std::string mtxMode = "real";
//         std::vector<int> row_ptr_host, col_ind_host;
//         std::string filename = base + "/csr_" + file + ".mtx";
//         if (readCSR(filename, m, nnz, row_ptr_host, col_ind_host, mtxMode) == 0) {
//             std::cerr << "Read CSR matrix success." << std::endl;
//         } else {
//             std::cerr << "Failed to read the matrix from file." << std::endl;
//         }

//         computeRowlimitsCPU(nblocks, NNZ_PER_BLOCK, m, row_ptr_host, row_limits);

//         printf("row_limits:\n");
//         for(int i=0; i<nblocks+1; i++) printf("%d ", row_limits[i]);
//         printf("\n");
//     }



//    return 0;
// }