#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <numeric>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define LeakyRelu(x, negative_slope) ((x > 0) ? (x) : ((x)*negative_slope))
#define testRid -2
// #define rowptr_file "./matrix/m352w_nnz1919w/csr_rowptr.txt"
#define rowptr_file ""
//  ./test 2>&1 | tee output_cpu.log

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

        size_t freeMemory = 0;
        size_t totalMemory = 0;
        cudaError_t status = cudaMemGetInfo(&freeMemory, &totalMemory);
        std::cout << "  Free memory: " << freeMemory / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Total memory: " << totalMemory / (1024.0 * 1024.0) << " MB" << std::endl;

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

int getLimitsParams(const int& max_nnzPerRow, const int& overload_threshold, const int& nnz, int& NNZ_PER_BLOCK, int& nblocks){
    // if(DEBUG_MODE>=2) printf("【0a】into get row limits params()\n");
    
    int tmp_max_nnzPerRow = max_nnzPerRow;
    if (tmp_max_nnzPerRow <= 0) {
        std::cout << "计算 NNZ_PER_BLOCK 失败， max_nnzPerRow<=0" << std::endl;
        return -1;
    }
    if(tmp_max_nnzPerRow > overload_threshold) {
        return overload_threshold;
    }

    // power: >= tmp_max_nnzPerRow 的最小的 2次幂
    if ((tmp_max_nnzPerRow & (tmp_max_nnzPerRow - 1)) == 0) {
        NNZ_PER_BLOCK = tmp_max_nnzPerRow;
    }else{
        NNZ_PER_BLOCK = 1;
        while (tmp_max_nnzPerRow > 0) {
            tmp_max_nnzPerRow >>= 1;
            NNZ_PER_BLOCK <<= 1;
        }        
    }

    nblocks = (nnz + NNZ_PER_BLOCK - 1) / NNZ_PER_BLOCK; 

    return 0;
}

int getForwardParams(const int* row_limits, const int& nlimits, const std::vector<int>& row_ptr, 
  int& nthreads, int& m_per_block){

    int max_nnzPerBlock = 0;
    int max_mPerBlock = 0;
    for(int i = 0; i < nlimits - 1; ++i){
        int left = row_limits[i];
        int right = row_limits[i+1]; 
        if(right - left > max_mPerBlock) max_mPerBlock = right - left;
        
        int nnz = row_ptr[right] - row_ptr[left];
        if(nnz > max_nnzPerBlock) max_nnzPerBlock = nnz;
    }

    // power: >= max_nnzPerBlock 的最小的 2次幂
    // if ((max_nnzPerBlock & (max_nnzPerBlock - 1)) == 0) {
    //     nthreads = max_nnzPerBlock;
    // }else{
    //     nthreads = 1;
    //     while (max_nnzPerBlock > 0) {
    //         max_nnzPerBlock >>= 1;
    //         nthreads <<= 1;
    //     }        
    // }
    nthreads = max_nnzPerBlock;

    m_per_block = max_mPerBlock;
    

    if(rowptr_file != "" ){
        std::ofstream file(rowptr_file, std::ios::app);
        if (!file.is_open()) {
            std::cerr << "【getForwardParams】 to open file: " << rowptr_file << std::endl;
        }

        file << "\n\nrow_limits:\n";
        for(int i = 0; i < nlimits - 1 && i < 1000; i++){
            if(i % 10 == 0){
                file << "\n【bid=" << i <<  "】";
            }
            file << row_limits[i] << " ";
        }

    }
    return 0;
}


// compue rowlimits cpu
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
            if(mtxMode == "pattern"){

            }else if(mtxMode=="real") {
                file >> data;
            }else if(mtxMode=="complex"){
                file >> data >> data;
            }else{
                printf("wrong mtxMode:%s\n", mtxMode.c_str());
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

        std::vector<int> sorted_indices(row_indices.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0); 
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                [&row_indices](int i1, int i2) {
                    return row_indices[i1] < row_indices[i2];
                });

        // 用排序后的索引重排row_indices和col_indices
        std::vector<int> sorted_row_indices(row_indices.size());
        std::vector<int> sorted_col_indices(col_indices.size());
        for (size_t i = 0; i < sorted_indices.size(); ++i) {
            sorted_row_indices[i] = row_indices[sorted_indices[i]];
            sorted_col_indices[i] = col_indices[sorted_indices[i]];
        }
        col_indices = sorted_col_indices;
        
        for (int i = 0; i < numRows; i++) {
            row_ptr[i + 1] += row_ptr[i];
        }


        return 0;
    } else {
        return -1; 
    }
}

void saveRowptrToFile(const std::string& filename, const std::vector<int>& row_ptr,  const std::vector<int>& col_ind) {
    std::ofstream file(filename, std::ios::app); // 打开文件

    if (!file.is_open()) {
        std::cerr << "【saveRowptrToFile】Failed to open file: " << filename << std::endl;
        return;
    }

    file << "\n\norigin file:\n";
    file << filename << std::endl;

    file << "\nrow_ptr:\n";
    for (int i = 0; i < row_ptr.size() && i < 1000; i++) {
        if(i % 10 == 0){
            file << "\n【rid=" << i << "】";
        }
        file << row_ptr[i] << " ";
    }
    file << std::endl;

    file << "\ncol_ind:\n";
    for(int i = 0; i < col_ind.size() && i < 2000; i++){
        if(i % 10 == 0){
            file << "\n【nnz=" << i << "】";
        }
        file << col_ind[i] << " ";
    }
    file << std::endl;

    file.close(); // 关闭文件
}

// compute res cpu + save row_ptr
int readCSR(const std::string& filename, int& numRows, int& numEntries,
    std::vector<int>& row_ptr, std::vector<int>& col_indices) 
{

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return -1;
    }

    std::string line;
    std::string mtxMode;
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string word;
        int wordCount = 0;
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
        // .mtx的行列从1开始
        for (int i = 0; i < numEntries; ++i) {
            int row, col;
            float data;
            file >> row >> col;
            if(mtxMode=="pattern"){
            }else if(mtxMode=="real") {
                file >> data;
            }else if(mtxMode=="complex"){
                file >> data >> data;
            }else{
                printf("csr read error: new data type\n");
                return -1;
            }
            row_indices.push_back(row - 1);
            col_indices.push_back(col - 1);
            row_ptr[row]++;
        }

        file.close();

        std::vector<int> sorted_indices(row_indices.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0); 
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                [&row_indices](int i1, int i2) {
                    return row_indices[i1] < row_indices[i2];
                });

        // 用排序后的索引重排row_indices和col_indices
        std::vector<int> sorted_row_indices(row_indices.size());
        std::vector<int> sorted_col_indices(col_indices.size());
        for (size_t i = 0; i < sorted_indices.size(); ++i) {
            sorted_row_indices[i] = row_indices[sorted_indices[i]];
            sorted_col_indices[i] = col_indices[sorted_indices[i]];

            // if(i<100)printf("mtx 第 %d 行： row=%d  col=%d\n", sorted_indices[i]+15, sorted_row_indices[i]+1, sorted_col_indices[i]+1);
        }

        col_indices = sorted_col_indices;

        for (int i = 0; i < numRows; i++) {
            row_ptr[i + 1] += row_ptr[i];
        }

        if(rowptr_file != "")
        saveRowptrToFile(rowptr_file, row_ptr, col_indices);

        return 0;
    } else {
        return -1; 
    }
}

// kernel
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
        // .mtx的行列从1开始
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
    
            // if(i<32) std::cout<<"i="<<i<<"  数据["<<row<<", " << col << "] 新增后row_ptr["<<row<<"]=" <<row_ptr[row]<<endl;
        }

        // std::cout << "\n\nrow_ptr(每行个数) contains: ";
        // for(int i = 0; i < row_ptr.size(); ++i) {
        //     std::cout << "【" << i << "】" << row_ptr[i] << " ";
        //     if(i%10 == 0) std::cout << std::endl;
        // }
        // std::cout << std::endl;

        file.close();

        std::vector<int> sorted_indices(row_indices.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0); 
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                [&row_indices](int i1, int i2) {
                    return row_indices[i1] < row_indices[i2];
                });

        // 用排序后的索引重排row_indices和col_indices
        std::vector<int> sorted_row_indices(row_indices.size());
        std::vector<int> sorted_col_indices(col_indices.size());
        for (size_t i = 0; i < sorted_indices.size(); ++i) {
            sorted_row_indices[i] = row_indices[sorted_indices[i]];
            sorted_col_indices[i] = col_indices[sorted_indices[i]];
        }
        col_indices = sorted_col_indices;

        // 此时的row_ptr存储存储了每行有多少个nnz，行id的起始位置为1
        max_nnzPerRow = 0;
        for(int i = 1; i <= numRows; i++) {
            if(row_ptr[i] > max_nnzPerRow) {
                max_nnzPerRow = row_ptr[i];
            }
        }

        for(int i = 1; i <= numRows; i++) {
            if(row_ptr[i] > overload_threshold) {
                row_overload.push_back(i-1);
            }
        }
        if(row_overload.size()==0){
            std::printf("The CSR matrix does not have more than %d(overload_threshold) rows with non-zero elements.\n", overload_threshold);
        }else{
            std::cout << "row_overload contains: ";
            for(int i = 0; i < row_overload.size(); ++i) {
                std::cout << row_ptr[row_overload[i]] << "(r=" << row_overload[i] << ")  ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        
        for (int i = 0; i < numRows; i++) {
            row_ptr[i + 1] += row_ptr[i];
        }

        // std::cout << "\n\nrow_ptr(总偏移量) contains: ";
        // for(int i = 0; i < row_ptr.size(); i++) {
        //     if(i%10 == 0) std::cout << std::endl << "【" << i << "】";
        //     std::cout << row_ptr[i] << " ";
        // }
        // std::cout << std::endl;

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

void readDenseMtx(const std::string& filename, std::vector<float>& values) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Unable to open file " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float num;
        while (iss >> num) {
            values.push_back(num);
        }
    }
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

void saveToFile(const std::string& filename, float* out_feat, int m, int h, int f) {

    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < f; ++k) {
                outfile << out_feat[i * h * f + j * f + k] << " ";
            }
            outfile << "   ";
        }
        outfile << std::endl;
    }

    outfile.close();
    std::cout << "Data saved to '" << filename << "' successfully!" << std::endl;
}

void compareResults(const std::string& file1, const std::string& file2, int h, int f) {
    std::cout << "file1: " << file1 << std::endl;
    std::cout << "file2: " << file2 << std::endl;
    std::ifstream stream1(file1);
    std::ifstream stream2(file2);

    if (!stream1.is_open() || !stream2.is_open()) {
        std::cerr << "Error opening files." << std::endl;
        return;
    }

    std::string line1, line2;
    int lineCount = 0;
    int misNum = 0;

    while (getline(stream1, line1) && getline(stream2, line2)) {
        // if(misNum > 5 ) break;
        std::istringstream iss1(line1);
        std::istringstream iss2(line2);

        bool lineMismatch = false;
        float num1, num2;

        for (int i = 0; i < h * f; ++i) {
            if (!(iss1 >> num1) || !(iss2 >> num2)) {
                std::cerr << "Error reading numbers." << std::endl;
                return;
            }

            if (num1 - num2 > 1e-4) {
                lineMismatch = true;
                break;
            }
        }

        if (lineMismatch) {
            misNum++;
            if(misNum <= 5) std::cout << "rid " << lineCount  << ": " << num1 << "  " << num2 << std::endl;
        }

        ++lineCount;
    }

    if(misNum==0) std::cout << "same!\n";
    else std::cout << "misNum=" << misNum << std::endl;
}


void computeResultCPU(const std::string& attn_row_file, const std::string& attn_col_file, 
    const std::string& in_feat_file, const std::string& csr_file, const std::string& out_feat_file,
    const int& h, const int& f, const float& negative_slope, const float& attn_drop){

    int m = 0, nnz = 0; 
    std::vector<int> row_ptr, col_ind;
    readCSR(csr_file, m, nnz, row_ptr, col_ind);

    std::vector<float> attn_row, attn_col, in_feat;
    readDenseMtx(attn_row_file, attn_row);
    readDenseMtx(attn_col_file, attn_col);
    readDenseMtx(in_feat_file, in_feat);

    int size = m * h * f;
    float* out_feat = new float[size];

    if(testRid==-2){
        for(int hid = 0; hid < h; ++hid){
            for(int rid = 0; rid < row_ptr.size()-1; ++rid){

                float expAll = 0, weightMax = -1e38;
                float attn_row_val = attn_row[rid * h + hid];
                if(rid==testRid) {printf("testRid=%d, 查看过程中间各结果\n",testRid );}
                if(rid==testRid) printf("attn_row_val=%f\n", attn_row_val);

                for(int i = row_ptr[rid]; i < row_ptr[rid+1]; ++i){
                    int cid = col_ind[i];
                    float w = attn_row_val + attn_col[cid * h + hid];
                    if(rid==testRid) printf("attn_col_val[%d]=%f\n", cid, attn_col[cid * h + hid]);
                    w = LeakyRelu(w, negative_slope);
                    
                    weightMax = std::max(w, weightMax);
                }
                if(rid==testRid) printf("weightMax=%f\n", weightMax);

                for(int i = row_ptr[rid]; i < row_ptr[rid+1]; ++i){
                    int cid = col_ind[i];
                    float w = LeakyRelu(attn_row_val + attn_col[cid * h + hid], negative_slope);
                    
                    float exp = std::exp(w - weightMax);
                    // if(rid==testRid) printf("cid=%d:  w=%f  weightMax=%f  exp=%f\n", cid, w, weightMax[hid], exp);
                    expAll += exp;
                }
                if(rid==testRid) printf("expAll=%f\n", expAll);

                for(int fid = 0; fid < f; ++fid){
                    float acc = 0;
                    for(int i = row_ptr[rid]; i < row_ptr[rid+1]; ++i){
                        int cid = col_ind[i];
                        float w = attn_row_val + attn_col[cid * h + hid];
                        w = LeakyRelu(w, negative_slope);
                        
                        w = std::exp(w - weightMax)/expAll;
                        w = w / (1 - attn_drop);

                        // if(mask>attn_drop)
                        acc += w * in_feat[cid * h * f + hid * f + fid];
                        out_feat[rid * h * f + hid * f + fid] = acc;
                        if(rid==testRid) printf("cid=%d fid=%d【acc+=w*in_feat】[%f=%f*%f] \n", cid, fid, acc, w, in_feat[cid * h * f + hid * f + fid]);
                    }
                    if(rid==testRid) printf("\n");
                }

            }
        }
        saveToFile(out_feat_file, out_feat, m, h, f);        
    }
    else{
        for(int hid = 0; hid < h; ++hid){
        // for(int rid = 0; rid < row_ptr.size()-1; ++rid){
            int rid = testRid;
            float expAll = 0, weightMax = -1e38;
            float attn_row_val = attn_row[rid * h + hid];
            if(rid==testRid) {printf("查看过程中间各结果\n");}
            if(rid==testRid) printf("testRid=%d, attn_row_val=%f\n", rid, attn_row_val);

            for(int i = row_ptr[rid]; i < row_ptr[rid+1]; ++i){
                int cid = col_ind[i];
                float w = attn_row_val + attn_col[cid * h + hid];
                if(rid==testRid) printf("attn_col_val[%d]=%f ", cid,  attn_col[cid * h + hid]);
                w = LeakyRelu(w, negative_slope);

                weightMax = std::max(w, weightMax);
            }

            if(rid==testRid) printf("\n\nweightMax=%f\n", weightMax);

            for(int i = row_ptr[rid]; i < row_ptr[rid+1]; ++i){
                int cid = col_ind[i];
                float w = LeakyRelu(attn_row_val + attn_col[cid * h + hid], negative_slope);
                
                float exp = std::exp(w - weightMax);
                // if(rid==testRid) printf("cid=%d:  w=%f  weightMax=%f  exp=%f\n", cid, w, weightMax[hid], exp);
                expAll += exp;
            }
            if(rid==testRid) printf("expAll=%f\n\n", expAll);

            for(int fid = 0; fid < f; ++fid){
                if(rid==testRid) printf("fid=%d\n", fid);
                float acc = 0;
                for(int i = row_ptr[rid]; i < row_ptr[rid+1]; ++i){
                    int cid = col_ind[i];
                    float w = attn_row_val + attn_col[cid * h + hid];
                    w = LeakyRelu(w, negative_slope);
                    
                    w = std::exp(w - weightMax)/expAll;
                    // w = w / (1 - attn_drop);

                    // if(mask>attn_drop)
                    float tmp_feat = w * in_feat[cid * h * f + hid * f + fid];
                    acc += tmp_feat;
                    out_feat[rid * h * f + hid * f + fid] = acc;
                    if(rid==testRid) printf("cid=%d  tmpf=w*in_feat [%f=%f*%f]\n", cid, tmp_feat, w, in_feat[cid * h * f + hid * f + fid]);
                }
                if(rid==testRid) printf("out_feat=%f\n\n", acc);
            }

        // }
        }
    }


    delete[] out_feat;

}

//   ./test 2>&1 | tee output_cpu.log

// int main()
// {
//     // nvcc -o test utils.cpp
//     // ./test 2>&1 | tee output_cpu.log
//     std::string file = "m352w_nnz1919w";
//     int h = 1;
//     int f = 4;

//     int m;
//     std::string base = "./matrix/" + file + "/";
//     std::string csr_file = base + "csr_" + file + ".mtx";
//     std::string mtxMode;

//     readCSR(csr_file, m, mtxMode);
//     std::printf("csr_file=%s  m=%d  mtxMode=%s  \n", csr_file.c_str(), m, mtxMode.c_str());

//     // bool newFeatMtx = false;
//     // if(newFeatMtx){
//     //     std::string tail = std::to_string(m)+"_h"+std::to_string(h);

//     //     std::string  attnrow_path = base + "attn_row_m"+ tail +".txt";
//     //     const char* file_attn_row = attnrow_path.c_str();
//     //     generateDenseMtx(m, h, 1, file_attn_row);

//     //     std::string  attncol_path = base + "attn_col_m"+ tail +".txt";
//     //     const char* file_attn_col = attncol_path.c_str();
//     //     generateDenseMtx(m, h, 1, file_attn_col);

//     //     std::string  infeat_path = base + "in_feat_m"+ tail +"_f"+std::to_string(f)+".txt";
//     //     const char* file_in_feat = infeat_path.c_str();
//     //     generateDenseMtx(m, h, f, file_in_feat);   

//     //     // std::cout << "attn_row_file: " << attnrow_path << std::endl;  
//     //     // std::cout << "attn_col_file: " << attncol_path << std::endl;  
//     //     // std::cout << "in_feat_file: " << infeat_path << std::endl;  
//     // }

//     bool computeResCPU = true;
//     if(computeResCPU){
//         std::string attn_row_file = base + "attn_row_m" + std::to_string(m) + "_h" + std::to_string(h) + ".txt";
//         std::string attn_col_file = base + "attn_col_m" + std::to_string(m) + "_h" + std::to_string(h) + ".txt";
//         std::string in_feat_file = base + "in_feat_m" + std::to_string(m) + "_h" + std::to_string(h) + "_f" + std::to_string(f) + ".txt";

//         std::string out_feat_file = base + "result_CPU_h" + std::to_string(h) + "_f" + std::to_string(f) + ".txt";
        
//         float attn_drop = 0.1, negative_slope = 0.01;
//         computeResultCPU(attn_row_file, attn_col_file, in_feat_file, csr_file, 
//           out_feat_file, h, f, negative_slope, attn_drop);
//     }
    
    
//     bool compare = true;
//     if(compare){
//         std::string file1 = base + "result_test_h"+ std::to_string(h) + "_f" + std::to_string(f) +".txt";
//         std::string file2 = base + "result_dgnn_h"+ std::to_string(h) + "_f" + std::to_string(f) +".txt";
//         std::string file3 = base + "result_CPU_h"+ std::to_string(h) + "_f" + std::to_string(f) +".txt";

//         // file1 放test， file2放dgnn
//         printf("compare test&CPU:\n");
//         compareResults(file1, file3, h, f);
//         printf("\ncompare dgnn&CPU:\n");
//         compareResults(file2, file3, h, f);
//     }

//    return 0;
// }