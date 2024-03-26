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
        // printf("i=%d   left=%d\n", i, left);
        
        int nnz = row_ptr[right] - row_ptr[left];
        if(nnz > max_nnzPerBlock) max_nnzPerBlock = nnz;
    }

    nthreads = max_nnzPerBlock;
    m_per_block = max_mPerBlock;

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

// compute res cpu
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

        // printf("row_indices:\n");
        // for(int i=0;i<row_indices.size();i++)
        //     if(row_indices[i] < 51)std::cout << row_indices[i] << "(i=" << i+15<<")" << " ";
        // std::cout<<std::endl<<std::endl;

        
        // printf("col_indices:\n");
        // for(int i=0;i<50;i++)
        //     std::cout << col_indices[i] << " ";
        // std::cout<<std::endl<<std::endl;


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

        // std::cout << "\n\nrow_ptr(总偏移量) contains: ";
        // for(int i = 0; i < row_ptr.size(); i++) {
        //     if(i%10 == 0) std::cout << std::endl << "【" << i << "】";
        //     std::cout << row_ptr[i] << " ";
        // }
        // std::cout << std::endl;

        // printf("sorted_row_indices:\n");
        // for(int i=0;i<50;i++)
        //     std::cout << sorted_row_indices[i] << " ";
        // std::cout<<std::endl<<std::endl;


        // printf("row_ptr:\n");
        // for(int i=0;i<50;i++)
        //     std::cout << row_ptr[i] << " ";
        // std::cout<<std::endl<<std::endl;

        
        // printf("sorted_col_indices:\n");
        // for(int i=0;i<50;i++)
        //     std::cout << col_indices[i] << " ";
        // std::cout<<std::endl<<std::endl;


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

void compareResults(const std::string& file1, const std::string& file2, int& numDif, int& difPos) {

    numDif = 0;

    std::ifstream infile1(file1);
    std::ifstream infile2(file2);

    if (!infile1.is_open() ) {
        std::cerr << "Error opening file_test: " << file1 << std::endl;
        return ;
    }
    if (!infile2.is_open() ) {
        std::cerr << "Error opening file_dgnn: " << file2 << std::endl;
        return ;
    }

    std::vector<float> data1, data2;
    float num;

    while (infile1 >> num) {
        data1.push_back(num);
    }

    while (infile2 >> num) {
        data2.push_back(num);
    }

    infile1.close();
    infile2.close();

    if (data1.size() != data2.size()) {
        return ; 
    }

    for (size_t i = 0; i < data1.size(); ++i) {
        // if (std::abs(data1[i] - data2[i]) > std::numeric_limits<float>::epsilon()) {
        if (std::abs(data1[i] - data2[i]) > 1e-4) {
            if(numDif == 0) difPos = i;
            numDif++;
        }
    }

    return ; 
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

    float out_feat[m * h * f] = {0};

    for(int hid = 0; hid < h; ++hid){
        for(int rid = 0; rid < row_ptr.size()-1; ++rid){

            float expAll = 0, weightMax = -1e38;
            float attn_row_val = attn_row[rid * h + hid];
            if(rid==testRid) {printf("testRid=%d, 查看过程中间各结果\n",testRid );}
            if(rid==testRid) printf("attn_row_val=%f\n", attn_row_val);

            for(int i = row_ptr[rid]; i < row_ptr[rid+1]; ++i){
                int cid = col_ind[i];
                float w = attn_row_val + attn_col[cid * h + hid];
                if(rid==testRid) printf("attn_col_val[%d]=%f\n", cid,  attn_col[cid * h + hid]);
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
                    if(rid==testRid&&fid==0) printf("cid=%d【acc+=w*in_feat】[%f=%f*%f] \n\n",cid, acc, w, in_feat[cid * h * f + hid * f + fid]);
                }
            }

        }
    }
    saveToFile(out_feat_file, out_feat, m, h, f);

}


// int main()
// {
//     // nvcc -o test utils.cpp

//     bool newFeatMtx = false;
//     if(newFeatMtx){
//         int m;
//         int h = 1;
//         int f = 4;

//         // attn_row[m, h]
//         std::string file = "m1138_nnz2596";
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

//     bool computeResCPU = true;
//     if(computeResCPU){
//         std::string file = "m1138_nnz2596";
//         std::string base = "./matrix/" + file;

//         std::string csr_file = base + "/csr_" + file + ".mtx";

//         int m = 1138;
//         int h = 1;
//         int f = 4;
//         std::string attn_row_file = base + "/attn_row_m" + std::to_string(m) + "_h" + std::to_string(h) + ".txt";
//         std::string attn_col_file = base + "/attn_col_m" + std::to_string(m) + "_h" + std::to_string(h) + ".txt";
//         std::string in_feat_file = base + "/in_feat_m" + std::to_string(m) + "_h" + std::to_string(h) + "_f" + std::to_string(f) + ".txt";

//         std::string out_feat_file = base + "/result_CPU_h" + std::to_string(h) + "_f" + std::to_string(f) + ".txt";
        
//         float attn_drop = 0.1, negative_slope = 0.01;
//         computeResultCPU(attn_row_file, attn_col_file, in_feat_file, csr_file, 
//         out_feat_file, h, f, negative_slope, attn_drop);
//     }
    
    
//     bool compare = true;
//     if(compare){
//         int h = 1, f = 4;
//         std::string file = "m1138_nnz2596";
//         std::string base = "./matrix/" + file;
//         std::string file1 = base + "/result_test_h"+ std::to_string(h) + "_f" + std::to_string(f) +".txt";
//         std::string file2 = base + "/result_dgnn_h"+ std::to_string(h) + "_f" + std::to_string(f) +".txt";
//         std::string file3 = base + "/result_CPU_h"+ std::to_string(h) + "_f" + std::to_string(f) +".txt";

//         int numDif = 0, difPos=0;
//         // file1 放test， file2放dgnn
//         compareResults(file1, file3, numDif, difPos);
//         if ( numDif== 0) {
//             std::cout << "\ntest res == CPU res" << std::endl;
//         } else {
//             std::cout << "【test&CPU】num of diffrent rows["<<difPos<<"] = " << numDif << std::endl;
//         }

//         compareResults(file2, file3, numDif, difPos);
//         if ( numDif== 0) {
//             std::cout << "dgnn res == CPU res" << std::endl;
//         } else {
//             std::cout << "【dgnn&CPU】num of diffrent rows["<<difPos<<"] = " << numDif << std::endl;
//         }

//         compareResults(file1, file2, numDif, difPos);
//         if ( numDif== 0) {
//             std::cout << "test res == dgnn res" << std::endl;
//         } else {
//             std::cout << "【test&dgnn】num of diffrent rows["<<difPos+1<<"] = " << numDif << std::endl;
//         }
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