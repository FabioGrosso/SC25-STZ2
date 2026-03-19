#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <cfloat>
#include <algorithm> // For std::min, std::max
#include "sz.hpp"

const int full_dim_z = 1024;
const int full_dim_y = 1024;
const int full_dim_x = 1024;
const int dim_z = 512;
const int dim_y = 512;
const int dim_x = 512;
const int low_dim_z = 256;
const int low_dim_y = 256;
const int low_dim_x = 256;

void merge_sub_blocks_to_full(float* sub_blocks[8], float* full_data, int sub_dim_x, int sub_dim_y, int sub_dim_z) {
    int full_x = sub_dim_x * 2;
    int full_y = sub_dim_y * 2;
    int full_z = sub_dim_z * 2;

    #pragma omp parallel for 
    for (int z = 0; z < full_z; ++z) {
        for (int y = 0; y < full_y; ++y) {
            for (int x = 0; x < full_x; ++x) {
                int sub_index = ((z & 1) << 2) | ((y & 1) << 1) | (x & 1);
                int sub_z = z / 2;
                int sub_y = y / 2;
                int sub_x = x / 2;
                int pos = sub_z * (sub_dim_y * sub_dim_x) + sub_y * sub_dim_x + sub_x;
                full_data[z * (full_y * full_x) + y * full_x + x] = sub_blocks[sub_index][pos];
            }
        }
    }
}

double computeRange(const float* data, size_t num_elements) {
    double high = -FLT_MAX;
    double low = FLT_MAX;
    for (size_t i = 0; i < num_elements; i++) {
        if (data[i] > high) high = data[i];
        if (data[i] < low)  low = data[i];
    }
    return high - low;
}

void slice_full_data(const float* full_data, float* sub_block_data[8],int dim_x, int dim_y, int dim_z) {
    #pragma omp parallel for 
    for (int z = 0; z < dim_z; ++z) {
        for (int y = 0; y < dim_y; ++y) {
            for (int x = 0; x < dim_x; ++x) {
                int sub_index = ((z & 1) << 2) | ((y & 1) << 1) | (x & 1);
                int sub_z = z / 2;
                int sub_y = y / 2;
                int sub_x = x / 2;
                int pos = sub_z * (dim_y/2 * dim_x/2) + sub_y * dim_x/2 + sub_x;
                sub_block_data[sub_index][pos] = full_data[z * (dim_x * dim_y) + y * dim_x + x];
            }
        }
    }
}

//---------------------------------------------------------------------
// Preprocessing: Multi-directional Cubic in Truncated 3D Diamond-Cube
//---------------------------------------------------------------------
void preprocess_block(int block, const float* sub_block, float** all_de_blocks, float* diff,
                        int dim_x, int dim_y, int dim_z)
{
    int dim_xy= dim_x * dim_y;
    const float w_in = 9.0f / 16.0f;
    const float w_out = -1.0f / 16.0f;

    #pragma omp parallel for 
    for (int z = 0; z < dim_z; ++z)
    {
        for (int y = 0; y < dim_y; ++y)
        {
            for (int x = 0; x < dim_x; ++x)
            {
                int idx = z * dim_xy + y * dim_x + x;
                
                const float* ref_000 = all_de_blocks[0]; 
                const float* ref_011 = all_de_blocks[3]; 
                const float* ref_101 = all_de_blocks[5]; 
                const float* ref_110 = all_de_blocks[6]; 
                const float* ref_111 = all_de_blocks[7]; 

                float pred_sum = 0.0f;
                int dir_count = 0;
                float mean_sum = 0.0f;
                int mean_count = 0;

                switch(block)
                {
                    case 0: 
                        if (x >= 1 && x < dim_x - 2) { pred_sum += w_out * ref_000[idx - 1] + w_in * ref_000[idx] + w_in * ref_000[idx + 1] + w_out * ref_000[idx + 2]; dir_count++; }
                        mean_sum += ref_000[idx]; mean_count++; if (x + 1 < dim_x) { mean_sum += ref_000[idx + 1]; mean_count++; }
                        if (y >= 1 && y < dim_y - 1) { pred_sum += w_out * ref_011[idx - dim_x*2] + w_in * ref_011[idx - dim_x] + w_in * ref_011[idx] + w_out * ref_011[idx + dim_x]; dir_count++; }
                        mean_sum += ref_011[idx]; mean_count++; if (y - 1 >= 0) { mean_sum += ref_011[idx - dim_x]; mean_count++; }
                        if (z >= 1 && z < dim_z - 1) { pred_sum += w_out * ref_101[idx - dim_xy*2] + w_in * ref_101[idx - dim_xy] + w_in * ref_101[idx] + w_out * ref_101[idx + dim_xy]; dir_count++; }
                        mean_sum += ref_101[idx]; mean_count++; if (z - 1 >= 0) { mean_sum += ref_101[idx - dim_xy]; mean_count++; }
                        if (dir_count > 0) diff[idx] = sub_block[idx] - (pred_sum / dir_count); else diff[idx] = sub_block[idx] - (mean_sum / mean_count);
                        break;

                    case 1: 
                        if (y >= 1 && y < dim_y - 2) { pred_sum += w_out * ref_000[idx - dim_x] + w_in * ref_000[idx] + w_in * ref_000[idx + dim_x] + w_out * ref_000[idx + dim_x*2]; dir_count++; }
                        mean_sum += ref_000[idx]; mean_count++; if (y + 1 < dim_y) { mean_sum += ref_000[idx + dim_x]; mean_count++; }
                        if (x >= 1 && x < dim_x - 1) { pred_sum += w_out * ref_011[idx - 2] + w_in * ref_011[idx - 1] + w_in * ref_011[idx] + w_out * ref_011[idx + 1]; dir_count++; }
                        mean_sum += ref_011[idx]; mean_count++; if (x - 1 >= 0) { mean_sum += ref_011[idx - 1]; mean_count++; }
                        if (z >= 1 && z < dim_z - 1) { pred_sum += w_out * ref_110[idx - dim_xy*2] + w_in * ref_110[idx - dim_xy] + w_in * ref_110[idx] + w_out * ref_110[idx + dim_xy]; dir_count++; }
                        mean_sum += ref_110[idx]; mean_count++; if (z - 1 >= 0) { mean_sum += ref_110[idx - dim_xy]; mean_count++; }
                        if (dir_count > 0) diff[idx] = sub_block[idx] - (pred_sum / dir_count); else diff[idx] = sub_block[idx] - (mean_sum / mean_count);
                        break;

                    case 3: 
                        if (z >= 1 && z < dim_z - 2) { pred_sum += w_out * ref_000[idx - dim_xy] + w_in * ref_000[idx] + w_in * ref_000[idx + dim_xy] + w_out * ref_000[idx + dim_xy*2]; dir_count++; }
                        mean_sum += ref_000[idx]; mean_count++; if (z + 1 < dim_z) { mean_sum += ref_000[idx + dim_xy]; mean_count++; }
                        if (x >= 1 && x < dim_x - 1) { pred_sum += w_out * ref_101[idx - 2] + w_in * ref_101[idx - 1] + w_in * ref_101[idx] + w_out * ref_101[idx + 1]; dir_count++; }
                        mean_sum += ref_101[idx]; mean_count++; if (x - 1 >= 0) { mean_sum += ref_101[idx - 1]; mean_count++; }
                        if (y >= 1 && y < dim_y - 1) { pred_sum += w_out * ref_110[idx - dim_x*2] + w_in * ref_110[idx - dim_x] + w_in * ref_110[idx] + w_out * ref_110[idx + dim_x]; dir_count++; }
                        mean_sum += ref_110[idx]; mean_count++; if (y - 1 >= 0) { mean_sum += ref_110[idx - dim_x]; mean_count++; }
                        if (dir_count > 0) diff[idx] = sub_block[idx] - (pred_sum / dir_count); else diff[idx] = sub_block[idx] - (mean_sum / mean_count);
                        break;

                    case 2: 
                        if (x >= 1 && y >= 1 && x < dim_x - 2 && y < dim_y - 2) { pred_sum += w_out * ref_000[idx - dim_x - 1] + w_in * ref_000[idx] + w_in * ref_000[idx + dim_x + 1] + w_out * ref_000[idx + dim_x*2 + 2]; dir_count++; }
                        mean_sum += ref_000[idx]; mean_count++; if (x + 1 < dim_x && y + 1 < dim_y) { mean_sum += ref_000[idx + dim_x + 1]; mean_count++; }
                        if (x >= 1 && y >= 0 && x < dim_x - 1 && y < dim_y - 2) { pred_sum += w_out * ref_000[idx - dim_x + 2] + w_in * ref_000[idx + 1] + w_in * ref_000[idx + dim_x] + w_out * ref_000[idx + dim_x*2 - 1]; dir_count++; }
                        if (x + 1 < dim_x) { mean_sum += ref_000[idx + 1]; mean_count++; } if (y + 1 < dim_y) { mean_sum += ref_000[idx + dim_x]; mean_count++; }
                        if (z >= 1 && z < dim_z - 1) { pred_sum += w_out * ref_111[idx - dim_xy*2] + w_in * ref_111[idx - dim_xy] + w_in * ref_111[idx] + w_out * ref_111[idx + dim_xy]; dir_count++; }
                        mean_sum += ref_111[idx]; mean_count++; if (z - 1 >= 0) { mean_sum += ref_111[idx - dim_xy]; mean_count++; }
                        if (dir_count > 0) diff[idx] = sub_block[idx] - (pred_sum / dir_count); else diff[idx] = sub_block[idx] - (mean_sum / mean_count);
                        break;

                    case 4: 
                        if (x >= 1 && z >= 1 && x < dim_x - 2 && z < dim_z - 2) { pred_sum += w_out * ref_000[idx - dim_xy - 1] + w_in * ref_000[idx] + w_in * ref_000[idx + dim_xy + 1] + w_out * ref_000[idx + dim_xy*2 + 2]; dir_count++; }
                        mean_sum += ref_000[idx]; mean_count++; if (x + 1 < dim_x && z + 1 < dim_z) { mean_sum += ref_000[idx + dim_xy + 1]; mean_count++; }
                        if (x >= 1 && z >= 0 && x < dim_x - 1 && z < dim_z - 2) { pred_sum += w_out * ref_000[idx - dim_xy + 2] + w_in * ref_000[idx + 1] + w_in * ref_000[idx + dim_xy] + w_out * ref_000[idx + dim_xy*2 - 1]; dir_count++; }
                        if (x + 1 < dim_x) { mean_sum += ref_000[idx + 1]; mean_count++; } if (z + 1 < dim_z) { mean_sum += ref_000[idx + dim_xy]; mean_count++; }
                        if (y >= 1 && y < dim_y - 1) { pred_sum += w_out * ref_111[idx - dim_x*2] + w_in * ref_111[idx - dim_x] + w_in * ref_111[idx] + w_out * ref_111[idx + dim_x]; dir_count++; }
                        mean_sum += ref_111[idx]; mean_count++; if (y - 1 >= 0) { mean_sum += ref_111[idx - dim_x]; mean_count++; }
                        if (dir_count > 0) diff[idx] = sub_block[idx] - (pred_sum / dir_count); else diff[idx] = sub_block[idx] - (mean_sum / mean_count);
                        break;

                    case 5:
                        if (y >= 1 && z >= 1 && y < dim_y - 2 && z < dim_z - 2) { pred_sum += w_out * ref_000[idx - dim_xy - dim_x] + w_in * ref_000[idx] + w_in * ref_000[idx + dim_xy + dim_x] + w_out * ref_000[idx + dim_xy*2 + dim_x*2]; dir_count++; }
                        mean_sum += ref_000[idx]; mean_count++; if (y + 1 < dim_y && z + 1 < dim_z) { mean_sum += ref_000[idx + dim_xy + dim_x]; mean_count++; }
                        if (y >= 1 && z >= 0 && y < dim_y - 1 && z < dim_z - 2) { pred_sum += w_out * ref_000[idx - dim_xy + dim_x*2] + w_in * ref_000[idx + dim_x] + w_in * ref_000[idx + dim_xy] + w_out * ref_000[idx + dim_xy*2 - dim_x]; dir_count++; }
                        if (y + 1 < dim_y) { mean_sum += ref_000[idx + dim_x]; mean_count++; } if (z + 1 < dim_z) { mean_sum += ref_000[idx + dim_xy]; mean_count++; }
                        if (x >= 1 && x < dim_x - 1) { pred_sum += w_out * ref_111[idx - 2] + w_in * ref_111[idx - 1] + w_in * ref_111[idx] + w_out * ref_111[idx + 1]; dir_count++; }
                        mean_sum += ref_111[idx]; mean_count++; if (x - 1 >= 0) { mean_sum += ref_111[idx - 1]; mean_count++; }
                        if (dir_count > 0) diff[idx] = sub_block[idx] - (pred_sum / dir_count); else diff[idx] = sub_block[idx] - (mean_sum / mean_count);
                        break;

                    case 6:
                        if (x >= 1 && y >= 1 && z >= 1 && x < dim_x - 2 && y < dim_y - 2 && z < dim_z - 2) {
                            pred_sum += w_out * ref_000[idx - dim_xy - dim_x - 1] + w_in * ref_000[idx] + w_in * ref_000[idx + dim_xy + dim_x + 1] + w_out * ref_000[idx + dim_xy*2 + dim_x*2 + 2]; dir_count++;
                            pred_sum += w_out * ref_000[idx - dim_xy - dim_x + 2] + w_in * ref_000[idx + 1] + w_in * ref_000[idx + dim_xy + dim_x] + w_out * ref_000[idx + dim_xy*2 + dim_x*2 - 1]; dir_count++;
                            pred_sum += w_out * ref_000[idx - dim_xy + dim_x*2 - 1] + w_in * ref_000[idx + dim_x] + w_in * ref_000[idx + dim_xy + 1] + w_out * ref_000[idx + dim_xy*2 - dim_x + 2]; dir_count++;
                            pred_sum += w_out * ref_000[idx - dim_xy + dim_x*2 + 2] + w_in * ref_000[idx + dim_x + 1] + w_in * ref_000[idx + dim_xy] + w_out * ref_000[idx + dim_xy*2 - dim_x - 1]; dir_count++;
                        }
                        mean_sum += ref_000[idx]; mean_count++; if (x + 1 < dim_x) { mean_sum += ref_000[idx + 1]; mean_count++; }
                        if (y + 1 < dim_y) { mean_sum += ref_000[idx + dim_x]; mean_count++; } if (x + 1 < dim_x && y + 1 < dim_y) { mean_sum += ref_000[idx + dim_x + 1]; mean_count++; }
                        if (z + 1 < dim_z) { mean_sum += ref_000[idx + dim_xy]; mean_count++; if (x + 1 < dim_x) { mean_sum += ref_000[idx + dim_xy + 1]; mean_count++; }
                            if (y + 1 < dim_y) { mean_sum += ref_000[idx + dim_xy + dim_x]; mean_count++; } if (x + 1 < dim_x && y + 1 < dim_y) { mean_sum += ref_000[idx + dim_xy + dim_x + 1]; mean_count++; }
                        }
                        if (dir_count > 0) diff[idx] = sub_block[idx] - (pred_sum / dir_count); else diff[idx] = sub_block[idx] - (mean_sum / mean_count);
                        break;
                } 
            }
        }
    }
}

//---------------------------------------------------------------------
// De-preprocessing: 镜像重构
//---------------------------------------------------------------------
void depreprocess_block(int block, const float* deData, float** all_de_blocks, float* de_sub,
                          int dim_x, int dim_y, int dim_z)
{
    int dim_xy= dim_x * dim_y;
    const float w_in = 9.0f / 16.0f;
    const float w_out = -1.0f / 16.0f;

    #pragma omp parallel for 
    for (int z = 0; z < dim_z; ++z)
    {
        for (int y = 0; y < dim_y; ++y)
        {
            for (int x = 0; x < dim_x; ++x)
            {
                int idx = z * dim_xy + y * dim_x + x;
                
                const float* ref_000 = all_de_blocks[0];
                const float* ref_011 = all_de_blocks[3]; 
                const float* ref_101 = all_de_blocks[5]; 
                const float* ref_110 = all_de_blocks[6]; 
                const float* ref_111 = all_de_blocks[7]; 

                float pred_sum = 0.0f;
                int dir_count = 0;
                float mean_sum = 0.0f;
                int mean_count = 0;

                switch(block)
                {
                    case 0: 
                        if (x >= 1 && x < dim_x - 2) { pred_sum += w_out * ref_000[idx - 1] + w_in * ref_000[idx] + w_in * ref_000[idx + 1] + w_out * ref_000[idx + 2]; dir_count++; }
                        mean_sum += ref_000[idx]; mean_count++; if (x + 1 < dim_x) { mean_sum += ref_000[idx + 1]; mean_count++; }
                        if (y >= 1 && y < dim_y - 1) { pred_sum += w_out * ref_011[idx - dim_x*2] + w_in * ref_011[idx - dim_x] + w_in * ref_011[idx] + w_out * ref_011[idx + dim_x]; dir_count++; }
                        mean_sum += ref_011[idx]; mean_count++; if (y - 1 >= 0) { mean_sum += ref_011[idx - dim_x]; mean_count++; }
                        if (z >= 1 && z < dim_z - 1) { pred_sum += w_out * ref_101[idx - dim_xy*2] + w_in * ref_101[idx - dim_xy] + w_in * ref_101[idx] + w_out * ref_101[idx + dim_xy]; dir_count++; }
                        mean_sum += ref_101[idx]; mean_count++; if (z - 1 >= 0) { mean_sum += ref_101[idx - dim_xy]; mean_count++; }
                        if (dir_count > 0) de_sub[idx] = deData[idx] + (pred_sum / dir_count); else de_sub[idx] = deData[idx] + (mean_sum / mean_count);
                        break;

                    case 1: 
                        if (y >= 1 && y < dim_y - 2) { pred_sum += w_out * ref_000[idx - dim_x] + w_in * ref_000[idx] + w_in * ref_000[idx + dim_x] + w_out * ref_000[idx + dim_x*2]; dir_count++; }
                        mean_sum += ref_000[idx]; mean_count++; if (y + 1 < dim_y) { mean_sum += ref_000[idx + dim_x]; mean_count++; }
                        if (x >= 1 && x < dim_x - 1) { pred_sum += w_out * ref_011[idx - 2] + w_in * ref_011[idx - 1] + w_in * ref_011[idx] + w_out * ref_011[idx + 1]; dir_count++; }
                        mean_sum += ref_011[idx]; mean_count++; if (x - 1 >= 0) { mean_sum += ref_011[idx - 1]; mean_count++; }
                        if (z >= 1 && z < dim_z - 1) { pred_sum += w_out * ref_110[idx - dim_xy*2] + w_in * ref_110[idx - dim_xy] + w_in * ref_110[idx] + w_out * ref_110[idx + dim_xy]; dir_count++; }
                        mean_sum += ref_110[idx]; mean_count++; if (z - 1 >= 0) { mean_sum += ref_110[idx - dim_xy]; mean_count++; }
                        if (dir_count > 0) de_sub[idx] = deData[idx] + (pred_sum / dir_count); else de_sub[idx] = deData[idx] + (mean_sum / mean_count);
                        break;

                    case 3:
                        if (z >= 1 && z < dim_z - 2) { pred_sum += w_out * ref_000[idx - dim_xy] + w_in * ref_000[idx] + w_in * ref_000[idx + dim_xy] + w_out * ref_000[idx + dim_xy*2]; dir_count++; }
                        mean_sum += ref_000[idx]; mean_count++; if (z + 1 < dim_z) { mean_sum += ref_000[idx + dim_xy]; mean_count++; }
                        if (x >= 1 && x < dim_x - 1) { pred_sum += w_out * ref_101[idx - 2] + w_in * ref_101[idx - 1] + w_in * ref_101[idx] + w_out * ref_101[idx + 1]; dir_count++; }
                        mean_sum += ref_101[idx]; mean_count++; if (x - 1 >= 0) { mean_sum += ref_101[idx - 1]; mean_count++; }
                        if (y >= 1 && y < dim_y - 1) { pred_sum += w_out * ref_110[idx - dim_x*2] + w_in * ref_110[idx - dim_x] + w_in * ref_110[idx] + w_out * ref_110[idx + dim_x]; dir_count++; }
                        mean_sum += ref_110[idx]; mean_count++; if (y - 1 >= 0) { mean_sum += ref_110[idx - dim_x]; mean_count++; }
                        if (dir_count > 0) de_sub[idx] = deData[idx] + (pred_sum / dir_count); else de_sub[idx] = deData[idx] + (mean_sum / mean_count);
                        break;

                    case 2: 
                        if (x >= 1 && y >= 1 && x < dim_x - 2 && y < dim_y - 2) { pred_sum += w_out * ref_000[idx - dim_x - 1] + w_in * ref_000[idx] + w_in * ref_000[idx + dim_x + 1] + w_out * ref_000[idx + dim_x*2 + 2]; dir_count++; }
                        mean_sum += ref_000[idx]; mean_count++; if (x + 1 < dim_x && y + 1 < dim_y) { mean_sum += ref_000[idx + dim_x + 1]; mean_count++; }
                        if (x >= 1 && y >= 0 && x < dim_x - 1 && y < dim_y - 2) { pred_sum += w_out * ref_000[idx - dim_x + 2] + w_in * ref_000[idx + 1] + w_in * ref_000[idx + dim_x] + w_out * ref_000[idx + dim_x*2 - 1]; dir_count++; }
                        if (x + 1 < dim_x) { mean_sum += ref_000[idx + 1]; mean_count++; } if (y + 1 < dim_y) { mean_sum += ref_000[idx + dim_x]; mean_count++; }
                        if (z >= 1 && z < dim_z - 1) { pred_sum += w_out * ref_111[idx - dim_xy*2] + w_in * ref_111[idx - dim_xy] + w_in * ref_111[idx] + w_out * ref_111[idx + dim_xy]; dir_count++; }
                        mean_sum += ref_111[idx]; mean_count++; if (z - 1 >= 0) { mean_sum += ref_111[idx - dim_xy]; mean_count++; }
                        if (dir_count > 0) de_sub[idx] = deData[idx] + (pred_sum / dir_count); else de_sub[idx] = deData[idx] + (mean_sum / mean_count);
                        break;

                    case 4: 
                        if (x >= 1 && z >= 1 && x < dim_x - 2 && z < dim_z - 2) { pred_sum += w_out * ref_000[idx - dim_xy - 1] + w_in * ref_000[idx] + w_in * ref_000[idx + dim_xy + 1] + w_out * ref_000[idx + dim_xy*2 + 2]; dir_count++; }
                        mean_sum += ref_000[idx]; mean_count++; if (x + 1 < dim_x && z + 1 < dim_z) { mean_sum += ref_000[idx + dim_xy + 1]; mean_count++; }
                        if (x >= 1 && z >= 0 && x < dim_x - 1 && z < dim_z - 2) { pred_sum += w_out * ref_000[idx - dim_xy + 2] + w_in * ref_000[idx + 1] + w_in * ref_000[idx + dim_xy] + w_out * ref_000[idx + dim_xy*2 - 1]; dir_count++; }
                        if (x + 1 < dim_x) { mean_sum += ref_000[idx + 1]; mean_count++; } if (z + 1 < dim_z) { mean_sum += ref_000[idx + dim_xy]; mean_count++; }
                        if (y >= 1 && y < dim_y - 1) { pred_sum += w_out * ref_111[idx - dim_x*2] + w_in * ref_111[idx - dim_x] + w_in * ref_111[idx] + w_out * ref_111[idx + dim_x]; dir_count++; }
                        mean_sum += ref_111[idx]; mean_count++; if (y - 1 >= 0) { mean_sum += ref_111[idx - dim_x]; mean_count++; }
                        if (dir_count > 0) de_sub[idx] = deData[idx] + (pred_sum / dir_count); else de_sub[idx] = deData[idx] + (mean_sum / mean_count);
                        break;

                    case 5:
                        if (y >= 1 && z >= 1 && y < dim_y - 2 && z < dim_z - 2) { pred_sum += w_out * ref_000[idx - dim_xy - dim_x] + w_in * ref_000[idx] + w_in * ref_000[idx + dim_xy + dim_x] + w_out * ref_000[idx + dim_xy*2 + dim_x*2]; dir_count++; }
                        mean_sum += ref_000[idx]; mean_count++; if (y + 1 < dim_y && z + 1 < dim_z) { mean_sum += ref_000[idx + dim_xy + dim_x]; mean_count++; }
                        if (y >= 1 && z >= 0 && y < dim_y - 1 && z < dim_z - 2) { pred_sum += w_out * ref_000[idx - dim_xy + dim_x*2] + w_in * ref_000[idx + dim_x] + w_in * ref_000[idx + dim_xy] + w_out * ref_000[idx + dim_xy*2 - dim_x]; dir_count++; }
                        if (y + 1 < dim_y) { mean_sum += ref_000[idx + dim_x]; mean_count++; } if (z + 1 < dim_z) { mean_sum += ref_000[idx + dim_xy]; mean_count++; }
                        if (x >= 1 && x < dim_x - 1) { pred_sum += w_out * ref_111[idx - 2] + w_in * ref_111[idx - 1] + w_in * ref_111[idx] + w_out * ref_111[idx + 1]; dir_count++; }
                        mean_sum += ref_111[idx]; mean_count++; if (x - 1 >= 0) { mean_sum += ref_111[idx - 1]; mean_count++; }
                        if (dir_count > 0) de_sub[idx] = deData[idx] + (pred_sum / dir_count); else de_sub[idx] = deData[idx] + (mean_sum / mean_count);
                        break;

                    case 6:
                        if (x >= 1 && y >= 1 && z >= 1 && x < dim_x - 2 && y < dim_y - 2 && z < dim_z - 2) {
                            pred_sum += w_out * ref_000[idx - dim_xy - dim_x - 1] + w_in * ref_000[idx] + w_in * ref_000[idx + dim_xy + dim_x + 1] + w_out * ref_000[idx + dim_xy*2 + dim_x*2 + 2]; dir_count++;
                            pred_sum += w_out * ref_000[idx - dim_xy - dim_x + 2] + w_in * ref_000[idx + 1] + w_in * ref_000[idx + dim_xy + dim_x] + w_out * ref_000[idx + dim_xy*2 + dim_x*2 - 1]; dir_count++;
                            pred_sum += w_out * ref_000[idx - dim_xy + dim_x*2 - 1] + w_in * ref_000[idx + dim_x] + w_in * ref_000[idx + dim_xy + 1] + w_out * ref_000[idx + dim_xy*2 - dim_x + 2]; dir_count++;
                            pred_sum += w_out * ref_000[idx - dim_xy + dim_x*2 + 2] + w_in * ref_000[idx + dim_x + 1] + w_in * ref_000[idx + dim_xy] + w_out * ref_000[idx + dim_xy*2 - dim_x - 1]; dir_count++;
                        }
                        mean_sum += ref_000[idx]; mean_count++; if (x + 1 < dim_x) { mean_sum += ref_000[idx + 1]; mean_count++; }
                        if (y + 1 < dim_y) { mean_sum += ref_000[idx + dim_x]; mean_count++; } if (x + 1 < dim_x && y + 1 < dim_y) { mean_sum += ref_000[idx + dim_x + 1]; mean_count++; }
                        if (z + 1 < dim_z) { mean_sum += ref_000[idx + dim_xy]; mean_count++; if (x + 1 < dim_x) { mean_sum += ref_000[idx + dim_xy + 1]; mean_count++; }
                            if (y + 1 < dim_y) { mean_sum += ref_000[idx + dim_xy + dim_x]; mean_count++; } if (x + 1 < dim_x && y + 1 < dim_y) { mean_sum += ref_000[idx + dim_xy + dim_x + 1]; mean_count++; }
                        }
                        if (dir_count > 0) de_sub[idx] = deData[idx] + (pred_sum / dir_count); else de_sub[idx] = deData[idx] + (mean_sum / mean_count);
                        break;
                }
            }
        }
    }
}

//---------------------------------------------------------------------
// SZ compression/decompression and file I/O routines (unchanged)
//---------------------------------------------------------------------
char* SZ_compress(float* oriData, size_t blksize_x, size_t blksize_y, size_t blksize_z, double eb, size_t& outSize)
{
    SZ3::Config conf(blksize_z, blksize_y, blksize_x);
    conf.cmprAlgo = SZ3::ALGO_NOPRED;
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = eb;
    char* compressedData = SZ_compress<float>(conf, oriData, outSize);
    return compressedData;
}

char* SZ_compress4De(float* oriData, size_t blksize_x, size_t blksize_y, size_t blksize_z, double eb, size_t& outSize)
{
    SZ3::Config conf(blksize_z, blksize_y, blksize_x);
    conf.cmprAlgo = SZ3::ALGO_INTERP;
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = eb;
    char* compressedData = SZ_compress<float>(conf, oriData, outSize);
    return compressedData;
}

float* SZ_decompress4De(char* compressedData, size_t outSize, size_t blksize_x, size_t blksize_y, size_t blksize_z)
{
    SZ3::Config conf(blksize_z, blksize_y, blksize_x);
    conf.cmprAlgo = SZ3::ALGO_INTERP;
    conf.errorBoundMode = SZ3::EB_ABS;
    float* deData = new float[blksize_x * blksize_y * blksize_z];
    SZ_decompress<float>(conf, compressedData, outSize, deData);
    return deData;
}

float* SZ_decompress_separated(char* compressedData, size_t outSize, size_t blksize_x, size_t blksize_y, size_t blksize_z)
{
    SZ3::Config conf(blksize_z, blksize_y, blksize_x);
    conf.cmprAlgo = SZ3::ALGO_NOPRED;
    conf.errorBoundMode = SZ3::EB_ABS;
    float* deData = new float[blksize_x * blksize_y * blksize_z];
    SZ_decompress<float>(conf, compressedData, outSize, deData);
    return deData;
}

bool readBinaryData(const std::string& filepath, float* data, size_t dataSize)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return false;
    file.read(reinterpret_cast<char*>(data), dataSize * sizeof(float));
    file.close();
    return true;
}

//---------------------------------------------------------------------
// Main routine
//---------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <error_bound>" << std::endl;
        return 1;
    }

    std::string full_file_path = "/N/u/daocwang/BigRed200/data/miranda_1024x1024x1024_float32.raw";
    size_t full_size = full_dim_z * full_dim_y * full_dim_x;
    float* full_data = new float[full_size];
    if (!readBinaryData(full_file_path, full_data, full_size)) {
        delete[] full_data; return 1;
    }

    auto split_start = std::chrono::high_resolution_clock::now();
    float* sub_block_data[8];
    for (int i = 0; i < 8; ++i) sub_block_data[i] = new float[dim_z * dim_x * dim_y];
    slice_full_data(full_data, sub_block_data, full_dim_x, full_dim_y, full_dim_z);

    float* low_block_data[8];
    for (int i = 0; i < 8; ++i) low_block_data[i] = new float[low_dim_z * low_dim_y * low_dim_x];
    slice_full_data(sub_block_data[0], low_block_data, dim_x, dim_y, dim_z);

    auto split_end = std::chrono::high_resolution_clock::now();
    double sz_time_taken_split = std::chrono::duration_cast<std::chrono::nanoseconds>(split_end - split_start).count() * 1e-9;

    double eb_base = atof(argv[1]);
    double a = atof(argv[2]);
    size_t allSize = 0;
    
    // --- Step 0: Compress & Decompress base ref (000) ---
    size_t szcompressedSize;
    auto sz_start = std::chrono::high_resolution_clock::now();
    char* tmp = SZ_compress4De(low_block_data[0], low_dim_x, low_dim_y, low_dim_z, eb_base, szcompressedSize);
    allSize += szcompressedSize;
    auto sz_end = std::chrono::high_resolution_clock::now();
    double sz_time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(sz_end - sz_start).count() * 1e-9;

    auto sz_destart = std::chrono::high_resolution_clock::now();
    float* decompressed_data = SZ_decompress4De(tmp, szcompressedSize, low_dim_x, low_dim_y, low_dim_z);
    auto sz_deend = std::chrono::high_resolution_clock::now();
    double sz_time_taken_decompress = std::chrono::duration_cast<std::chrono::nanoseconds>(sz_deend - sz_destart).count() * 1e-9;

    // --- Allocate buffers for Low Level ---
    char* low_comp[7];
    float* low_diff_data[7];
    float* low_deData[7];
    float* low_de_sub_block[7];
    size_t low_compressedSize[7];
    for (int i = 0; i < 7; ++i) {
        low_diff_data[i]    = new float[low_dim_z * low_dim_y * low_dim_x];
        low_deData[i]       = new float[low_dim_z * low_dim_y * low_dim_x];
        low_de_sub_block[i] = new float[low_dim_z * low_dim_y * low_dim_x];
    }
    
    // 构建全局指针映射数组
    float* current_low_de_blocks[8];
    current_low_de_blocks[0] = decompressed_data; 
    for (int i = 0; i < 7; ++i) {
        current_low_de_blocks[i+1] = low_de_sub_block[i]; 
    }

    double low_time_taken = 0, low_time_taken_decompress_sz = 0, low_time_taken_decompress = 0;

    auto process_low_stage = [&](const std::vector<int>& blocks, double current_eb) {
        auto t1 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for 
        for (size_t i = 0; i < blocks.size(); ++i) {
            int block = blocks[i];
            preprocess_block(block, low_block_data[block+1], current_low_de_blocks, low_diff_data[block], low_dim_x, low_dim_y, low_dim_z);
            low_comp[block] = SZ_compress(low_diff_data[block], low_dim_x, low_dim_y, low_dim_z, current_eb, low_compressedSize[block]);
            #pragma omp atomic
            allSize += low_compressedSize[block];
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        low_time_taken += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() * 1e-9;

        auto t3 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for 
        for (size_t i = 0; i < blocks.size(); ++i) {
            int block = blocks[i];
            low_deData[block] = SZ_decompress_separated(low_comp[block], low_compressedSize[block], low_dim_x, low_dim_y, low_dim_z);
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        low_time_taken_decompress_sz += std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() * 1e-9;

        auto t5 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for 
        for (size_t i = 0; i < blocks.size(); ++i) {
            int block = blocks[i];
            depreprocess_block(block, low_deData[block], current_low_de_blocks, low_de_sub_block[block], low_dim_x, low_dim_y, low_dim_z);
        }
        auto t6 = std::chrono::high_resolution_clock::now();
        low_time_taken_decompress += std::chrono::duration_cast<std::chrono::nanoseconds>(t6 - t5).count() * 1e-9;
    };

    // Low Level 串行调度
    double eb_stage1 = eb_base * a;
    process_low_stage({6}, eb_stage1);       // Stage 1: Body Center
    
    double eb_stage2 = eb_stage1 * a;
    process_low_stage({2, 4, 5}, eb_stage2); // Stage 2: Face Centers
    
    double eb_stage3 = eb_stage2 * a;
    process_low_stage({0, 1, 3}, eb_stage3); // Stage 3: Edge Centers

    auto reconstructed_low_start = std::chrono::high_resolution_clock::now();
    float* reconstructed_sub_0 = new float[dim_z * dim_x * dim_y];
    merge_sub_blocks_to_full(current_low_de_blocks, reconstructed_sub_0, low_dim_x, low_dim_y, low_dim_z);
    auto reconstructed_low_end = std::chrono::high_resolution_clock::now();
    double time_taken_reconstructed_low = std::chrono::duration_cast<std::chrono::nanoseconds>(reconstructed_low_end - reconstructed_low_start).count() * 1e-9;
    
    // --- Allocate buffers for High Level ---
    char* comp[7];
    float* diff_data[7];
    float* deData[7];
    float* de_sub_block[7];
    size_t compressedSize[7];
    
    for (int i = 0; i < 7; ++i) {
        diff_data[i]    = new float[dim_z * dim_x * dim_y];
        deData[i]       = new float[dim_z * dim_x * dim_y];
        de_sub_block[i] = new float[dim_z * dim_x * dim_y];
    }

    // High level 全局指针映射数组
    float* current_high_de_blocks[8];
    current_high_de_blocks[0] = reconstructed_sub_0; 
    for (int i = 0; i < 7; ++i) {
        current_high_de_blocks[i+1] = de_sub_block[i];
    }

    double time_taken = 0, time_taken_decompress_sz = 0, time_taken_decompress = 0;

    auto process_high_stage = [&](const std::vector<int>& blocks, double current_eb) {
        auto t1 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for 
        for (size_t i = 0; i < blocks.size(); ++i) {
            int block = blocks[i];
            preprocess_block(block, sub_block_data[block+1], current_high_de_blocks, diff_data[block], dim_x, dim_y, dim_z);
            comp[block] = SZ_compress(diff_data[block], dim_x, dim_y, dim_z, current_eb, compressedSize[block]);
            #pragma omp atomic
            allSize += compressedSize[block];
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        time_taken += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() * 1e-9;

        auto t3 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for 
        for (size_t i = 0; i < blocks.size(); ++i) {
            int block = blocks[i];
            deData[block] = SZ_decompress_separated(comp[block], compressedSize[block], dim_x, dim_y, dim_z);
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        time_taken_decompress_sz += std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() * 1e-9;

        auto t5 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for 
        for (size_t i = 0; i < blocks.size(); ++i) {
            int block = blocks[i];
            depreprocess_block(block, deData[block], current_high_de_blocks, de_sub_block[block], dim_x, dim_y, dim_z);
        }
        auto t6 = std::chrono::high_resolution_clock::now();
        time_taken_decompress += std::chrono::duration_cast<std::chrono::nanoseconds>(t6 - t5).count() * 1e-9;
    };

    // High Level 串行调度 (继续使用 a 放大)
    double eb_high_stage1 = eb_stage3 * a;
    process_high_stage({6}, eb_high_stage1);       // Stage 1: Body Center
    
    double eb_high_stage2 = eb_high_stage1 * a;
    process_high_stage({2, 4, 5}, eb_high_stage2); // Stage 2: Face Centers
    
    double eb_high_stage3 = eb_high_stage2 * a;
    process_high_stage({0, 1, 3}, eb_high_stage3); // Stage 3: Edge Centers

    double original_size = (double)full_dim_z * full_dim_y * full_dim_x * sizeof(float);
    double CR = original_size / (double)allSize;
    std::cout << "CR: " << CR << std::endl;

    auto reconstructed_full_start = std::chrono::high_resolution_clock::now();
    float* reconstructed_full_data = new float[full_dim_z * full_dim_y * full_dim_x];
    merge_sub_blocks_to_full(current_high_de_blocks, reconstructed_full_data, dim_x, dim_y, dim_z);
    auto reconstructed_full_end = std::chrono::high_resolution_clock::now();
    double time_taken_reconstructed_full = std::chrono::duration_cast<std::chrono::nanoseconds>(reconstructed_full_end - reconstructed_full_start).count() * 1e-9;

    // writeBinaryData("ours.raw", reconstructed_full_data, full_dim_z * full_dim_y * full_dim_x);
    // ==============================================================
    // Calculate Overall PSNR & Performance metrics
    // ==============================================================
    double mse_full = 0.0;
    for (size_t i = 0; i < full_dim_z * full_dim_y * full_dim_x; ++i) {
        double diff = full_data[i] - reconstructed_full_data[i];
        mse_full += diff * diff;
    }
    mse_full /= (full_dim_z * full_dim_y * full_dim_x);
    double range_full = computeRange(full_data, full_dim_z * full_dim_y * full_dim_x);
    double psnr_full = 20 * log10(range_full) - 10 * log10(mse_full);
    std::cout << "Global PSNR: " << psnr_full << std::endl;

    std::cout << "compress time is: " << std::fixed << std::setprecision(5)
              << sz_time_taken_split + low_time_taken + low_time_taken_decompress - low_time_taken_decompress_sz + time_taken_reconstructed_low + time_taken << " sec" << std::endl;
    
    std::cout << "decompress time is: " << std::fixed << std::setprecision(5)
              << sz_time_taken_decompress + low_time_taken_decompress + time_taken_reconstructed_low + time_taken_decompress + time_taken_reconstructed_full << " sec" << std::endl;

    // ==============================================================
    // Memory Clean up
    // ==============================================================
    delete[] full_data;
    delete[] reconstructed_sub_0;
    delete[] decompressed_data;
    delete[] reconstructed_full_data;
    
    for (int i = 0; i < 8; ++i) {
        delete[] sub_block_data[i];
        delete[] low_block_data[i];
    }
    for (int i = 0; i < 7; ++i) {
        delete[] low_diff_data[i]; 
        delete[] low_deData[i]; 
        delete[] low_de_sub_block[i];
        delete[] diff_data[i]; 
        delete[] deData[i]; 
        delete[] de_sub_block[i];
    }

    return 0;
}
