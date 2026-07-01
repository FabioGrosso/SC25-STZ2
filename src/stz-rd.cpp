#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <cfloat>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <algorithm>
#include <vector>
#include "sz.hpp"

int full_dim_z = 512;
int full_dim_y = 512;
int full_dim_x = 512;
int dim_z = 256;
int dim_y = 256;
int dim_x = 256;
int low_dim_z = 128;
int low_dim_y = 128;
int low_dim_x = 128;

static void set_full_dims(int x, int y, int z)
{
    full_dim_x = x;
    full_dim_y = y;
    full_dim_z = z;
    dim_x = x / 2;
    dim_y = y / 2;
    dim_z = z / 2;
    low_dim_x = x / 4;
    low_dim_y = y / 4;
    low_dim_z = z / 4;
}
// const int full_dim_z = 100;
// const int full_dim_y = 500;
// const int full_dim_x = 500;
// const int dim_z = 50;
// const int dim_y = 250;
// const int dim_x = 250;
// const int low_dim_z = 25;
// const int low_dim_y = 125;
// const int low_dim_x = 125;
// const int full_dim_z = 1024;
// const int full_dim_y = 1024;
// const int full_dim_x = 1024;
// const int dim_z = 512;
// const int dim_y = 512;
// const int dim_x = 512;
// const int low_dim_z = 256;
// const int low_dim_y = 256;
// const int low_dim_x = 256;
template <typename T>
void merge_sub_blocks_to_full(T* sub_blocks[8], T* full_data, int sub_dim_x, int sub_dim_y, int sub_dim_z) {
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

template <typename T>
double computeRange(const T* data, size_t num_elements) {
    double high = -std::numeric_limits<double>::max();
    double low = std::numeric_limits<double>::max();
    for (size_t i = 0; i < num_elements; i++) {
        if (data[i] > high)
            high = data[i];
        if (data[i] < low)
            low = data[i];
    }
    double range = high-low;
    return range;
}


template <typename T>
void slice_full_data(const T* full_data, T* sub_block_data[8],int dim_x, int dim_y, int dim_z) {
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
// Preprocessing: Compute diff = sub_block - reference (sz_out_data)
// block index: 0 corresponds to "001", 1 to "010", 2 to "011", 3 to "100",
// 4 to "101", 5 to "110", 6 to "111"
//---------------------------------------------------------------------
template <typename T>
void preprocess_block(int block, const T* sub_block, const T* ref, T* diff,
                        int dim_x, int dim_y, int dim_z)
{
    int dim_xy= dim_x * dim_y;
    #pragma omp parallel for 
    for (int z = 0; z < dim_z; ++z)
    {
        for (int y = 0; y < dim_y; ++y)
        {
            for (int x = 0; x < dim_x; ++x)
            {
                int idx = z * dim_xy + y * dim_x + x;
                switch(block)
                {
                    case 0: // "001"
                        if (x == dim_x - 1)
                            diff[idx] = sub_block[idx] - ref[idx];
                        else if (x == dim_x - 2 || x == 0)
                        {
                            int idx_x1 = idx + 1;
                            diff[idx] = sub_block[idx] - (0.5f * ref[idx] + 0.5f * ref[idx_x1]);
                        }
                        else
                        {
                            int idx_x1 = idx + 1;
                            diff[idx] = sub_block[idx] - (-(1.0f/16.0f)*ref[idx-1] + (9.0f/16.0f)*ref[idx] +
                                                           (9.0f/16.0f)*ref[idx_x1] - (1.0f/16.0f)*ref[idx_x1+1]);
                        }
                        break;
                    case 1: // "010"
                        if (y == dim_y - 1)
                            diff[idx] = sub_block[idx] - ref[idx];
                        else if (y == dim_y - 2 || y == 0)
                        {
                            int idx_y1 = idx + dim_x;
                            diff[idx] = sub_block[idx] - (0.5f * ref[idx] + 0.5f * ref[idx_y1]);
                        }
                        else
                        {
                            int idx_y1 = idx + dim_x;
                            diff[idx] = sub_block[idx] - (-(1.0f/16.0f)*ref[idx-dim_x] + (9.0f/16.0f)*ref[idx] +
                                                           (9.0f/16.0f)*ref[idx_y1] - (1.0f/16.0f)*ref[idx_y1+dim_x]);
                        }
                        break;
                    case 2: // "011"
                        if (x == dim_x - 1 || y == dim_y - 1)
                            diff[idx] = sub_block[idx] - ref[idx];
                        else if (y == dim_y - 2 || y == 0 || x == dim_x - 2 || x == 0)
                        {
                            int idx_y1  = idx + dim_x;
                            int idx_xy1 = idx_y1 + 1;
                            int idx_x1  = idx + 1;
                            diff[idx] = sub_block[idx] - (0.25f * ref[idx] + 0.25f * ref[idx_xy1] +
                                                           0.25f * ref[idx_y1] + 0.25f * ref[idx_x1]);
                        }
                        else
                        {
                            int idx_px_py = idx - dim_x - 1;
                            int idx_xy1 = idx + dim_x + 1;
                            int idx_xy1_nx_ny = idx_xy1 + dim_x + 1;
                            int idx_y1 = idx + dim_x;
                            int idx_y1_px_ny = idx_y1 + dim_x - 1;
                            int idx_x1 = idx + 1;
                            int idx_x1_nx_py = idx_x1 - dim_x + 1;
                            diff[idx] = sub_block[idx] - (0.28125f * ref[idx] + 0.28125f * ref[idx_xy1] +
                                                           0.28125f * ref[idx_y1] + 0.28125f * ref[idx_x1] -
                                                           0.03125f * ref[idx_px_py] - 0.03125f * ref[idx_xy1_nx_ny] -
                                                           0.03125f * ref[idx_y1_px_ny] - 0.03125f * ref[idx_x1_nx_py]);
                        }
                        break;
                    case 3: // "100"
                        if (z == dim_z - 1)
                            diff[idx] = sub_block[idx] - ref[idx];
                        else if (z == dim_z - 2 || z == 0)
                        {
                            int idx_z1 = idx + dim_xy;
                            diff[idx] = sub_block[idx] - (0.5f * ref[idx] + 0.5f * ref[idx_z1]);
                        }
                        else
                        {
                            int idx_z1 = idx + dim_xy;
                            diff[idx] = sub_block[idx] - (-(1.0f/16.0f)*ref[idx-dim_xy] + (9.0f/16.0f)*ref[idx] +
                                                           (9.0f/16.0f)*ref[idx_z1] - (1.0f/16.0f)*ref[idx_z1+dim_xy]);
                        }
                        break;
                    case 4: // "101"
                        if (x == dim_x - 1 || z == dim_z - 1)
                            diff[idx] = sub_block[idx] - ref[idx];
                        else if (z == dim_z - 2 || z == 0 || x == dim_x - 2 || x == 0)
                        {
                            int idx_x1 = idx + 1;
                            int idx_z1 = idx + dim_xy;
                            int idx_zx1 = idx_z1 + 1;
                            diff[idx] = sub_block[idx] - (0.25f * ref[idx] + 0.25f * ref[idx_zx1] +
                                                           0.25f * ref[idx_x1] + 0.25f * ref[idx_z1]);
                        }
                        else
                        {
                            int idx_px_pz = idx - dim_xy - 1;
                            int idx_x1 = idx + 1;
                            int idx_x1_nx_pz = idx_x1 - dim_xy + 1;
                            int idx_z1 = idx + dim_xy;
                            int idx_z1_px_nz = idx_z1 + dim_xy - 1;
                            int idx_zx1 = idx + dim_xy + 1;
                            int idx_zx1_px_pz = idx_zx1 + dim_xy + 1;
                            diff[idx] = sub_block[idx] - (0.28125f * ref[idx] + 0.28125f * ref[idx_zx1] +
                                                           0.28125f * ref[idx_x1] + 0.28125f * ref[idx_z1] -
                                                           0.03125f * ref[idx_px_pz] - 0.03125f * ref[idx_x1_nx_pz] -
                                                           0.03125f * ref[idx_z1_px_nz] - 0.03125f * ref[idx_zx1_px_pz]);
                        }
                        break;
                    case 5: // "110"
                        if (z == dim_z - 1 || y == dim_y - 1)
                            diff[idx] = sub_block[idx] - ref[idx];
                        else if (y == dim_y - 2 || y == 0 || z == dim_z - 2 || z == 0)
                        {
                            int idx_z1 = idx + dim_xy;
                            int idx_zy1 = idx_z1 + dim_x;
                            int idx_y1 = idx + dim_x;
                            diff[idx] = sub_block[idx] - (0.25f * ref[idx] + 0.25f * ref[idx_zy1] +
                                                           0.25f * ref[idx_z1] + 0.25f * ref[idx_y1]);
                        }
                        else
                        {
                            int idx_py_pz = idx - dim_xy - dim_x;
                            int idx_zy1 = idx + dim_xy + dim_x;
                            int idx_zy1_nz_ny = idx_zy1 + dim_xy + dim_x;
                            int idx_z1 = idx + dim_xy;
                            int idx_z1_nz_py = idx_z1 + dim_xy - dim_x;
                            int idx_y1 = idx + dim_x;
                            int idx_y1_pz_ny = idx_y1 - dim_xy + dim_x;
                            diff[idx] = sub_block[idx] - (0.28125f * ref[idx] + 0.28125f * ref[idx_zy1] +
                                                           0.28125f * ref[idx_z1] + 0.28125f * ref[idx_y1] -
                                                           0.03125f * ref[idx_py_pz] - 0.03125f * ref[idx_zy1_nz_ny] -
                                                           0.03125f * ref[idx_z1_nz_py] - 0.03125f * ref[idx_y1_pz_ny]);
                        }
                        break;
                    case 6: // "111"
                        if (x == dim_x - 1 || y == dim_y - 1 || z == dim_z - 1)
                            diff[idx] = sub_block[idx] - ref[idx];
                        else if (z == dim_z - 2 || z == 0 || y == dim_y - 2 || y == 0 || x == dim_x - 2 || x == 0)
                        // else 
                        {
                            int idx_z1 = idx + dim_xy;
                            int idx_y1 = idx + dim_x;
                            int idx_x1 = idx + 1;
                            int idx_xy1 = idx_y1 + 1;
                            int idx_zy1 = idx_z1 + dim_x;
                            int idx_zx1 = idx_z1 + 1;
                            int idx_zyx1 = idx_zy1 + 1;
                            diff[idx] = sub_block[idx] - (0.125f * ref[idx] + 0.125f * ref[idx_zyx1] +
                                                           0.125f * ref[idx_zy1] + 0.125f * ref[idx_zx1] +
                                                           0.125f * ref[idx_xy1] + 0.125f * ref[idx_x1] +
                                                           0.125f * ref[idx_y1] + 0.125f * ref[idx_z1]);
                        }
                        else
                        {
                            int idx_px_py_pz = idx - dim_xy - dim_x - 1;
                            int idx_zyx1 = idx + dim_xy + dim_x + 1;
                            int idx_zyx1_nx_ny_nz = idx + dim_xy + dim_x + 1 + dim_xy + dim_x + 1;
                            int idx_zx1 = idx + dim_xy + 1;
                            int idx_zx1_nx_py_nz = idx + dim_xy + 1 + dim_xy - dim_x + 1;
                            int idx_zy1 = idx + dim_xy + dim_x;
                            int idx_zy1_px_ny_nz = idx + dim_xy + dim_x + dim_xy + dim_x - 1;
                            int idx_xy1 = idx + dim_x + 1;
                            int idx_xy1_nx_ny_pz = idx + dim_x + 1 - dim_xy + dim_x + 1;
                            int idx_z1 = idx + dim_xy;
                            int idx_z1_px_py_nz = idx + dim_xy + dim_xy - dim_x - 1;
                            int idx_y1 = idx + dim_x;
                            int idx_y1_px_ny_pz = idx + dim_x - dim_xy + dim_x - 1;
                            int idx_x1 = idx + 1;
                            int idx_x1_nx_py_pz = idx + 1 - dim_xy - dim_x + 1;
                            diff[idx] = sub_block[idx] - (0.140625f * ref[idx] + 0.140625f * ref[idx_zyx1] +
                                                           0.140625f * ref[idx_zy1] + 0.140625f * ref[idx_zx1] +
                                                           0.140625f * ref[idx_xy1] + 0.140625f * ref[idx_x1] +
                                                           0.140625f * ref[idx_y1] + 0.140625f * ref[idx_z1] -
                                                           0.015625f * ref[idx_px_py_pz] - 0.015625f * ref[idx_zyx1_nx_ny_nz] -
                                                           0.015625f * ref[idx_zx1_nx_py_nz] - 0.015625f * ref[idx_zy1_px_ny_nz] -
                                                           0.015625f * ref[idx_xy1_nx_ny_pz] - 0.015625f * ref[idx_z1_px_py_nz] -
                                                           0.015625f * ref[idx_y1_px_ny_pz] - 0.015625f * ref[idx_x1_nx_py_pz]);
                        }
                        break;
                    default:
                        std::cerr << "Unsupported block index in preprocess_block\n";
                        break;
                } // end switch
            }
        }
    }
}

//---------------------------------------------------------------------
// De-preprocessing: Reconstruct de_sub = deData + reference
//---------------------------------------------------------------------
template <typename T>
void depreprocess_block(int block, const T* deData, const T* ref, T* de_sub,
                          int dim_x, int dim_y, int dim_z,
                          int roi_x = -1, int roi_y = -1, int roi_z = -1,
                          int start_x = 0, int start_y = 0, int start_z = 0)
{
    int dim_xy= dim_x * dim_y;
    const int end_x = roi_x < 0 ? dim_x : roi_x;
    const int end_y = roi_y < 0 ? dim_y : roi_y;
    const int end_z = roi_z < 0 ? dim_z : roi_z;
    for (int z = start_z; z < end_z; ++z)
    {
        for (int y = start_y; y < end_y; ++y)
        {
            for (int x = start_x; x < end_x; ++x)
            {
                int idx = z * dim_xy + y * dim_x + x;
                switch(block)
                {
                    case 0: // "001"
                        if (x == dim_x - 1)
                            de_sub[idx] = deData[idx] + ref[idx];
                        else if (x == dim_x - 2 || x == 0)
                        {
                            int idx_x1 = idx + 1;
                            de_sub[idx] = deData[idx] + (0.5f * ref[idx] + 0.5f * ref[idx_x1]);
                        }
                        else
                        {
                            int idx_x1 = idx + 1;
                            de_sub[idx] = deData[idx] + (-(1.0f/16.0f)*ref[idx-1] + (9.0f/16.0f)*ref[idx] +
                                                         (9.0f/16.0f)*ref[idx_x1] - (1.0f/16.0f)*ref[idx_x1+1]);
                        }
                        break;
                    case 1: // "010"
                        if (y == dim_y - 1)
                            de_sub[idx] = deData[idx] + ref[idx];
                        else if (y == dim_y - 2 || y == 0)
                        {
                            int idx_y1 = idx + dim_x;
                            de_sub[idx] = deData[idx] + (0.5f * ref[idx] + 0.5f * ref[idx_y1]);
                        }
                        else
                        {
                            int idx_y1 = idx + dim_x;
                            de_sub[idx] = deData[idx] + (-(1.0f/16.0f)*ref[idx-dim_x] + (9.0f/16.0f)*ref[idx] +
                                                         (9.0f/16.0f)*ref[idx_y1] - (1.0f/16.0f)*ref[idx_y1+dim_x]);
                        }
                        break;
                    case 2: // "011"
                        if (x == dim_x - 1 || y == dim_y - 1)
                            de_sub[idx] = deData[idx] + ref[idx];
                        else if (y == dim_y - 2 || y == 0 || x == dim_x - 2 || x == 0)
                        {
                            int idx_y1  = idx + dim_x;
                            int idx_xy1 = idx_y1 + 1;
                            int idx_x1  = idx + 1;
                            de_sub[idx] = deData[idx] + (0.25f * ref[idx] + 0.25f * ref[idx_xy1] +
                                                         0.25f * ref[idx_y1] + 0.25f * ref[idx_x1]);
                        }
                        else
                        {
                            int idx_px_py = idx - dim_x - 1;
                            int idx_xy1 = idx + dim_x + 1;
                            int idx_xy1_nx_ny = idx_xy1 + dim_x + 1;
                            int idx_y1 = idx + dim_x;
                            int idx_y1_px_ny = idx_y1 + dim_x - 1;
                            int idx_x1 = idx + 1;
                            int idx_x1_nx_py = idx_x1 - dim_x + 1;
                            de_sub[idx] = deData[idx] + (0.28125f * ref[idx] + 0.28125f * ref[idx_xy1] +
                                                         0.28125f * ref[idx_y1] + 0.28125f * ref[idx_x1] -
                                                         0.03125f * ref[idx_px_py] - 0.03125f * ref[idx_xy1_nx_ny] -
                                                         0.03125f * ref[idx_y1_px_ny] - 0.03125f * ref[idx_x1_nx_py]);
                        }
                        break;
                    case 3: // "100"
                        if (z == dim_z - 1)
                            de_sub[idx] = deData[idx] + ref[idx];
                        else if (z == dim_z - 2 || z == 0)
                        {
                            int idx_z1 = idx + dim_xy;
                            de_sub[idx] = deData[idx] + (0.5f * ref[idx] + 0.5f * ref[idx_z1]);
                        }
                        else
                        {
                            int idx_z1 = idx + dim_xy;
                            de_sub[idx] = deData[idx] + (-(1.0f/16.0f)*ref[idx-dim_xy] + (9.0f/16.0f)*ref[idx] +
                                                         (9.0f/16.0f)*ref[idx_z1] - (1.0f/16.0f)*ref[idx_z1+dim_xy]);
                        }
                        break;
                    case 4: // "101"
                        if (x == dim_x - 1 || z == dim_z - 1)
                            de_sub[idx] = deData[idx] + ref[idx];
                        else if (z == dim_z - 2 || z == 0 || x == dim_x - 2 || x == 0)
                        {
                            int idx_x1 = idx + 1;
                            int idx_z1 = idx + dim_xy;
                            int idx_zx1 = idx_z1 + 1;
                            de_sub[idx] = deData[idx] + (0.25f * ref[idx] + 0.25f * ref[idx_zx1] +
                                                         0.25f * ref[idx_x1] + 0.25f * ref[idx_z1]);
                        }
                        else
                        {
                            int idx_px_pz = idx - dim_xy - 1;
                            int idx_x1 = idx + 1;
                            int idx_x1_nx_pz = idx_x1 - dim_xy + 1;
                            int idx_z1 = idx + dim_xy;
                            int idx_z1_px_nz = idx_z1 + dim_xy - 1;
                            int idx_zx1 = idx + dim_xy + 1;
                            int idx_zx1_px_pz = idx_zx1 + dim_xy + 1;
                            de_sub[idx] = deData[idx] + (0.28125f * ref[idx] + 0.28125f * ref[idx_zx1] +
                                                         0.28125f * ref[idx_x1] + 0.28125f * ref[idx_z1] -
                                                         0.03125f * ref[idx_px_pz] - 0.03125f * ref[idx_x1_nx_pz] -
                                                         0.03125f * ref[idx_z1_px_nz] - 0.03125f * ref[idx_zx1_px_pz]);
                        }
                        break;
                    case 5: // "110"
                        if (z == dim_z - 1 || y == dim_y - 1)
                            de_sub[idx] = deData[idx] + ref[idx];
                        else if (y == dim_y - 2 || y == 0 || z == dim_z - 2 || z == 0)
                        {
                            int idx_z1 = idx + dim_xy;
                            int idx_zy1 = idx_z1 + dim_x;
                            int idx_y1 = idx + dim_x;
                            de_sub[idx] = deData[idx] + (0.25f * ref[idx] + 0.25f * ref[idx_zy1] +
                                                         0.25f * ref[idx_z1] + 0.25f * ref[idx_y1]);
                        }
                        else
                        {
                            int idx_py_pz = idx - dim_xy - dim_x;
                            int idx_zy1 = idx + dim_xy + dim_x;
                            int idx_zy1_nz_ny = idx_zy1 + dim_xy + dim_x;
                            int idx_z1 = idx + dim_xy;
                            int idx_z1_nz_py = idx_z1 + dim_xy - dim_x;
                            int idx_y1 = idx + dim_x;
                            int idx_y1_pz_ny = idx_y1 - dim_xy + dim_x;
                            de_sub[idx] = deData[idx] + (0.28125f * ref[idx] + 0.28125f * ref[idx_zy1] +
                                                         0.28125f * ref[idx_z1] + 0.28125f * ref[idx_y1] -
                                                         0.03125f * ref[idx_py_pz] - 0.03125f * ref[idx_zy1_nz_ny] -
                                                         0.03125f * ref[idx_z1_nz_py] - 0.03125f * ref[idx_y1_pz_ny]);
                        }
                        break;
                    case 6: // "111"
                        if (x == dim_x - 1 || y == dim_y - 1 || z == dim_z - 1)
                            de_sub[idx] = deData[idx] + ref[idx];
                        else if (z == dim_z - 2 || z == 0 || y == dim_y - 2 || y == 0 || x == dim_x - 2 || x == 0)
                        // else 
                        {
                            int idx_z1 = idx + dim_xy;
                            int idx_y1 = idx + dim_x;
                            int idx_x1 = idx + 1;
                            int idx_xy1 = idx_y1 + 1;
                            int idx_zy1 = idx_z1 + dim_x;
                            int idx_zx1 = idx_z1 + 1;
                            int idx_zyx1 = idx_zy1 + 1;
                            de_sub[idx] = deData[idx] + (0.125f * ref[idx] + 0.125f * ref[idx_zyx1] +
                                                         0.125f * ref[idx_zy1] + 0.125f * ref[idx_zx1] +
                                                         0.125f * ref[idx_xy1] + 0.125f * ref[idx_x1] +
                                                         0.125f * ref[idx_y1] + 0.125f * ref[idx_z1]);
                        }
                        else
                        {
                            int idx_px_py_pz = idx - dim_xy - dim_x - 1;
                            int idx_zyx1 = idx + dim_xy + dim_x + 1;
                            int idx_zyx1_nx_ny_nz = idx + dim_xy + dim_x + 1 + dim_xy + dim_x + 1;
                            int idx_zx1 = idx + dim_xy + 1;
                            int idx_zx1_nx_py_nz = idx + dim_xy + 1 + dim_xy - dim_x + 1;
                            int idx_zy1 = idx + dim_xy + dim_x;
                            int idx_zy1_px_ny_nz = idx + dim_xy + dim_x + dim_xy + dim_x - 1;
                            int idx_xy1 = idx + dim_x + 1;
                            int idx_xy1_nx_ny_pz = idx + dim_x + 1 - dim_xy + dim_x + 1;
                            int idx_z1 = idx + dim_xy;
                            int idx_z1_px_py_nz = idx + dim_xy + dim_xy - dim_x - 1;
                            int idx_y1 = idx + dim_x;
                            int idx_y1_px_ny_pz = idx + dim_x - dim_xy + dim_x - 1;
                            int idx_x1 = idx + 1;
                            int idx_x1_nx_py_pz = idx + 1 - dim_xy - dim_x + 1;
                            de_sub[idx] = deData[idx] + (0.140625f * ref[idx] + 0.140625f * ref[idx_zyx1] +
                                                         0.140625f * ref[idx_zy1] + 0.140625f * ref[idx_zx1] +
                                                         0.140625f * ref[idx_xy1] + 0.140625f * ref[idx_x1] +
                                                         0.140625f * ref[idx_y1] + 0.140625f * ref[idx_z1] -
                                                         0.015625f * ref[idx_px_py_pz] - 0.015625f * ref[idx_zyx1_nx_ny_nz] -
                                                         0.015625f * ref[idx_zx1_nx_py_nz] - 0.015625f * ref[idx_zy1_px_ny_nz] -
                                                         0.015625f * ref[idx_xy1_nx_ny_pz] - 0.015625f * ref[idx_z1_px_py_nz] -
                                                         0.015625f * ref[idx_y1_px_ny_pz] - 0.015625f * ref[idx_x1_nx_py_pz]);
                        }
                        break;
                    default:
                        std::cerr << "Unsupported block index in depreprocess_block\n";
                        break;
                }
            }
        }
    }
}

//---------------------------------------------------------------------
// SZ compression/decompression and file I/O routines (unchanged)
//---------------------------------------------------------------------
template <typename T>
char* SZ_compress(T* oriData, size_t blksize_x, size_t blksize_y, size_t blksize_z, double eb, size_t& outSize)
{
    SZ3::Config conf(blksize_z, blksize_y, blksize_x);
    conf.cmprAlgo = SZ3::ALGO_NOPRED;
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = eb;
    char* compressedData = SZ_compress<T>(conf, oriData, outSize);
    return compressedData;
}

template <typename T>
char* SZ_compress4De(T* oriData, size_t blksize_x, size_t blksize_y, size_t blksize_z, double eb, size_t& outSize)
{
    SZ3::Config conf(blksize_z, blksize_y, blksize_x);
    conf.cmprAlgo = SZ3::ALGO_INTERP;
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = eb;
    char* compressedData = SZ_compress<T>(conf, oriData, outSize);
    // Note: Returning original data (as in your original code)
    return compressedData;
}

template <typename T>
T* SZ_decompress4De(char* compressedData, size_t outSize, size_t blksize_x, size_t blksize_y, size_t blksize_z)
{
    SZ3::Config conf(blksize_z, blksize_y, blksize_x);
    conf.cmprAlgo = SZ3::ALGO_INTERP;
    conf.errorBoundMode = SZ3::EB_ABS;
    T* deData = new T[blksize_x * blksize_y * blksize_z];
    SZ_decompress<T>(conf, compressedData, outSize, deData);
    return deData;
}

template <typename T>
T* SZ_decompress_separated(char* compressedData, size_t outSize, size_t blksize_x, size_t blksize_y, size_t blksize_z)
{
    SZ3::Config conf(blksize_z, blksize_y, blksize_x);
    conf.cmprAlgo = SZ3::ALGO_NOPRED;
    conf.errorBoundMode = SZ3::EB_ABS;
    T* deData = new T[blksize_x * blksize_y * blksize_z];
    SZ_decompress<T>(conf, compressedData, outSize, deData);
    return deData;
}

template <typename T>
bool readBinaryData(const std::string& filepath, T* data, size_t dataSize)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open file for reading: " << filepath << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(data), dataSize * sizeof(T));
    if (!file)
    {
        std::cerr << "Failed to read data from file: " << filepath << std::endl;
        return false;
    }
    file.close();
    return true;
}

template <typename T>
bool writeBinaryData(const std::string& filepath, const T* data, size_t dataSize)
{
    std::ofstream file(filepath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    file.write(reinterpret_cast<const char*>(data), dataSize * sizeof(T));
    if (!file)
    {
        std::cerr << "Failed to write data to file: " << filepath << std::endl;
        return false;
    }
    file.close();
    return true;
}

//---------------------------------------------------------------------
// Random-access query planning (baseline: same footprint analysis as STZ2,
// but WITHOUT a marker table, so the residual streams of the needed sub-blocks
// are still decoded in full -- only depreprocessing and merging are restricted
// to the query footprint).
//---------------------------------------------------------------------
struct AxisRange { int begin = 0; int end = -1; };
struct QueryBox { AxisRange x; AxisRange y; AxisRange z; bool is_slice = false; };
struct BlockAccessPlan { int block = 0; AxisRange x; AxisRange y; AxisRange z; };

static AxisRange normalize_axis(int a, int b, int dim, bool& is_slice)
{
    if (a == b) {
        is_slice = true;
        const int v = std::max(0, std::min(a, dim - 1));
        return {v, v};
    }
    int lo = std::max(0, std::min(a, b));
    int hi = std::min(dim - 1, std::max(a, b));
    if (hi < lo) hi = lo;
    return {lo, hi};
}

static QueryBox make_query(int ax, int ay, int az, int bx, int by, int bz)
{
    QueryBox q;
    q.x = normalize_axis(ax, bx, full_dim_x, q.is_slice);
    q.y = normalize_axis(ay, by, full_dim_y, q.is_slice);
    q.z = normalize_axis(az, bz, full_dim_z, q.is_slice);
    return q;
}

static AxisRange range_for_parity(AxisRange fullRange, int parity)
{
    int first = fullRange.begin;
    if ((first & 1) != parity) first++;
    if (first > fullRange.end) return {1, 0};
    int last = fullRange.end;
    if ((last & 1) != parity) last--;
    return {first / 2, last / 2};
}

static void merge_axis(AxisRange& dst, AxisRange src)
{
    if (src.begin > src.end) return;
    if (dst.begin > dst.end) { dst = src; return; }
    dst.begin = std::min(dst.begin, src.begin);
    dst.end = std::max(dst.end, src.end);
}

// Reference cells read by the prediction stencil along a "used" (offset) axis.
static AxisRange ref_range_for_used_axis(AxisRange target, int dim)
{
    if (target.begin > target.end) return target;
    int lo = target.begin, hi = target.end;
    const int interior_begin = std::max(target.begin, 1);
    const int interior_end = std::min(target.end, dim - 3);
    if (interior_begin <= interior_end) {
        lo = std::min(lo, interior_begin - 1);
        hi = std::max(hi, interior_end + 2);
    }
    const int nonlast_begin = std::max(target.begin, 0);
    const int nonlast_end = std::min(target.end, dim - 2);
    if (nonlast_begin <= nonlast_end) hi = std::max(hi, nonlast_end + 1);
    return {std::max(0, lo), std::min(dim - 1, hi)};
}

static QueryBox build_high_ref_region(const BlockAccessPlan& highPlan)
{
    const int sub_index = highPlan.block + 1;
    QueryBox q;
    q.x = (sub_index & 1) ? ref_range_for_used_axis(highPlan.x, dim_x) : highPlan.x;
    q.y = (sub_index & 2) ? ref_range_for_used_axis(highPlan.y, dim_y) : highPlan.y;
    q.z = (sub_index & 4) ? ref_range_for_used_axis(highPlan.z, dim_z) : highPlan.z;
    q.is_slice = (q.x.begin == q.x.end) || (q.y.begin == q.y.end) || (q.z.begin == q.z.end);
    return q;
}

static QueryBox build_query_sub0_region(const QueryBox& q)
{
    QueryBox sub0;
    sub0.x = range_for_parity(q.x, 0);
    sub0.y = range_for_parity(q.y, 0);
    sub0.z = range_for_parity(q.z, 0);
    return sub0;
}

// Which high (level-3) sub-blocks the query touches, and their sub-block ranges.
static std::vector<BlockAccessPlan> build_residual_plan(const QueryBox& q)
{
    std::vector<BlockAccessPlan> plans;
    for (int block = 0; block < 7; ++block) {
        const int sub_index = block + 1;
        BlockAccessPlan p;
        p.block = block;
        p.x = range_for_parity(q.x, sub_index & 1);
        p.y = range_for_parity(q.y, (sub_index >> 1) & 1);
        p.z = range_for_parity(q.z, (sub_index >> 2) & 1);
        if (p.x.begin > p.x.end || p.y.begin > p.y.end || p.z.begin > p.z.end) continue;
        plans.push_back(p);
    }
    return plans;
}

// Which low (level-2) sub-blocks (and ranges) are needed to rebuild the high
// reference footprint.
static std::vector<BlockAccessPlan> build_low_residual_plan(const std::vector<QueryBox>& highRefRegions)
{
    std::vector<BlockAccessPlan> accum(7);
    std::vector<bool> needed(7, false);
    for (int block = 0; block < 7; ++block) {
        accum[block].block = block;
        accum[block].x = accum[block].y = accum[block].z = {1, 0};
    }
    for (const auto& r : highRefRegions) {
        if (r.x.begin > r.x.end || r.y.begin > r.y.end || r.z.begin > r.z.end) continue;
        for (int block = 0; block < 7; ++block) {
            const int sub_index = block + 1;
            BlockAccessPlan p;
            p.x = range_for_parity(r.x, sub_index & 1);
            p.y = range_for_parity(r.y, (sub_index >> 1) & 1);
            p.z = range_for_parity(r.z, (sub_index >> 2) & 1);
            if (p.x.begin > p.x.end || p.y.begin > p.y.end || p.z.begin > p.z.end) continue;
            needed[block] = true;
            merge_axis(accum[block].x, p.x);
            merge_axis(accum[block].y, p.y);
            merge_axis(accum[block].z, p.z);
        }
    }
    std::vector<BlockAccessPlan> plans;
    for (int block = 0; block < 7; ++block)
        if (needed[block]) plans.push_back(accum[block]);
    return plans;
}

static std::string blocks_to_string(const std::vector<BlockAccessPlan>& plans)
{
    std::string s;
    for (const auto& p : plans) s += std::to_string(p.block) + " ";
    return s.empty() ? "(none)" : s;
}

//---------------------------------------------------------------------
// Main routine
//---------------------------------------------------------------------
static void print_usage(const char* exe)
{
    std::cerr << "Usage: " << exe
              << " <error_bound> <raw_file> <dim_x> [dim_y dim_z] <-f|-d> [--auto-query x0 y0 z0 x1 y1 z1]"
              << std::endl;
}

template <typename T>
int run_typed(int argc, char* argv[])
{
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    const double eb = std::atof(argv[1]);
    const std::string full_file_path = argv[2];
    const bool cube_args = std::string(argv[4]) == "-f" || std::string(argv[4]) == "-d";
    const int type_arg = cube_args ? 4 : 6;
    if (!cube_args && argc < 7) {
        print_usage(argv[0]);
        return 1;
    }

    const int input_full_x = std::atoi(argv[3]);
    const int input_full_y = cube_args ? input_full_x : std::atoi(argv[4]);
    const int input_full_z = cube_args ? input_full_x : std::atoi(argv[5]);
    if (input_full_x <= 0 || input_full_y <= 0 || input_full_z <= 0 ||
        input_full_x % 4 != 0 || input_full_y % 4 != 0 || input_full_z % 4 != 0) {
        std::cerr << "dims must be positive and divisible by 4, got "
                  << input_full_x << " x " << input_full_y << " x " << input_full_z << std::endl;
        return 1;
    }
    set_full_dims(input_full_x, input_full_y, input_full_z);

    // Optional random-access query: --auto-query x0 y0 z0 x1 y1 z1
    bool run_auto_query = false;
    std::vector<int> query_args;
    for (int arg = type_arg + 1; arg < argc; ++arg) {
        const std::string token = argv[arg];
        if (token == "--auto-query") { run_auto_query = true; continue; }
        char* e = nullptr;
        const long v = std::strtol(argv[arg], &e, 10);
        if (*argv[arg] == '\0' || *e != '\0') {
            std::cerr << "Invalid argument '" << argv[arg] << "'. Expected --auto-query or ROI integer." << std::endl;
            return 1;
        }
        query_args.push_back(static_cast<int>(v));
    }

    const size_t full_size = static_cast<size_t>(full_dim_z) * full_dim_y * full_dim_x;
    const size_t expected_bytes = full_size * sizeof(T);
    std::ifstream size_file(full_file_path, std::ios::binary | std::ios::ate);
    if (!size_file) {
        std::cerr << "Failed to open file for size check: " << full_file_path << std::endl;
        return 1;
    }
    const size_t actual_bytes = static_cast<size_t>(size_file.tellg());
    if (actual_bytes != expected_bytes) {
        std::cerr << "Input file size mismatch for "
                  << (sizeof(T) == sizeof(float) ? "float" : "double")
                  << ": expected " << expected_bytes << " bytes for "
                  << full_dim_x << " x " << full_dim_y << " x " << full_dim_z
                  << ", got " << actual_bytes << " bytes." << std::endl;
        return 1;
    }

    std::cout << "Input: " << full_file_path << ", dims: "
              << full_dim_x << " x " << full_dim_y << " x " << full_dim_z
              << ", type: " << (sizeof(T) == sizeof(float) ? "float" : "double")
              << std::endl;

    T* full_data = new T[full_size];
    if (!readBinaryData(full_file_path, full_data, full_size))
    {
        delete[] full_data;
        return 1;
    }

    auto global_compress_start = std::chrono::high_resolution_clock::now();

    auto split_start = std::chrono::high_resolution_clock::now();
    T* sub_block_data[8] = {};
    #pragma omp parallel for
    for (int i = 0; i < 8; ++i)
        sub_block_data[i] = new T[static_cast<size_t>(dim_z) * dim_x * dim_y];

    slice_full_data(full_data, sub_block_data, full_dim_x, full_dim_y, full_dim_z);

    T* low_block_data[8] = {};
    #pragma omp parallel for
    for (int i = 0; i < 8; ++i)
        low_block_data[i] = new T[static_cast<size_t>(low_dim_z) * low_dim_y * low_dim_x];

    slice_full_data(sub_block_data[0], low_block_data, dim_x, dim_y, dim_z);

    auto split_end = std::chrono::high_resolution_clock::now();
    double split_time = std::chrono::duration_cast<std::chrono::nanoseconds>(split_end - split_start).count() * 1e-9;

    size_t allSize = 0;
    size_t szcompressedSize = 0;
    auto base_compress_start = std::chrono::high_resolution_clock::now();
    char* tmp = SZ_compress4De(low_block_data[0], low_dim_x, low_dim_y, low_dim_z, eb, szcompressedSize);
    allSize += szcompressedSize;
    auto base_compress_end = std::chrono::high_resolution_clock::now();
    double base_compress_time = std::chrono::duration_cast<std::chrono::nanoseconds>(base_compress_end - base_compress_start).count() * 1e-9;

    auto low_buffer_start = std::chrono::high_resolution_clock::now();
    char* low_comp[7] = {};
    T* low_diff_data[7] = {};
    T* low_deData[7] = {};
    T* low_de_sub_block[7] = {};
    size_t low_compressedSize[7] = {};
    #pragma omp parallel for
    for (int i = 0; i < 7; ++i)
    {
        low_diff_data[i]    = new T[static_cast<size_t>(low_dim_z) * low_dim_y * low_dim_x];
        low_de_sub_block[i] = new T[static_cast<size_t>(low_dim_z) * low_dim_y * low_dim_x];
    }
    auto low_buffer_end = std::chrono::high_resolution_clock::now();
    double low_buffer_time = std::chrono::duration_cast<std::chrono::nanoseconds>(low_buffer_end - low_buffer_start).count() * 1e-9;

    auto low_compress_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:allSize)
    for (int block = 0; block < 7; ++block)
    {
        preprocess_block(block, low_block_data[block + 1], low_block_data[0], low_diff_data[block],
                         low_dim_x, low_dim_y, low_dim_z);
        low_comp[block] = SZ_compress(low_diff_data[block], low_dim_x, low_dim_y, low_dim_z,
                                      2.5 * eb, low_compressedSize[block]);
        allSize += low_compressedSize[block];
    }
    auto low_compress_end = std::chrono::high_resolution_clock::now();
    double low_compress_time = std::chrono::duration_cast<std::chrono::nanoseconds>(low_compress_end - low_compress_start).count() * 1e-9;

    auto low_deprocess_compress_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int block = 0; block < 7; ++block)
    {
        depreprocess_block(block, low_diff_data[block], low_block_data[0], low_de_sub_block[block],
                           low_dim_x, low_dim_y, low_dim_z);
    }
    auto low_deprocess_compress_end = std::chrono::high_resolution_clock::now();
    double low_deprocess_compress_time = std::chrono::duration_cast<std::chrono::nanoseconds>(low_deprocess_compress_end - low_deprocess_compress_start).count() * 1e-9;

    auto low_merge_compress_start = std::chrono::high_resolution_clock::now();
    T* reconstructed_sub_0 = new T[static_cast<size_t>(dim_z) * dim_x * dim_y];
    T* all_low_blocks[8] = {
        low_block_data[0],
        low_de_sub_block[0],
        low_de_sub_block[1],
        low_de_sub_block[2],
        low_de_sub_block[3],
        low_de_sub_block[4],
        low_de_sub_block[5],
        low_de_sub_block[6]
    };
    merge_sub_blocks_to_full(all_low_blocks, reconstructed_sub_0, low_dim_x, low_dim_y, low_dim_z);
    auto low_merge_compress_end = std::chrono::high_resolution_clock::now();
    double low_merge_compress_time = std::chrono::duration_cast<std::chrono::nanoseconds>(low_merge_compress_end - low_merge_compress_start).count() * 1e-9;

    auto high_buffer_start = std::chrono::high_resolution_clock::now();
    char* comp[7] = {};
    T* diff_data[7] = {};
    T* deData[7] = {};
    T* de_sub_block[7] = {};
    size_t compressedSize[7] = {};
    #pragma omp parallel for
    for (int i = 0; i < 7; ++i)
        diff_data[i] = new T[static_cast<size_t>(dim_z) * dim_x * dim_y];
    auto high_buffer_end = std::chrono::high_resolution_clock::now();
    double high_buffer_time = std::chrono::duration_cast<std::chrono::nanoseconds>(high_buffer_end - high_buffer_start).count() * 1e-9;

    auto high_compress_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:allSize)
    for (int block = 0; block < 7; ++block)
    {
        preprocess_block(block, sub_block_data[block + 1], reconstructed_sub_0, diff_data[block],
                         dim_x, dim_y, dim_z);
        comp[block] = SZ_compress(diff_data[block], dim_x, dim_y, dim_z,
                                  6.25 * eb, compressedSize[block]);
        allSize += compressedSize[block];
    }
    auto high_compress_end = std::chrono::high_resolution_clock::now();
    double high_compress_time = std::chrono::duration_cast<std::chrono::nanoseconds>(high_compress_end - high_compress_start).count() * 1e-9;
    auto global_compress_end = std::chrono::high_resolution_clock::now();

    const double original_size = static_cast<double>(full_dim_z) * full_dim_y * full_dim_x * sizeof(T);
    std::cout << "allSize: " << allSize << std::endl;
    std::cout << "CR: " << original_size / static_cast<double>(allSize) << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < 7; ++i)
    {
        low_deData[i] = new T[static_cast<size_t>(low_dim_z) * low_dim_y * low_dim_x];
        deData[i] = new T[static_cast<size_t>(dim_z) * dim_x * dim_y];
        de_sub_block[i] = new T[static_cast<size_t>(dim_z) * dim_x * dim_y];
    }

    auto global_decompress_start = std::chrono::high_resolution_clock::now();

    auto base_decompress_start = std::chrono::high_resolution_clock::now();
    T* decompressed_data = SZ_decompress4De<T>(tmp, szcompressedSize, low_dim_x, low_dim_y, low_dim_z);
    auto base_decompress_end = std::chrono::high_resolution_clock::now();
    double base_decompress_time = std::chrono::duration_cast<std::chrono::nanoseconds>(base_decompress_end - base_decompress_start).count() * 1e-9;

    auto low_decode_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int block = 0; block < 7; ++block)
    {
        low_deData[block] = SZ_decompress_separated<T>(low_comp[block], low_compressedSize[block],
                                                       low_dim_x, low_dim_y, low_dim_z);
    }
    auto low_decode_end = std::chrono::high_resolution_clock::now();
    double low_decode_time = std::chrono::duration_cast<std::chrono::nanoseconds>(low_decode_end - low_decode_start).count() * 1e-9;

    auto low_deprocess_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int block = 0; block < 7; ++block)
    {
        depreprocess_block(block, low_deData[block], decompressed_data, low_de_sub_block[block],
                           low_dim_x, low_dim_y, low_dim_z);
    }
    auto low_deprocess_end = std::chrono::high_resolution_clock::now();
    double low_deprocess_time = std::chrono::duration_cast<std::chrono::nanoseconds>(low_deprocess_end - low_deprocess_start).count() * 1e-9;

    auto low_merge_start = std::chrono::high_resolution_clock::now();
    all_low_blocks[0] = decompressed_data;
    merge_sub_blocks_to_full(all_low_blocks, reconstructed_sub_0, low_dim_x, low_dim_y, low_dim_z);
    auto low_merge_end = std::chrono::high_resolution_clock::now();
    double low_merge_time = std::chrono::duration_cast<std::chrono::nanoseconds>(low_merge_end - low_merge_start).count() * 1e-9;

    auto high_decode_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int block = 0; block < 7; ++block)
    {
        deData[block] = SZ_decompress_separated<T>(comp[block], compressedSize[block], dim_x, dim_y, dim_z);
    }
    auto high_decode_end = std::chrono::high_resolution_clock::now();
    double high_decode_time = std::chrono::duration_cast<std::chrono::nanoseconds>(high_decode_end - high_decode_start).count() * 1e-9;

    auto high_deprocess_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int block = 0; block < 7; ++block)
    {
        depreprocess_block(block, deData[block], reconstructed_sub_0, de_sub_block[block],
                           dim_x, dim_y, dim_z);
    }
    auto high_deprocess_end = std::chrono::high_resolution_clock::now();
    double high_deprocess_time = std::chrono::duration_cast<std::chrono::nanoseconds>(high_deprocess_end - high_deprocess_start).count() * 1e-9;

    auto high_merge_start = std::chrono::high_resolution_clock::now();
    T* reconstructed_full_data = new T[static_cast<size_t>(full_dim_z) * full_dim_y * full_dim_x];
    T* all_sub_blocks[8] = {
        reconstructed_sub_0,
        de_sub_block[0],
        de_sub_block[1],
        de_sub_block[2],
        de_sub_block[3],
        de_sub_block[4],
        de_sub_block[5],
        de_sub_block[6]
    };
    merge_sub_blocks_to_full(all_sub_blocks, reconstructed_full_data, dim_x, dim_y, dim_z);
    auto high_merge_end = std::chrono::high_resolution_clock::now();
    double high_merge_time = std::chrono::duration_cast<std::chrono::nanoseconds>(high_merge_end - high_merge_start).count() * 1e-9;
    auto global_decompress_end = std::chrono::high_resolution_clock::now();

    const size_t validate_size = static_cast<size_t>(full_dim_z) * full_dim_y * full_dim_x;
    double mse_full = 0.0;
    double low_full = std::numeric_limits<double>::max();
    double high_full = -std::numeric_limits<double>::max();
    #pragma omp parallel for reduction(+:mse_full) reduction(min:low_full) reduction(max:high_full)
    for (size_t i = 0; i < validate_size; ++i) {
        const double original = static_cast<double>(full_data[i]);
        const double diff = original - static_cast<double>(reconstructed_full_data[i]);
        mse_full += diff * diff;
        if (original < low_full) low_full = original;
        if (original > high_full) high_full = original;
    }
    mse_full /= validate_size;
    const double range_full = high_full - low_full;
    const double psnr_full = 20 * log10(range_full) - 10 * log10(mse_full);
    std::cout << "Global PSNR: " << psnr_full << std::endl;

    //-----------------------------------------------------------------
    // Baseline random-access query: full decode of the needed sub-block
    // residual streams (no marker table -> cannot skip within a stream;
    // for a slice, only the matching-parity sub-blocks are decoded -- the
    // only decode saving), then depreprocess + merge restricted to the ROI.
    //-----------------------------------------------------------------
    if (run_auto_query) {
        const int drx = std::min(64, full_dim_x);
        const int dry = std::min(64, full_dim_y);
        const int drz = std::min(64, full_dim_z);
        QueryBox query = make_query(0, 0, 0, drx - 1, dry - 1, drz - 1);
        if (!query_args.empty()) {
            if (query_args.size() != 6) {
                std::cerr << "ROI query must provide exactly six integers: x0 y0 z0 x1 y1 z1" << std::endl;
                return 1;
            }
            query = make_query(query_args[0], query_args[1], query_args[2],
                               query_args[3], query_args[4], query_args[5]);
        }
        std::cout << "Query [" << query.x.begin << "," << query.x.end << "] x ["
                  << query.y.begin << "," << query.y.end << "] x ["
                  << query.z.begin << "," << query.z.end << "] "
                  << (query.is_slice ? "(slice)" : "(box)") << std::endl;

        const auto highPlans = build_residual_plan(query);
        std::vector<QueryBox> highRefRegions;
        for (const auto& p : highPlans) highRefRegions.push_back(build_high_ref_region(p));
        const QueryBox sub0Region = build_query_sub0_region(query);
        if (sub0Region.x.begin <= sub0Region.x.end &&
            sub0Region.y.begin <= sub0Region.y.end &&
            sub0Region.z.begin <= sub0Region.z.end) {
            highRefRegions.push_back(sub0Region);
        }
        const auto lowPlans = build_low_residual_plan(highRefRegions);

        T* q_low_deData[7] = {};
        T* q_low_de_sub[7] = {};
        T* q_high_deData[7] = {};
        T* q_high_de_sub[7] = {};
        for (const auto& p : lowPlans) q_low_de_sub[p.block] = new T[static_cast<size_t>(low_dim_z) * low_dim_y * low_dim_x];
        for (const auto& p : highPlans) q_high_de_sub[p.block] = new T[static_cast<size_t>(dim_z) * dim_y * dim_x];

        // --- low level: FULL decode of the needed streams ---
        auto q_low_decode_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(lowPlans.size()); ++i) {
            const int b = lowPlans[i].block;
            q_low_deData[b] = SZ_decompress_separated<T>(low_comp[b], low_compressedSize[b],
                                                         low_dim_x, low_dim_y, low_dim_z);
        }
        auto q_low_decode_end = std::chrono::high_resolution_clock::now();
        double q_low_decode_time = std::chrono::duration_cast<std::chrono::nanoseconds>(q_low_decode_end - q_low_decode_start).count() * 1e-9;

        // --- low level: depreprocess restricted to ROI ---
        auto q_low_depre_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(lowPlans.size()); ++i) {
            const auto& p = lowPlans[i];
            depreprocess_block(p.block, q_low_deData[p.block], decompressed_data, q_low_de_sub[p.block],
                               low_dim_x, low_dim_y, low_dim_z,
                               p.x.end + 1, p.y.end + 1, p.z.end + 1, p.x.begin, p.y.begin, p.z.begin);
        }
        auto q_low_depre_end = std::chrono::high_resolution_clock::now();
        double q_low_depre_time = std::chrono::duration_cast<std::chrono::nanoseconds>(q_low_depre_end - q_low_depre_start).count() * 1e-9;

        // --- build the high-level reference (sub_0) ROI: merge restricted to footprint ---
        auto q_ref_start = std::chrono::high_resolution_clock::now();
        T* reconstructed_sub_0_ra = new T[static_cast<size_t>(dim_z) * dim_y * dim_x];
        for (const auto& region : highRefRegions) {
            for (int z = region.z.begin; z <= region.z.end; ++z)
                for (int y = region.y.begin; y <= region.y.end; ++y)
                    for (int x = region.x.begin; x <= region.x.end; ++x) {
                        const int sub_index = ((z & 1) << 2) | ((y & 1) << 1) | (x & 1);
                        const size_t dst = (static_cast<size_t>(z) * dim_y + y) * dim_x + x;
                        const size_t src = (static_cast<size_t>(z / 2) * low_dim_y + y / 2) * low_dim_x + x / 2;
                        reconstructed_sub_0_ra[dst] = sub_index == 0
                            ? decompressed_data[src] : q_low_de_sub[sub_index - 1][src];
                    }
        }
        auto q_ref_end = std::chrono::high_resolution_clock::now();
        double q_ref_time = std::chrono::duration_cast<std::chrono::nanoseconds>(q_ref_end - q_ref_start).count() * 1e-9;

        // --- high level: FULL decode of the needed streams ---
        auto q_high_decode_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(highPlans.size()); ++i) {
            const int b = highPlans[i].block;
            q_high_deData[b] = SZ_decompress_separated<T>(comp[b], compressedSize[b], dim_x, dim_y, dim_z);
        }
        auto q_high_decode_end = std::chrono::high_resolution_clock::now();
        double q_high_decode_time = std::chrono::duration_cast<std::chrono::nanoseconds>(q_high_decode_end - q_high_decode_start).count() * 1e-9;

        // --- high level: depreprocess restricted to ROI ---
        auto q_high_depre_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(highPlans.size()); ++i) {
            const auto& p = highPlans[i];
            depreprocess_block(p.block, q_high_deData[p.block], reconstructed_sub_0_ra, q_high_de_sub[p.block],
                               dim_x, dim_y, dim_z,
                               p.x.end + 1, p.y.end + 1, p.z.end + 1, p.x.begin, p.y.begin, p.z.begin);
        }
        auto q_high_depre_end = std::chrono::high_resolution_clock::now();
        double q_high_depre_time = std::chrono::duration_cast<std::chrono::nanoseconds>(q_high_depre_end - q_high_depre_start).count() * 1e-9;

        // --- assemble the queried ROI (merge restricted to query box) + validate ---
        auto q_merge_start = std::chrono::high_resolution_clock::now();
        const size_t qVol = static_cast<size_t>(query.x.end - query.x.begin + 1) *
                            static_cast<size_t>(query.y.end - query.y.begin + 1) *
                            static_cast<size_t>(query.z.end - query.z.begin + 1);
        std::vector<T> queryResult(qVol);
        size_t qout = 0;
        for (int z = query.z.begin; z <= query.z.end; ++z)
            for (int y = query.y.begin; y <= query.y.end; ++y)
                for (int x = query.x.begin; x <= query.x.end; ++x) {
                    const int sub_index = ((z & 1) << 2) | ((y & 1) << 1) | (x & 1);
                    const size_t src = (static_cast<size_t>(z / 2) * dim_y + y / 2) * dim_x + x / 2;
                    queryResult[qout++] = sub_index == 0
                        ? reconstructed_sub_0_ra[src] : q_high_de_sub[sub_index - 1][src];
                }
        auto q_merge_end = std::chrono::high_resolution_clock::now();
        double q_merge_time = std::chrono::duration_cast<std::chrono::nanoseconds>(q_merge_end - q_merge_start).count() * 1e-9;

        double q_max_abs_diff = 0.0;
        qout = 0;
        for (int z = query.z.begin; z <= query.z.end; ++z)
            for (int y = query.y.begin; y <= query.y.end; ++y)
                for (int x = query.x.begin; x <= query.x.end; ++x) {
                    const T full_v = reconstructed_full_data[(static_cast<size_t>(z) * full_dim_y + y) * full_dim_x + x];
                    q_max_abs_diff = std::max(q_max_abs_diff,
                        static_cast<double>(std::abs(queryResult[qout++] - full_v)));
                }

        const double q_end_to_end = base_decompress_time + q_low_decode_time + q_low_depre_time +
                                    q_ref_time + q_high_decode_time + q_high_depre_time + q_merge_time;
        std::cout << "Auto query high residual blocks decoded: " << blocks_to_string(highPlans) << std::endl;
        std::cout << "Auto query low residual blocks decoded: " << blocks_to_string(lowPlans) << std::endl;
        std::cout << "Auto query decompress breakdown: "
                  << "base decomp " << base_decompress_time
                  << ", low decode " << q_low_decode_time
                  << ", low deprocess " << q_low_depre_time
                  << ", low merge " << q_ref_time
                  << ", high decode " << q_high_decode_time
                  << ", high deprocess " << q_high_depre_time
                  << ", high merge " << q_merge_time << " sec" << std::endl;
        std::cout << "Auto query end-to-end decompress time: " << q_end_to_end << " sec" << std::endl;
        std::cout << "Auto query final ROI max abs diff: " << q_max_abs_diff << std::endl;

        for (int b = 0; b < 7; ++b) {
            delete[] q_low_deData[b];
            delete[] q_low_de_sub[b];
            delete[] q_high_deData[b];
            delete[] q_high_de_sub[b];
        }
        delete[] reconstructed_sub_0_ra;
    }

    const double compression_breakdown_sum =
        split_time + base_compress_time + low_buffer_time + low_compress_time +
        low_deprocess_compress_time + low_merge_compress_time + high_buffer_time +
        high_compress_time;
    const double global_compress_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(global_compress_end - global_compress_start).count() * 1e-9;
    const double decompress_breakdown_sum =
        base_decompress_time + low_decode_time + low_deprocess_time + low_merge_time +
        high_decode_time + high_deprocess_time + high_merge_time;
    const double global_decompress_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(global_decompress_end - global_decompress_start).count() * 1e-9;

    std::cout << "Compression breakdown: "
              << "split " << split_time
              << ", base compress " << base_compress_time
              << ", low buffer " << low_buffer_time
              << ", low encode " << low_compress_time
              << ", low deprocess " << low_deprocess_compress_time
              << ", low merge " << low_merge_compress_time
              << ", high buffer " << high_buffer_time
              << ", high encode " << high_compress_time
              << " sec" << std::endl;
    std::cout << "Compression breakdown sum: "
              << compression_breakdown_sum << " sec" << std::endl;
    std::cout << "Full decompress breakdown: "
              << "base decomp " << base_decompress_time
              << ", low decode " << low_decode_time
              << ", low deprocess " << low_deprocess_time
              << ", low merge " << low_merge_time
              << ", high decode " << high_decode_time
              << ", high deprocess " << high_deprocess_time
              << ", high merge " << high_merge_time
              << " sec" << std::endl;
    std::cout << "Full decompress breakdown sum: "
              << decompress_breakdown_sum << " sec" << std::endl;
    std::cout << "compress time is: " << std::fixed << std::setprecision(5)
              << compression_breakdown_sum << " sec" << std::endl;
    std::cout << "decompress time is: " << std::fixed << std::setprecision(5)
              << decompress_breakdown_sum << " sec" << std::endl;
    std::cout << "global compress time is: " << std::fixed << std::setprecision(5)
              << global_compress_time << " sec" << std::endl;
    std::cout << "global decompress time is: " << std::fixed << std::setprecision(5)
              << global_decompress_time << " sec" << std::endl;

    delete[] full_data;
    delete[] decompressed_data;
    delete[] reconstructed_sub_0;
    delete[] reconstructed_full_data;
    for (int i = 0; i < 8; ++i) {
        delete[] sub_block_data[i];
        delete[] low_block_data[i];
    }
    for (int i = 0; i < 7; ++i)
    {
        delete[] low_diff_data[i];
        delete[] low_deData[i];
        delete[] low_de_sub_block[i];
        delete[] diff_data[i];
        delete[] deData[i];
        delete[] de_sub_block[i];
    }

    return 0;
}

int main(int argc, char* argv[])
{
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    int type_arg = -1;
    if (std::string(argv[4]) == "-f" || std::string(argv[4]) == "-d") {
        type_arg = 4;
    } else if (argc >= 7 && (std::string(argv[6]) == "-f" || std::string(argv[6]) == "-d")) {
        type_arg = 6;
    }
    if (type_arg < 0) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string type_flag = argv[type_arg];
    if (type_flag == "-f") {
        return run_typed<float>(argc, argv);
    }
    if (type_flag == "-d") {
        return run_typed<double>(argc, argv);
    }

    std::cerr << "Invalid data type flag '" << type_flag << "'. Use -f or -d." << std::endl;
    return 1;
}
