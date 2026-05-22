#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <cfloat>
#include "sz.hpp"

const int full_dim_z = 512;
const int full_dim_y = 512;
const int full_dim_x = 512;
const int dim_z = 256;
const int dim_y = 256;
const int dim_x = 256;
const int low_dim_z = 128;
const int low_dim_y = 128;
const int low_dim_x = 128;
// const int full_dim_z = 1024;
// const int full_dim_y = 1024;
// const int full_dim_x = 1024;
// const int dim_z = 512;
// const int dim_y = 512;
// const int dim_x = 512;
// const int low_dim_z = 256;
// const int low_dim_y = 256;
// const int low_dim_x = 256;
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
        if (data[i] > high)
            high = data[i];
        if (data[i] < low)
            low = data[i];
    }
    double range = high-low;
    return range;
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

static inline float cubic_line_pred(float a, float b, float c, float d)
{
    return -(1.0f / 16.0f) * a + (9.0f / 16.0f) * b +
           (9.0f / 16.0f) * c - (1.0f / 16.0f) * d;
}

static inline float cubic_line_var(float a, float b, float c, float d)
{
    constexpr float eps = 1e-12f;
    const float c0 = a - 2.0f * b + c;
    const float c1 = b - 2.0f * c + d;
    return 0.5f * (c0 * c0 + c1 * c1) + eps;
}

static inline float blend2(float p0, float v0, float p1, float v1)
{
    return (p0 * v1 + p1 * v0) / (v0 + v1);
}

static inline float predict_12_nodynamic(int block, const float* ref, int idx, int dim_x, int dim_xy)
{
    // Use explicit 12-point truncated bicubic weights for blocks 2 (XY), 4 (XZ), 5 (YZ)
    if (block == 2) {
        const float w_core = 0.31640625f;   // 81/256
        const float w_edge = -0.033203125f; // -17/512
        float core_sum = ref[idx] + ref[idx + 1] + ref[idx + dim_x] + ref[idx + dim_x + 1];
        float edge_sum = ref[idx - 1] + ref[idx + 2] +
                         ref[idx + dim_x - 1] + ref[idx + dim_x + 2] +
                         ref[idx - dim_x] + ref[idx - dim_x + 1] +
                         ref[idx + 2 * dim_x] + ref[idx + 2 * dim_x + 1];
        return w_core * core_sum + w_edge * edge_sum;
    }

    if (block == 4) {
        const float w_core = 0.31640625f;
        const float w_edge = -0.033203125f;
        float core_sum = ref[idx] + ref[idx + 1] + ref[idx + dim_xy] + ref[idx + dim_xy + 1];
        float edge_sum = ref[idx - 1] + ref[idx + 2] +
                         ref[idx + dim_xy - 1] + ref[idx + dim_xy + 2] +
                         ref[idx - dim_xy] + ref[idx - dim_xy + 1] +
                         ref[idx + 2 * dim_xy] + ref[idx + 2 * dim_xy + 1];
        return w_core * core_sum + w_edge * edge_sum;
    }

    if (block == 5) {
        const float w_core = 0.31640625f;
        const float w_edge = -0.033203125f;
        float core_sum = ref[idx] + ref[idx + dim_x] + ref[idx + dim_xy] + ref[idx + dim_xy + dim_x];
        float edge_sum = ref[idx - dim_x] + ref[idx + 2 * dim_x] +
                         ref[idx + dim_xy - dim_x] + ref[idx + dim_xy + 2 * dim_x] +
                         ref[idx - dim_xy] + ref[idx - dim_xy + dim_x] +
                         ref[idx + 2 * dim_xy] + ref[idx + 2 * dim_xy + dim_x];
        return w_core * core_sum + w_edge * edge_sum;
    }

    // Fallback: average two 1D cubic predictions in remaining cases
    const int sy = dim_x;
    const int sz = dim_xy;
    const float hy0 = cubic_line_pred(ref[idx - sy], ref[idx], ref[idx + sy], ref[idx + 2 * sy]);
    const float hy1 = cubic_line_pred(ref[idx + sz - sy], ref[idx + sz],
                                      ref[idx + sz + sy], ref[idx + sz + 2 * sy]);

    const float hz0 = cubic_line_pred(ref[idx - sz], ref[idx], ref[idx + sz], ref[idx + 2 * sz]);
    const float hz1 = cubic_line_pred(ref[idx - sz + sy], ref[idx + sy],
                                      ref[idx + sz + sy], ref[idx + 2 * sz + sy]);

    return 0.25f * (hy0 + hy1 + hz0 + hz1);
}

//---------------------------------------------------------------------
// Preprocessing: Compute diff = sub_block - reference (sz_out_data)
// block index: 0 corresponds to "001", 1 to "010", 2 to "011", 3 to "100",
// 4 to "101", 5 to "110", 6 to "111"
//---------------------------------------------------------------------
void preprocess_block(int block, const float* sub_block, const float* ref, float* diff,
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
                            const float w_core = 0.31640625f;
                            const float w_edge = -0.033203125f;
                            float core_sum = ref[idx] + ref[idx + 1] + ref[idx + dim_x] + ref[idx + dim_x + 1];
                            float edge_sum = ref[idx - 1] + ref[idx + 2] +
                                             ref[idx + dim_x - 1] + ref[idx + dim_x + 2] +
                                             ref[idx - dim_x] + ref[idx - dim_x + 1] +
                                             ref[idx + 2 * dim_x] + ref[idx + 2 * dim_x + 1];
                            diff[idx] = sub_block[idx] - (w_core * core_sum + w_edge * edge_sum);
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
                            const float w_core = 0.31640625f;
                            const float w_edge = -0.033203125f;
                            float core_sum = ref[idx] + ref[idx + 1] + ref[idx + dim_xy] + ref[idx + dim_xy + 1];
                            float edge_sum = ref[idx - 1] + ref[idx + 2] +
                                             ref[idx + dim_xy - 1] + ref[idx + dim_xy + 2] +
                                             ref[idx - dim_xy] + ref[idx - dim_xy + 1] +
                                             ref[idx + 2 * dim_xy] + ref[idx + 2 * dim_xy + 1];
                            diff[idx] = sub_block[idx] - (w_core * core_sum + w_edge * edge_sum);
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
                            const float w_core = 0.31640625f;
                            const float w_edge = -0.033203125f;
                            float core_sum = ref[idx] + ref[idx + dim_x] + ref[idx + dim_xy] + ref[idx + dim_xy + dim_x];
                            float edge_sum = ref[idx - dim_x] + ref[idx + 2 * dim_x] +
                                             ref[idx + dim_xy - dim_x] + ref[idx + dim_xy + 2 * dim_x] +
                                             ref[idx - dim_xy] + ref[idx - dim_xy + dim_x] +
                                             ref[idx + 2 * dim_xy] + ref[idx + 2 * dim_xy + dim_x];
                            diff[idx] = sub_block[idx] - (w_core * core_sum + w_edge * edge_sum);
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
void depreprocess_block(int block, const float* deData, const float* ref, float* de_sub,
                          int dim_x, int dim_y, int dim_z)
{
    int dim_xy= dim_x * dim_y;
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
                            const float w_core = 0.31640625f;
                            const float w_edge = -0.033203125f;
                            float core_sum = ref[idx] + ref[idx + 1] + ref[idx + dim_x] + ref[idx + dim_x + 1];
                            float edge_sum = ref[idx - 1] + ref[idx + 2] +
                                             ref[idx + dim_x - 1] + ref[idx + dim_x + 2] +
                                             ref[idx - dim_x] + ref[idx - dim_x + 1] +
                                             ref[idx + 2 * dim_x] + ref[idx + 2 * dim_x + 1];
                            de_sub[idx] = deData[idx] + (w_core * core_sum + w_edge * edge_sum);
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
                            const float w_core = 0.31640625f;
                            const float w_edge = -0.033203125f;
                            float core_sum = ref[idx] + ref[idx + 1] + ref[idx + dim_xy] + ref[idx + dim_xy + 1];
                            float edge_sum = ref[idx - 1] + ref[idx + 2] +
                                             ref[idx + dim_xy - 1] + ref[idx + dim_xy + 2] +
                                             ref[idx - dim_xy] + ref[idx - dim_xy + 1] +
                                             ref[idx + 2 * dim_xy] + ref[idx + 2 * dim_xy + 1];
                            de_sub[idx] = deData[idx] + (w_core * core_sum + w_edge * edge_sum);
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
                            const float w_core = 0.31640625f;
                            const float w_edge = -0.033203125f;
                            float core_sum = ref[idx] + ref[idx + dim_x] + ref[idx + dim_xy] + ref[idx + dim_xy + dim_x];
                            float edge_sum = ref[idx - dim_x] + ref[idx + 2 * dim_x] +
                                             ref[idx + dim_xy - dim_x] + ref[idx + dim_xy + 2 * dim_x] +
                                             ref[idx - dim_xy] + ref[idx - dim_xy + dim_x] +
                                             ref[idx + 2 * dim_xy] + ref[idx + 2 * dim_xy + dim_x];
                            de_sub[idx] = deData[idx] + (w_core * core_sum + w_edge * edge_sum);
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
    conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = eb;
    char* compressedData = SZ_compress<float>(conf, oriData, outSize);
    // Note: Returning original data (as in your original code)
    return compressedData;
}

float* SZ_decompress4De(char* compressedData, size_t outSize, size_t blksize_x, size_t blksize_y, size_t blksize_z)
{
    SZ3::Config conf(blksize_z, blksize_y, blksize_x);
    conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
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
    if (!file)
    {
        std::cerr << "Failed to open file for reading: " << filepath << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(data), dataSize * sizeof(float));
    if (!file)
    {
        std::cerr << "Failed to read data from file: " << filepath << std::endl;
        return false;
    }
    file.close();
    return true;
}

bool writeBinaryData(const std::string& filepath, const float* data, size_t dataSize)
{
    std::ofstream file(filepath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    file.write(reinterpret_cast<const char*>(data), dataSize * sizeof(float));
    if (!file)
    {
        std::cerr << "Failed to write data to file: " << filepath << std::endl;
        return false;
    }
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

    std::string full_file_path = "/N/u/daocwang/BigRed200/data/SDRBENCH-EXASKY-NYX-512x512x512/baryon_density.f32";
    // std::string full_file_path = "/N/u/daocwang/BigRed200/data/magnetic_reconnection_512x512x512_float32.raw";
    // std::string full_file_path = "/N/u/daocwang/BigRed200/data/miranda_1024x1024x1024_float32.raw";
    size_t full_size = full_dim_z * full_dim_y * full_dim_x;
    float* full_data = new float[full_size];
    if (!readBinaryData(full_file_path, full_data, full_size))
    {
        delete[] full_data;
        return 1;
    }

    auto split_start = std::chrono::high_resolution_clock::now();
    // Allocate 8 sub-blocks (each 256^3)
    float* sub_block_data[8];
    #pragma omp parallel for 
    for (int i = 0; i < 8; ++i)
        sub_block_data[i] = new float[dim_z * dim_x * dim_y];

    // Slice full_data into 8 sub-blocks using bit masking.
    slice_full_data(full_data, sub_block_data,full_dim_x,full_dim_y,full_dim_z);

    float* low_block_data[8];
    #pragma omp parallel for 
    for (int i = 0; i < 8; ++i)
        low_block_data[i] = new float[low_dim_z * low_dim_y * low_dim_x];

    slice_full_data(sub_block_data[0], low_block_data,dim_x,dim_y,dim_z);

    auto split_end = std::chrono::high_resolution_clock::now();
    double sz_time_taken_split = std::chrono::duration_cast<std::chrono::nanoseconds>(split_end - split_start).count() * 1e-9;
    // std::cout << "Time taken by split is: " << std::fixed << std::setprecision(5)
    //         << sz_time_taken_split << " sec" << std::endl;

    // Use sub_block_data[0] as the reference (sz_out_data)
    // and sub_block_data[1] ... sub_block_data[7] as the seven data blocks.
    double eb = atof(argv[1]);
    
    size_t allSize = 0;
    size_t szcompressedSize;
    auto sz_start = std::chrono::high_resolution_clock::now();
    char* tmp = SZ_compress4De(low_block_data[0], low_dim_x, low_dim_y, low_dim_z, eb, szcompressedSize);
    // std::cout << "outSize: " << szcompressedSize << std::endl;
    allSize += szcompressedSize;
    auto sz_end = std::chrono::high_resolution_clock::now();
    double sz_time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(sz_end - sz_start).count() * 1e-9;
    // std::cout << "Time taken by sz compression is: " << std::fixed << std::setprecision(5)
    //           << sz_time_taken << " sec" << std::endl;

    float* decompressed_data = nullptr;
    double sz_time_taken_decompress = 0.0;

    char* low_comp[7];
    float* low_diff_data[7];
    float* low_deData[7];
    float* low_de_sub_block[7];
    size_t low_compressedSize[7];
    #pragma omp parallel for 
    for (int i = 0; i < 7; ++i)
    {
        low_diff_data[i]    = new float[low_dim_z * low_dim_y * low_dim_x];
        low_deData[i]       = new float[low_dim_z * low_dim_y * low_dim_x];
        low_de_sub_block[i] = new float[low_dim_z * low_dim_y * low_dim_x];
    }

    auto low_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:allSize)
    for (int block = 0; block < 7; ++block)
    {
        // For block i, use sub_block_data[i+1] as the input.
        preprocess_block(block, low_block_data[block+1], low_block_data[0], low_diff_data[block],
                           low_dim_x, low_dim_y, low_dim_z);
        low_comp[block] = SZ_compress(low_diff_data[block], low_dim_x, low_dim_y, low_dim_z, 2.5 * eb, low_compressedSize[block]);
        allSize += low_compressedSize[block];
        depreprocess_block(block, low_diff_data[block], low_block_data[0], low_de_sub_block[block],
                           low_dim_x, low_dim_y, low_dim_z);
        // std::cout << "outSize: " << low_compressedSize[block] << std::endl;
    }
    auto low_end = std::chrono::high_resolution_clock::now();
    double low_time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(low_end - low_start).count() * 1e-9;
    // std::cout << "Time taken by low compression is: " << std::fixed << std::setprecision(5)
    //           << low_time_taken << " sec" << std::endl;

    auto reconstructed_low_start = std::chrono::high_resolution_clock::now();
    float* reconstructed_sub_0 = new float[dim_z * dim_x * dim_y];
    float* all_low_blocks[8] = {
        low_block_data[0],          // vs ori sub_block_0
        low_de_sub_block[0],      // sub_block_1
        low_de_sub_block[1],      // sub_block_2
        low_de_sub_block[2],      // sub_block_3
        low_de_sub_block[3],      // sub_block_4
        low_de_sub_block[4],      // sub_block_5
        low_de_sub_block[5],      // sub_block_6
        low_de_sub_block[6]       // sub_block_7
    };
    merge_sub_blocks_to_full(all_low_blocks, reconstructed_sub_0, low_dim_x, low_dim_y, low_dim_z);
    // writeBinaryData("tur-mid-2.raw", reconstructed_sub_0, full_dim_z/2 * full_dim_y/2 * full_dim_x/2);
    auto reconstructed_low_end = std::chrono::high_resolution_clock::now();
    double time_taken_reconstructed_low = std::chrono::duration_cast<std::chrono::nanoseconds>(reconstructed_low_end - reconstructed_low_start).count() * 1e-9;
    // std::cout << "Time taken by reconstructed_low is: " << std::fixed << std::setprecision(5)
    //           << time_taken_reconstructed_low << " sec" << std::endl;
    

    // Allocate buffers for diff, decompressed, and reconstructed data for 7 blocks.
    char* comp[7];
    float* diff_data[7];
    float* deData[7];
    float* de_sub_block[7];
    size_t compressedSize[7];
    for (int i = 0; i < 7; ++i)
    {
        diff_data[i]    = new float[dim_z * dim_x * dim_y];
        deData[i]       = new float[dim_z * dim_x * dim_y];
        de_sub_block[i] = new float[dim_z * dim_x * dim_y];
    }

    auto start = std::chrono::high_resolution_clock::now();
    // Pre-process each of the 7 blocks in parallel.
    #pragma omp parallel for reduction(+:allSize)
    for (int block = 0; block < 7; ++block)
    {
        // For block i, use sub_block_data[i+1] as the input.
        preprocess_block(block, sub_block_data[block+1], reconstructed_sub_0, diff_data[block],
                           dim_x, dim_y, dim_z);
        comp[block] = SZ_compress(diff_data[block], dim_x, dim_y, dim_z, 6.25 * eb, compressedSize[block]);
        allSize += compressedSize[block];
        // std::cout << "outSize: " << compressedSize[block] << std::endl;
    }
    double original_size = (double)full_dim_z * full_dim_y * full_dim_x * sizeof(float);
    double CR = original_size / (double)allSize;
    std::cout << "CR: " << CR << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;
    double global_compress_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - split_start).count() * 1e-9;
    // std::cout << "Time taken by compression is: " << std::fixed << std::setprecision(5)
    //           << time_taken << " sec" << std::endl;

    auto global_decompress_start = std::chrono::high_resolution_clock::now();
    decompressed_data = SZ_decompress4De(tmp, szcompressedSize, low_dim_x, low_dim_y, low_dim_z);
    auto sz_deend = std::chrono::high_resolution_clock::now();
    sz_time_taken_decompress = std::chrono::duration_cast<std::chrono::nanoseconds>(sz_deend - global_decompress_start).count() * 1e-9;

    auto low_decompress_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for 
    for (int block = 0; block < 7; ++block)
    {
        low_deData[block] = SZ_decompress_separated(low_comp[block], low_compressedSize[block], low_dim_x, low_dim_y, low_dim_z);
    }
    auto low_decompress_end_sz = std::chrono::high_resolution_clock::now();
    double low_time_taken_decompress_sz = std::chrono::duration_cast<std::chrono::nanoseconds>(low_decompress_end_sz - low_decompress_start).count() * 1e-9;
    // std::cout << "Time taken by low_decompression_sz is: " << std::fixed << std::setprecision(5)
    //           << low_time_taken_decompress_sz << " sec" << std::endl;
    #pragma omp parallel for 
    for (int block = 0; block < 7; ++block)
    {
        // std::cout << "De-preprocessing block " << block + 1 << std::endl;
        depreprocess_block(block, low_deData[block], decompressed_data, low_de_sub_block[block],
                           low_dim_x, low_dim_y, low_dim_z);
    }
    auto low_decompress_end = std::chrono::high_resolution_clock::now();
    double low_time_taken_decompress = std::chrono::duration_cast<std::chrono::nanoseconds>(low_decompress_end - low_decompress_start).count() * 1e-9;
    // std::cout << "Time taken by low_decompression is: " << std::fixed << std::setprecision(5)
    //           << low_time_taken_decompress << " sec" << std::endl;

    auto reconstructed_low_decompress_start = std::chrono::high_resolution_clock::now();
    all_low_blocks[0] = decompressed_data;
    merge_sub_blocks_to_full(all_low_blocks, reconstructed_sub_0, low_dim_x, low_dim_y, low_dim_z);
    auto reconstructed_low_decompress_end = std::chrono::high_resolution_clock::now();
    double time_taken_reconstructed_low_decompress = std::chrono::duration_cast<std::chrono::nanoseconds>(reconstructed_low_decompress_end - reconstructed_low_decompress_start).count() * 1e-9;

    auto decompress_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for 
    for (int block = 0; block < 7; ++block)
    {
        deData[block] = SZ_decompress_separated(comp[block], compressedSize[block], dim_x, dim_y, dim_z);
    }
    auto decompress_end_sz = std::chrono::high_resolution_clock::now();
    double time_taken_decompress_sz = std::chrono::duration_cast<std::chrono::nanoseconds>(decompress_end_sz - decompress_start).count() * 1e-9;
    // std::cout << "Time taken by decompression_sz is: " << std::fixed << std::setprecision(5)
    //           << time_taken_decompress_sz << " sec" << std::endl;
    #pragma omp parallel for 
    for (int block = 0; block < 7; ++block)
    {
        // std::cout << "De-preprocessing block " << block + 1 << std::endl;
        depreprocess_block(block, deData[block], reconstructed_sub_0, de_sub_block[block],
                           dim_x, dim_y, dim_z);
    }
    auto decompress_end = std::chrono::high_resolution_clock::now();
    double time_taken_decompress = std::chrono::duration_cast<std::chrono::nanoseconds>(decompress_end - decompress_start).count() * 1e-9;
    // std::cout << "Time taken by decompression is: " << std::fixed << std::setprecision(5)
    //           << time_taken_decompress << " sec" << std::endl;

    auto reconstructed_full_start = std::chrono::high_resolution_clock::now();
    float* reconstructed_full_data = new float[full_dim_z * full_dim_y * full_dim_x];
    float* all_sub_blocks[8] = {
        reconstructed_sub_0,          // vs ori sub_block_0
        de_sub_block[0],      // sub_block_1
        de_sub_block[1],      // sub_block_2
        de_sub_block[2],      // sub_block_3
        de_sub_block[3],      // sub_block_4
        de_sub_block[4],      // sub_block_5
        de_sub_block[5],      // sub_block_6
        de_sub_block[6]       // sub_block_7
    };
    merge_sub_blocks_to_full(all_sub_blocks, reconstructed_full_data, dim_x, dim_y, dim_z);
    auto reconstructed_full_end = std::chrono::high_resolution_clock::now();
    double time_taken_reconstructed_full = std::chrono::duration_cast<std::chrono::nanoseconds>(reconstructed_full_end - reconstructed_full_start).count() * 1e-9;
    double global_decompress_time = std::chrono::duration_cast<std::chrono::nanoseconds>(reconstructed_full_end - global_decompress_start).count() * 1e-9;
    // std::cout << "Time taken by reconstructed_full is: " << std::fixed << std::setprecision(5)
    //           << time_taken_reconstructed_full << " sec" << std::endl;

    // writeBinaryData("ours.raw", reconstructed_full_data, full_dim_z * full_dim_y * full_dim_x);

    double mse_full = 0.0;
    for (size_t i = 0; i < full_dim_z * full_dim_y * full_dim_x; ++i) {
        double diff = full_data[i] - reconstructed_full_data[i];
        mse_full += diff * diff;
    }
    mse_full /= (full_dim_z * full_dim_y * full_dim_x);
    double range_full = computeRange(full_data, full_dim_z * full_dim_y * full_dim_x);
    double psnr_full = 20 * log10(range_full) - 10 * log10(mse_full);
    std::cout << "Global PSNR: " << psnr_full << std::endl;

    // //Write out the reconstructed sub-blocks.
    // std::string de_sub_block_paths[7] = {
    //     "/N/u/daocwang/BigRed200/data/de_sub_block_1.bin",
    //     "/N/u/daocwang/BigRed200/data/de_sub_block_2.bin",
    //     "/N/u/daocwang/BigRed200/data/de_sub_block_3.bin",
    //     "/N/u/daocwang/BigRed200/data/de_sub_block_4.bin",
    //     "/N/u/daocwang/BigRed200/data/de_sub_block_5.bin",
    //     "/N/u/daocwang/BigRed200/data/de_sub_block_6.bin",
    //     "/N/u/daocwang/BigRed200/data/de_sub_block_7.bin"
    // };
    // for (int block = 0; block < 7; ++block)
    // {
    //     if (!writeBinaryData(de_sub_block_paths[block], de_sub_block[block], dim_z * dim_xy))
    //         return 1;
    // }

    // writeBinaryData("/N/u/daocwang/BigRed200/stream/sz.out_0", reconstructed_sub_0, dim_z * dim_xy);

    std::cout << "compress time is: " << std::fixed << std::setprecision(5)
              << sz_time_taken_split + sz_time_taken + low_time_taken + time_taken_reconstructed_low + time_taken << " sec" << std::endl;
    
    std::cout << "decompress time is: " << std::fixed << std::setprecision(5)
              << sz_time_taken_decompress + low_time_taken_decompress + time_taken_reconstructed_low_decompress + time_taken_decompress + time_taken_reconstructed_full << " sec" << std::endl;

    std::cout << "global compress time is: " << std::fixed << std::setprecision(5)
              << global_compress_time << " sec" << std::endl;

    std::cout << "global decompress time is: " << std::fixed << std::setprecision(5)
              << global_decompress_time << " sec" << std::endl;

    // (Free allocated memory as needed.)
    delete[] full_data;
    delete[] decompressed_data;
    for (int i = 0; i < 8; ++i)
        delete[] sub_block_data[i];
    for (int i = 0; i < 7; ++i)
    {
        delete[] diff_data[i];
        delete[] deData[i];
        delete[] de_sub_block[i];
    }

    return 0;
}
