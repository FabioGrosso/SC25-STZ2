#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <cfloat>
#include <algorithm>
#include <cmath>
#include <set>
#include "SZ3/encoder/HuffmanEncoder.hpp"
#include "SZ3/lossless/Lossless_zstd.hpp"
#include "SZ3/quantizer/LinearQuantizer.hpp"
#include "SZ3/utils/MemoryUtil.hpp"
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
const int roi_full_dim = 64;
const int roi_high_dim = roi_full_dim / 2;
const int roi_low_dim = roi_high_dim / 2;
const size_t marker_table_bytes_per_stream = 512 * sizeof(uint64_t);
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

static inline float blend4(float p0, float v0, float p1, float v1,
                           float p2, float v2, float p3, float v3)
{
    const float w0 = 1.0f / v0;
    const float w1 = 1.0f / v1;
    const float w2 = 1.0f / v2;
    const float w3 = 1.0f / v3;
    return (w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3) / (w0 + w1 + w2 + w3);
}

static inline float predict_dynamic_lines(int block, const float* ref, int idx,
                                          int dim_x, int dim_xy)
{
    if (block == 2) {
        const float p0 = cubic_line_pred(ref[idx - dim_x - 1], ref[idx], ref[idx + dim_x + 1],
                                         ref[idx + 2 * dim_x + 2]);
        const float v0 = cubic_line_var(ref[idx - dim_x - 1], ref[idx], ref[idx + dim_x + 1],
                                        ref[idx + 2 * dim_x + 2]);
        const float p1 = cubic_line_pred(ref[idx - dim_x + 2], ref[idx + 1], ref[idx + dim_x],
                                         ref[idx + 2 * dim_x - 1]);
        const float v1 = cubic_line_var(ref[idx - dim_x + 2], ref[idx + 1], ref[idx + dim_x],
                                        ref[idx + 2 * dim_x - 1]);
        return blend2(p0, v0, p1, v1);
    }

    if (block == 4) {
        const float p0 = cubic_line_pred(ref[idx - dim_xy - 1], ref[idx], ref[idx + dim_xy + 1],
                                         ref[idx + 2 * dim_xy + 2]);
        const float v0 = cubic_line_var(ref[idx - dim_xy - 1], ref[idx], ref[idx + dim_xy + 1],
                                        ref[idx + 2 * dim_xy + 2]);
        const float p1 = cubic_line_pred(ref[idx - dim_xy + 2], ref[idx + 1], ref[idx + dim_xy],
                                         ref[idx + 2 * dim_xy - 1]);
        const float v1 = cubic_line_var(ref[idx - dim_xy + 2], ref[idx + 1], ref[idx + dim_xy],
                                        ref[idx + 2 * dim_xy - 1]);
        return blend2(p0, v0, p1, v1);
    }

    if (block == 5) {
        const float p0 = cubic_line_pred(ref[idx - dim_xy - dim_x], ref[idx],
                                         ref[idx + dim_xy + dim_x],
                                         ref[idx + 2 * dim_xy + 2 * dim_x]);
        const float v0 = cubic_line_var(ref[idx - dim_xy - dim_x], ref[idx],
                                        ref[idx + dim_xy + dim_x],
                                        ref[idx + 2 * dim_xy + 2 * dim_x]);
        const float p1 = cubic_line_pred(ref[idx - dim_xy + 2 * dim_x], ref[idx + dim_x],
                                         ref[idx + dim_xy],
                                         ref[idx + 2 * dim_xy - dim_x]);
        const float v1 = cubic_line_var(ref[idx - dim_xy + 2 * dim_x], ref[idx + dim_x],
                                        ref[idx + dim_xy],
                                        ref[idx + 2 * dim_xy - dim_x]);
        return blend2(p0, v0, p1, v1);
    }

    const float p0 = cubic_line_pred(ref[idx - dim_xy - dim_x - 1], ref[idx],
                                     ref[idx + dim_xy + dim_x + 1],
                                     ref[idx + 2 * dim_xy + 2 * dim_x + 2]);
    const float v0 = cubic_line_var(ref[idx - dim_xy - dim_x - 1], ref[idx],
                                    ref[idx + dim_xy + dim_x + 1],
                                    ref[idx + 2 * dim_xy + 2 * dim_x + 2]);
    const float p1 = cubic_line_pred(ref[idx - dim_xy - dim_x + 2], ref[idx + 1],
                                     ref[idx + dim_xy + dim_x],
                                     ref[idx + 2 * dim_xy + 2 * dim_x - 1]);
    const float v1 = cubic_line_var(ref[idx - dim_xy - dim_x + 2], ref[idx + 1],
                                    ref[idx + dim_xy + dim_x],
                                    ref[idx + 2 * dim_xy + 2 * dim_x - 1]);
    const float p2 = cubic_line_pred(ref[idx - dim_xy + 2 * dim_x - 1], ref[idx + dim_x],
                                     ref[idx + dim_xy + 1],
                                     ref[idx + 2 * dim_xy - dim_x + 2]);
    const float v2 = cubic_line_var(ref[idx - dim_xy + 2 * dim_x - 1], ref[idx + dim_x],
                                    ref[idx + dim_xy + 1],
                                    ref[idx + 2 * dim_xy - dim_x + 2]);
    const float p3 = cubic_line_pred(ref[idx + 2 * dim_xy - dim_x - 1], ref[idx + dim_xy],
                                     ref[idx + dim_x + 1],
                                     ref[idx - dim_xy + 2 * dim_x + 2]);
    const float v3 = cubic_line_var(ref[idx + 2 * dim_xy - dim_x - 1], ref[idx + dim_xy],
                                    ref[idx + dim_x + 1],
                                    ref[idx - dim_xy + 2 * dim_x + 2]);
    return blend4(p0, v0, p1, v1, p2, v2, p3, v3);
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
                            diff[idx] = sub_block[idx] - predict_dynamic_lines(block, ref, idx, dim_x, dim_xy);
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
                            diff[idx] = sub_block[idx] - predict_dynamic_lines(block, ref, idx, dim_x, dim_xy);
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
                            diff[idx] = sub_block[idx] - predict_dynamic_lines(block, ref, idx, dim_x, dim_xy);
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
                            diff[idx] = sub_block[idx] - predict_dynamic_lines(block, ref, idx, dim_x, dim_xy);
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
                            de_sub[idx] = deData[idx] + predict_dynamic_lines(block, ref, idx, dim_x, dim_xy);
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
                            de_sub[idx] = deData[idx] + predict_dynamic_lines(block, ref, idx, dim_x, dim_xy);
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
                            de_sub[idx] = deData[idx] + predict_dynamic_lines(block, ref, idx, dim_x, dim_xy);
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
                            de_sub[idx] = deData[idx] + predict_dynamic_lines(block, ref, idx, dim_x, dim_xy);
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

static inline size_t chunk_index_3d(size_t cx, size_t cy, size_t cz, size_t chunks_per_dim)
{
    return (cz * chunks_per_dim + cy) * chunks_per_dim + cx;
}

std::vector<size_t> chunks_for_z_slice(size_t block_dim, size_t chunk_dim, size_t z)
{
    const size_t chunks_per_dim = block_dim / chunk_dim;
    const size_t cz = z / chunk_dim;
    std::vector<size_t> chunks;
    chunks.reserve(chunks_per_dim * chunks_per_dim);
    for (size_t cy = 0; cy < chunks_per_dim; ++cy) {
        for (size_t cx = 0; cx < chunks_per_dim; ++cx) {
            chunks.push_back(chunk_index_3d(cx, cy, cz, chunks_per_dim));
        }
    }
    return chunks;
}

struct AxisRange {
    int begin = 0;
    int end = -1;
};

struct QueryBox {
    AxisRange x;
    AxisRange y;
    AxisRange z;
    bool is_slice = false;
};

struct BlockAccessPlan {
    int block = 0;
    AxisRange x;
    AxisRange y;
    AxisRange z;
    std::vector<size_t> chunks;
};

AxisRange normalize_axis(int a, int b, int dim, bool& is_slice)
{
    if (a == b) {
        is_slice = true;
        const int v = std::max(0, std::min(a, dim - 1));
        return {v, v};
    }
    int lo = std::max(0, std::min(a, b));
    int hi = std::min(dim - 1, std::max(a, b));
    if (hi < lo) {
        hi = lo;
    }
    return {lo, hi};
}

QueryBox make_query(int ax, int ay, int az, int bx, int by, int bz)
{
    QueryBox q;
    q.x = normalize_axis(ax, bx, full_dim_x, q.is_slice);
    q.y = normalize_axis(ay, by, full_dim_y, q.is_slice);
    q.z = normalize_axis(az, bz, full_dim_z, q.is_slice);
    return q;
}

AxisRange range_for_parity(AxisRange fullRange, int parity)
{
    int first = fullRange.begin;
    if ((first & 1) != parity) {
        first++;
    }
    if (first > fullRange.end) {
        return {1, 0};
    }
    int last = fullRange.end;
    if ((last & 1) != parity) {
        last--;
    }
    return {first / 2, last / 2};
}

std::vector<size_t> chunks_for_box(size_t block_dim, size_t chunk_dim, AxisRange x, AxisRange y, AxisRange z)
{
    if (x.begin > x.end || y.begin > y.end || z.begin > z.end) {
        return {};
    }
    const size_t chunks_per_dim = block_dim / chunk_dim;
    const int cx0 = std::max(0, x.begin / static_cast<int>(chunk_dim));
    const int cy0 = std::max(0, y.begin / static_cast<int>(chunk_dim));
    const int cz0 = std::max(0, z.begin / static_cast<int>(chunk_dim));
    const int cx1 = std::min(static_cast<int>(chunks_per_dim) - 1, x.end / static_cast<int>(chunk_dim));
    const int cy1 = std::min(static_cast<int>(chunks_per_dim) - 1, y.end / static_cast<int>(chunk_dim));
    const int cz1 = std::min(static_cast<int>(chunks_per_dim) - 1, z.end / static_cast<int>(chunk_dim));
    std::vector<size_t> chunks;
    chunks.reserve((cx1 - cx0 + 1) * (cy1 - cy0 + 1) * (cz1 - cz0 + 1));
    for (int cz = cz0; cz <= cz1; ++cz) {
        for (int cy = cy0; cy <= cy1; ++cy) {
            for (int cx = cx0; cx <= cx1; ++cx) {
                chunks.push_back(chunk_index_3d(cx, cy, cz, chunks_per_dim));
            }
        }
    }
    return chunks;
}

std::vector<BlockAccessPlan> build_residual_plan(const QueryBox& q, int block_dim, int chunk_dim)
{
    std::vector<BlockAccessPlan> plans;
    for (int block = 0; block < 7; ++block) {
        const int sub_index = block + 1;
        const int px = sub_index & 1;
        const int py = (sub_index >> 1) & 1;
        const int pz = (sub_index >> 2) & 1;
        BlockAccessPlan p;
        p.block = block;
        p.x = range_for_parity(q.x, px);
        p.y = range_for_parity(q.y, py);
        p.z = range_for_parity(q.z, pz);
        if (p.x.begin > p.x.end || p.y.begin > p.y.end || p.z.begin > p.z.end) {
            continue;
        }
        p.chunks = chunks_for_box(block_dim, chunk_dim, p.x, p.y, p.z);
        plans.push_back(std::move(p));
    }
    return plans;
}

static inline void merge_axis(AxisRange& dst, AxisRange src)
{
    if (src.begin > src.end) {
        return;
    }
    if (dst.begin > dst.end) {
        dst = src;
        return;
    }
    dst.begin = std::min(dst.begin, src.begin);
    dst.end = std::max(dst.end, src.end);
}

AxisRange ref_range_for_used_axis(AxisRange target, int dim)
{
    if (target.begin > target.end) {
        return target;
    }

    int lo = target.begin;
    int hi = target.end;

    const int interior_begin = std::max(target.begin, 1);
    const int interior_end = std::min(target.end, dim - 3);
    if (interior_begin <= interior_end) {
        lo = std::min(lo, interior_begin - 1);
        hi = std::max(hi, interior_end + 2);
    }

    const int nonlast_begin = std::max(target.begin, 0);
    const int nonlast_end = std::min(target.end, dim - 2);
    if (nonlast_begin <= nonlast_end) {
        hi = std::max(hi, nonlast_end + 1);
    }

    return {std::max(0, lo), std::min(dim - 1, hi)};
}

QueryBox build_high_ref_region(const BlockAccessPlan& highPlan)
{
    const int sub_index = highPlan.block + 1;
    QueryBox q;
    q.x = (sub_index & 1) ? ref_range_for_used_axis(highPlan.x, dim_x) : highPlan.x;
    q.y = (sub_index & 2) ? ref_range_for_used_axis(highPlan.y, dim_y) : highPlan.y;
    q.z = (sub_index & 4) ? ref_range_for_used_axis(highPlan.z, dim_z) : highPlan.z;
    q.is_slice = (q.x.begin == q.x.end) || (q.y.begin == q.y.end) || (q.z.begin == q.z.end);
    return q;
}

std::vector<QueryBox> build_high_ref_regions(const std::vector<BlockAccessPlan>& highPlans)
{
    std::vector<QueryBox> regions;
    regions.reserve(highPlans.size());
    for (const auto& p : highPlans) {
        regions.push_back(build_high_ref_region(p));
    }
    return regions;
}

QueryBox build_query_sub0_region(const QueryBox& q)
{
    QueryBox sub0;
    sub0.x = range_for_parity(q.x, 0);
    sub0.y = range_for_parity(q.y, 0);
    sub0.z = range_for_parity(q.z, 0);
    sub0.is_slice = (sub0.x.begin == sub0.x.end) ||
                    (sub0.y.begin == sub0.y.end) ||
                    (sub0.z.begin == sub0.z.end);
    return sub0;
}

std::vector<BlockAccessPlan> build_low_residual_plan(const std::vector<QueryBox>& highRefRegions)
{
    std::vector<std::set<size_t>> chunk_sets(7);
    std::vector<BlockAccessPlan> accum(7);
    for (int block = 0; block < 7; ++block) {
        accum[block].block = block;
        accum[block].x = accum[block].y = accum[block].z = {1, 0};
    }

    for (const auto& highRefQuery : highRefRegions) {
        if (highRefQuery.x.begin > highRefQuery.x.end ||
            highRefQuery.y.begin > highRefQuery.y.end ||
            highRefQuery.z.begin > highRefQuery.z.end) {
            continue;
        }
        for (int block = 0; block < 7; ++block) {
            const int sub_index = block + 1;
            const int px = sub_index & 1;
            const int py = (sub_index >> 1) & 1;
            const int pz = (sub_index >> 2) & 1;
            BlockAccessPlan p;
            p.block = block;
            p.x = range_for_parity(highRefQuery.x, px);
            p.y = range_for_parity(highRefQuery.y, py);
            p.z = range_for_parity(highRefQuery.z, pz);
            if (p.x.begin > p.x.end || p.y.begin > p.y.end || p.z.begin > p.z.end) {
                continue;
            }
            const auto chunks = chunks_for_box(low_dim_x, roi_low_dim, p.x, p.y, p.z);
            chunk_sets[block].insert(chunks.begin(), chunks.end());
            merge_axis(accum[block].x, p.x);
            merge_axis(accum[block].y, p.y);
            merge_axis(accum[block].z, p.z);
        }
    }

    std::vector<BlockAccessPlan> plans;
    for (int block = 0; block < 7; ++block) {
        if (chunk_sets[block].empty()) {
            continue;
        }
        accum[block].chunks.assign(chunk_sets[block].begin(), chunk_sets[block].end());
        plans.push_back(std::move(accum[block]));
    }
    return plans;
}

std::string blocks_to_string(const std::vector<BlockAccessPlan>& plans)
{
    std::string s;
    for (const auto& p : plans) {
        s += std::to_string(p.block) + " ";
    }
    return s.empty() ? "(none)" : s;
}

float* SZ_decompress_nopred_marked_chunks(char* compressedData, size_t outSize,
                                           size_t blksize_x, size_t blksize_y, size_t blksize_z,
                                           size_t chunk_dim, const std::vector<size_t>& rowMajorChunks,
                                           size_t& decodedSymbols, size_t& unpredZeros)
{
    SZ3::uchar* buffer = nullptr;
    size_t bufferSize = 0;
    SZ3::Lossless_zstd lossless;
    SZ3::Config conf(blksize_z, blksize_y, blksize_x);
    const SZ3::uchar* cmp_pos = reinterpret_cast<const SZ3::uchar*>(compressedData) + conf.size_est();
    lossless.decompress(cmp_pos, outSize - conf.size_est(), buffer, bufferSize);

    const SZ3::uchar* buffer_pos = buffer;
    size_t remaining_length = bufferSize;
    SZ3::LinearQuantizer<float> quantizer;
    quantizer.load(buffer_pos, remaining_length);

    SZ3::HuffmanEncoder<int> encoder;
    encoder.load(buffer_pos, remaining_length);
    size_t quant_inds_size = 0;
    SZ3::read(quant_inds_size, buffer_pos, remaining_length);
    std::vector<int> quant_inds = encoder.decode_marked_chunks(buffer_pos, quant_inds_size, rowMajorChunks);
    encoder.postprocess_decode();

    float* deData = new float[blksize_x * blksize_y * blksize_z]();
    const size_t chunks_per_dim = blksize_x / chunk_dim;
    const size_t chunk_volume = chunk_dim * chunk_dim * chunk_dim;
    decodedSymbols = quant_inds.size();
    unpredZeros = 0;

    size_t q = 0;
    for (size_t rowMajorChunk : rowMajorChunks) {
        const size_t cx = rowMajorChunk % chunks_per_dim;
        const size_t cy = (rowMajorChunk / chunks_per_dim) % chunks_per_dim;
        const size_t cz = rowMajorChunk / (chunks_per_dim * chunks_per_dim);
        for (size_t local = 0; local < chunk_volume; ++local) {
            const size_t lx = local % chunk_dim;
            const size_t ly = (local / chunk_dim) % chunk_dim;
            const size_t lz = local / (chunk_dim * chunk_dim);
            const size_t x = cx * chunk_dim + lx;
            const size_t y = cy * chunk_dim + ly;
            const size_t z = cz * chunk_dim + lz;
            const size_t idx = (z * blksize_y + y) * blksize_x + x;
            const int qi = quant_inds[q++];
            if (qi == 0) {
                unpredZeros++;
            }
            deData[idx] = quantizer.recover(0, qi);
        }
    }

    free(buffer);
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
        std::cerr << "Usage: " << argv[0] << " <error_bound> [x0 y0 z0 x1 y1 z1]" << std::endl;
        return 1;
    }
    QueryBox query = make_query(0, 0, 0, roi_full_dim - 1, roi_full_dim - 1, roi_full_dim - 1);
    if (argc >= 8) {
        query = make_query(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]),
                           atoi(argv[5]), atoi(argv[6]), atoi(argv[7]));
    }
    std::cout << "Random-access Huffman marker test: low 128^3 uses 16^3 markers, "
              << "high 256^3 uses 32^3 markers." << std::endl;
    std::cout << "Query [" << query.x.begin << "," << query.x.end << "] x ["
              << query.y.begin << "," << query.y.end << "] x ["
              << query.z.begin << "," << query.z.end << "] "
              << (query.is_slice ? "(slice)" : "(box)") << std::endl;

    std::string full_file_path = "/home/daoce.wang/data/baryon_density.raw";
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
    std::cout << "raw marker bytes counted before zstd: "
              << marker_table_bytes_per_stream * 15 << " (15 streams x "
              << marker_table_bytes_per_stream << ")" << std::endl;

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

    auto roi_eval_start = std::chrono::high_resolution_clock::now();
    const auto highPlans = build_residual_plan(query, dim_x, roi_high_dim);
    auto highRefRegions = build_high_ref_regions(highPlans);
    const QueryBox querySub0Region = build_query_sub0_region(query);
    if (querySub0Region.x.begin <= querySub0Region.x.end &&
        querySub0Region.y.begin <= querySub0Region.y.end &&
        querySub0Region.z.begin <= querySub0Region.z.end) {
        highRefRegions.push_back(querySub0Region);
    }
    const auto lowPlans = build_low_residual_plan(highRefRegions);

    size_t queryLowChunks = 0;
    size_t queryHighChunks = 0;
    for (const auto& p : lowPlans) queryLowChunks += p.chunks.size();
    for (const auto& p : highPlans) queryHighChunks += p.chunks.size();

    float* query_low_ra_deData[7] = {};
    float* query_low_ra_de_sub_block[7] = {};
    float* query_high_ra_deData[7] = {};
    float* query_high_ra_de_sub_block[7] = {};
    float* query_old_high_de_sub_block[7] = {};
    for (int i = 0; i < 7; ++i) {
        query_low_ra_de_sub_block[i] = new float[low_dim_z * low_dim_y * low_dim_x]();
        query_high_ra_de_sub_block[i] = new float[dim_z * dim_y * dim_x]();
        query_old_high_de_sub_block[i] = new float[dim_z * dim_y * dim_x]();
    }

    size_t queryLowSymbols = 0;
    size_t queryHighSymbols = 0;
    auto query_low_decode_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:queryLowSymbols)
    for (int i = 0; i < static_cast<int>(lowPlans.size()); ++i) {
        const auto& plan = lowPlans[i];
        size_t decodedSymbols = 0;
        size_t unpredZeros = 0;
        query_low_ra_deData[plan.block] = SZ_decompress_nopred_marked_chunks(
            low_comp[plan.block], low_compressedSize[plan.block],
            low_dim_x, low_dim_y, low_dim_z, roi_low_dim, plan.chunks,
            decodedSymbols, unpredZeros);
        queryLowSymbols += decodedSymbols;
    }
    auto query_low_decode_end = std::chrono::high_resolution_clock::now();
    double query_low_decode_time = std::chrono::duration_cast<std::chrono::nanoseconds>(query_low_decode_end - query_low_decode_start).count() * 1e-9;

    auto query_low_depre_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(lowPlans.size()); ++i) {
        const auto& plan = lowPlans[i];
        depreprocess_block(plan.block, query_low_ra_deData[plan.block], decompressed_data,
                           query_low_ra_de_sub_block[plan.block],
                           low_dim_x, low_dim_y, low_dim_z,
                           plan.x.end + 1, plan.y.end + 1, plan.z.end + 1,
                           plan.x.begin, plan.y.begin, plan.z.begin);
    }
    auto query_low_depre_end = std::chrono::high_resolution_clock::now();
    double query_low_depre_time = std::chrono::duration_cast<std::chrono::nanoseconds>(query_low_depre_end - query_low_depre_start).count() * 1e-9;

    auto query_ref_reconstruct_start = std::chrono::high_resolution_clock::now();
    float* reconstructed_sub_0_ra = new float[dim_z * dim_y * dim_x]();
    for (const auto& region : highRefRegions) {
        for (int z = region.z.begin; z <= region.z.end; ++z) {
            for (int y = region.y.begin; y <= region.y.end; ++y) {
                for (int x = region.x.begin; x <= region.x.end; ++x) {
                    const int sub_index = ((z & 1) << 2) | ((y & 1) << 1) | (x & 1);
                    const int sub_z = z / 2;
                    const int sub_y = y / 2;
                    const int sub_x = x / 2;
                    const size_t dst = z * dim_y * dim_x + y * dim_x + x;
                    const size_t src = sub_z * low_dim_y * low_dim_x + sub_y * low_dim_x + sub_x;
                    reconstructed_sub_0_ra[dst] = sub_index == 0
                        ? decompressed_data[src]
                        : query_low_ra_de_sub_block[sub_index - 1][src];
                }
            }
        }
    }
    auto query_ref_reconstruct_end = std::chrono::high_resolution_clock::now();
    double query_ref_reconstruct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(query_ref_reconstruct_end - query_ref_reconstruct_start).count() * 1e-9;

    auto query_high_decode_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:queryHighSymbols)
    for (int i = 0; i < static_cast<int>(highPlans.size()); ++i) {
        const auto& plan = highPlans[i];
        size_t decodedSymbols = 0;
        size_t unpredZeros = 0;
        query_high_ra_deData[plan.block] = SZ_decompress_nopred_marked_chunks(
            comp[plan.block], compressedSize[plan.block],
            dim_x, dim_y, dim_z, roi_high_dim, plan.chunks,
            decodedSymbols, unpredZeros);
        queryHighSymbols += decodedSymbols;
    }
    auto query_high_decode_end = std::chrono::high_resolution_clock::now();
    double query_high_decode_time = std::chrono::duration_cast<std::chrono::nanoseconds>(query_high_decode_end - query_high_decode_start).count() * 1e-9;

    auto query_old_high_depre_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(highPlans.size()); ++i) {
        const auto& plan = highPlans[i];
        depreprocess_block(plan.block, deData[plan.block], reconstructed_sub_0,
                           query_old_high_de_sub_block[plan.block],
                           dim_x, dim_y, dim_z,
                           plan.x.end + 1, plan.y.end + 1, plan.z.end + 1,
                           plan.x.begin, plan.y.begin, plan.z.begin);
    }
    auto query_old_high_depre_end = std::chrono::high_resolution_clock::now();
    double query_old_high_depre_time = std::chrono::duration_cast<std::chrono::nanoseconds>(query_old_high_depre_end - query_old_high_depre_start).count() * 1e-9;

    auto query_high_depre_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(highPlans.size()); ++i) {
        const auto& plan = highPlans[i];
        depreprocess_block(plan.block, query_high_ra_deData[plan.block], reconstructed_sub_0_ra,
                           query_high_ra_de_sub_block[plan.block],
                           dim_x, dim_y, dim_z,
                           plan.x.end + 1, plan.y.end + 1, plan.z.end + 1,
                           plan.x.begin, plan.y.begin, plan.z.begin);
    }
    auto query_high_depre_end = std::chrono::high_resolution_clock::now();
    double query_high_depre_time = std::chrono::duration_cast<std::chrono::nanoseconds>(query_high_depre_end - query_high_depre_start).count() * 1e-9;

    auto query_desplit_start = std::chrono::high_resolution_clock::now();
    const size_t queryVolume = static_cast<size_t>(query.x.end - query.x.begin + 1) *
                               static_cast<size_t>(query.y.end - query.y.begin + 1) *
                               static_cast<size_t>(query.z.end - query.z.begin + 1);
    std::vector<float> queryResult(queryVolume);
    size_t qout = 0;
    for (int z = query.z.begin; z <= query.z.end; ++z) {
        for (int y = query.y.begin; y <= query.y.end; ++y) {
            for (int x = query.x.begin; x <= query.x.end; ++x) {
                const int sub_index = ((z & 1) << 2) | ((y & 1) << 1) | (x & 1);
                const int sub_z = z / 2;
                const int sub_y = y / 2;
                const int sub_x = x / 2;
                const size_t src = sub_z * dim_y * dim_x + sub_y * dim_x + sub_x;
                queryResult[qout++] = sub_index == 0
                    ? reconstructed_sub_0_ra[src]
                    : query_high_ra_de_sub_block[sub_index - 1][src];
            }
        }
    }
    auto query_desplit_end = std::chrono::high_resolution_clock::now();
    double query_desplit_time = std::chrono::duration_cast<std::chrono::nanoseconds>(query_desplit_end - query_desplit_start).count() * 1e-9;

    double query_max_abs_diff = 0.0;
    for (const auto& plan : highPlans) {
        for (int z = plan.z.begin; z <= plan.z.end; ++z) {
            for (int y = plan.y.begin; y <= plan.y.end; ++y) {
                for (int x = plan.x.begin; x <= plan.x.end; ++x) {
                    const size_t idx = z * dim_y * dim_x + y * dim_x + x;
                    query_max_abs_diff = std::max(query_max_abs_diff,
                        static_cast<double>(std::abs(query_old_high_de_sub_block[plan.block][idx] -
                                                     query_high_ra_de_sub_block[plan.block][idx])));
                }
            }
        }
    }

    double query_final_max_abs_diff = 0.0;
    qout = 0;
    for (int z = query.z.begin; z <= query.z.end; ++z) {
        for (int y = query.y.begin; y <= query.y.end; ++y) {
            for (int x = query.x.begin; x <= query.x.end; ++x) {
                const int sub_index = ((z & 1) << 2) | ((y & 1) << 1) | (x & 1);
                const int sub_z = z / 2;
                const int sub_y = y / 2;
                const int sub_x = x / 2;
                const size_t src = sub_z * dim_y * dim_x + sub_y * dim_x + sub_x;
                const float full_value = sub_index == 0
                    ? reconstructed_sub_0[src]
                    : de_sub_block[sub_index - 1][src];
                query_final_max_abs_diff = std::max(query_final_max_abs_diff,
                    static_cast<double>(std::abs(queryResult[qout++] - full_value)));
            }
        }
    }

    std::cout << "Auto query high residual blocks decoded: " << blocks_to_string(highPlans) << std::endl;
    std::cout << "Auto query low residual blocks decoded: " << blocks_to_string(lowPlans) << std::endl;
    std::cout << "Auto query chunks low/high: " << queryLowChunks << " / " << queryHighChunks << std::endl;
    std::cout << "Auto query marker fine decode time: "
              << query_low_decode_time + query_high_decode_time << " sec" << std::endl;
    std::cout << "Auto query marker fine depreprocess time: "
              << query_low_depre_time + query_high_depre_time << " sec" << std::endl;
    std::cout << "Auto query ref reconstruct/desplit time: "
              << query_ref_reconstruct_time << " / " << query_desplit_time << " sec" << std::endl;
    std::cout << "Auto query full fine decode baseline: "
              << low_time_taken_decompress_sz + time_taken_decompress_sz << " sec" << std::endl;
    std::cout << "Auto query full fine depreprocess baseline: "
              << (low_time_taken_decompress - low_time_taken_decompress_sz) +
                     (time_taken_decompress - time_taken_decompress_sz)
              << " sec" << std::endl;
    std::cout << "Auto query old high-only depreprocess time: "
              << query_old_high_depre_time << " sec" << std::endl;
    std::cout << "Auto query decoded symbols low/high: "
              << queryLowSymbols << " / " << queryHighSymbols << std::endl;
    std::cout << "Auto query marker vs full-decode max abs diff: "
              << query_max_abs_diff << std::endl;
    std::cout << "Auto query final ROI max abs diff: "
              << query_final_max_abs_diff << std::endl;

    for (int i = 0; i < 7; ++i) {
        delete[] query_low_ra_deData[i];
        delete[] query_low_ra_de_sub_block[i];
        delete[] query_high_ra_deData[i];
        delete[] query_high_ra_de_sub_block[i];
        delete[] query_old_high_de_sub_block[i];
    }
    delete[] reconstructed_sub_0_ra;

    if (false) {
    std::vector<size_t> roi_chunk0 = {0};
    float* low_ra_deData[7];
    float* low_ra_de_sub_block[7];
    size_t low_ra_symbols = 0;
    size_t low_ra_unpred = 0;
    for (int i = 0; i < 7; ++i) {
        low_ra_deData[i] = nullptr;
        low_ra_de_sub_block[i] = new float[low_dim_z * low_dim_y * low_dim_x]();
    }

    auto low_ra_decode_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:low_ra_symbols, low_ra_unpred)
    for (int block = 0; block < 7; ++block) {
        size_t decodedSymbols = 0;
        size_t unpredZeros = 0;
        low_ra_deData[block] = SZ_decompress_nopred_marked_chunks(
            low_comp[block], low_compressedSize[block],
            low_dim_x, low_dim_y, low_dim_z, roi_low_dim, roi_chunk0,
            decodedSymbols, unpredZeros);
        low_ra_symbols += decodedSymbols;
        low_ra_unpred += unpredZeros;
    }
    auto low_ra_decode_end = std::chrono::high_resolution_clock::now();
    double low_ra_decode_time = std::chrono::duration_cast<std::chrono::nanoseconds>(low_ra_decode_end - low_ra_decode_start).count() * 1e-9;

    auto low_ra_depre_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int block = 0; block < 7; ++block) {
        depreprocess_block(block, low_ra_deData[block], decompressed_data, low_ra_de_sub_block[block],
                           low_dim_x, low_dim_y, low_dim_z,
                           roi_low_dim, roi_low_dim, roi_low_dim);
    }
    auto low_ra_depre_end = std::chrono::high_resolution_clock::now();
    double low_ra_depre_time = std::chrono::duration_cast<std::chrono::nanoseconds>(low_ra_depre_end - low_ra_depre_start).count() * 1e-9;

    float* roi_old_de_sub_block[7];
    float* roi_ra_deData[7];
    float* roi_ra_de_sub_block[7];
    size_t high_ra_symbols = 0;
    size_t high_ra_unpred = 0;
    for (int i = 0; i < 7; ++i) {
        roi_old_de_sub_block[i] = new float[dim_z * dim_y * dim_x]();
        roi_ra_deData[i] = nullptr;
        roi_ra_de_sub_block[i] = new float[dim_z * dim_y * dim_x]();
    }

    auto roi_old_depre_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int block = 0; block < 7; ++block) {
        depreprocess_block(block, deData[block], reconstructed_sub_0, roi_old_de_sub_block[block],
                           dim_x, dim_y, dim_z,
                           roi_high_dim, roi_high_dim, roi_high_dim);
    }
    auto roi_old_depre_end = std::chrono::high_resolution_clock::now();
    double roi_old_depre_time = std::chrono::duration_cast<std::chrono::nanoseconds>(roi_old_depre_end - roi_old_depre_start).count() * 1e-9;

    auto high_ra_decode_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:high_ra_symbols, high_ra_unpred)
    for (int block = 0; block < 7; ++block) {
        size_t decodedSymbols = 0;
        size_t unpredZeros = 0;
        roi_ra_deData[block] = SZ_decompress_nopred_marked_chunks(
            comp[block], compressedSize[block],
            dim_x, dim_y, dim_z, roi_high_dim, roi_chunk0,
            decodedSymbols, unpredZeros);
        high_ra_symbols += decodedSymbols;
        high_ra_unpred += unpredZeros;
    }
    auto high_ra_decode_end = std::chrono::high_resolution_clock::now();
    double high_ra_decode_time = std::chrono::duration_cast<std::chrono::nanoseconds>(high_ra_decode_end - high_ra_decode_start).count() * 1e-9;

    auto high_ra_depre_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int block = 0; block < 7; ++block) {
        depreprocess_block(block, roi_ra_deData[block], reconstructed_sub_0, roi_ra_de_sub_block[block],
                           dim_x, dim_y, dim_z,
                           roi_high_dim, roi_high_dim, roi_high_dim);
    }
    auto high_ra_depre_end = std::chrono::high_resolution_clock::now();
    double high_ra_depre_time = std::chrono::duration_cast<std::chrono::nanoseconds>(high_ra_depre_end - high_ra_depre_start).count() * 1e-9;

    double roi_max_abs_diff = 0.0;
    for (int block = 0; block < 7; ++block) {
        for (int z = 0; z < roi_high_dim; ++z) {
            for (int y = 0; y < roi_high_dim; ++y) {
                for (int x = 0; x < roi_high_dim; ++x) {
                    const size_t idx = z * dim_y * dim_x + y * dim_x + x;
                    roi_max_abs_diff = std::max(roi_max_abs_diff,
                        static_cast<double>(std::abs(roi_old_de_sub_block[block][idx] - roi_ra_de_sub_block[block][idx])));
                }
            }
        }
    }

    std::cout << "ROI 64^3 old fine decode time: "
              << low_time_taken_decompress_sz + time_taken_decompress_sz << " sec" << std::endl;
    std::cout << "ROI 64^3 marker fine decode time: "
              << low_ra_decode_time + high_ra_decode_time << " sec" << std::endl;
    std::cout << "ROI 64^3 full fine depreprocess time: "
              << (low_time_taken_decompress - low_time_taken_decompress_sz) +
                     (time_taken_decompress - time_taken_decompress_sz)
              << " sec" << std::endl;
    std::cout << "ROI 64^3 old high ROI depreprocess time: "
              << roi_old_depre_time << " sec" << std::endl;
    std::cout << "ROI 64^3 marker fine depreprocess time: "
              << low_ra_depre_time + high_ra_depre_time << " sec" << std::endl;
    std::cout << "ROI marker decoded symbols low/high: "
              << low_ra_symbols << " / " << high_ra_symbols << std::endl;
    std::cout << "ROI marker unpred-zero count low/high: "
              << low_ra_unpred << " / " << high_ra_unpred << std::endl;
    std::cout << "ROI marker vs full-decode max abs diff: "
              << roi_max_abs_diff << std::endl;

    const int full_slice_z = full_dim_z / 2;
    const int high_slice_z = full_slice_z / 2;
    const int low_slice_z = high_slice_z / 2;
    const int high_z_parity = full_slice_z & 1;
    const int low_z_parity = high_slice_z & 1;
    std::vector<int> high_slice_blocks;
    std::vector<int> low_slice_blocks;
    for (int block = 0; block < 7; ++block) {
        const int sub_index = block + 1;
        if (((sub_index >> 2) & 1) == high_z_parity) {
            high_slice_blocks.push_back(block);
        }
        if (((sub_index >> 2) & 1) == low_z_parity) {
            low_slice_blocks.push_back(block);
        }
    }
    const std::vector<size_t> low_slice_chunks = chunks_for_z_slice(low_dim_x, roi_low_dim, low_slice_z);
    const std::vector<size_t> high_slice_chunks = chunks_for_z_slice(dim_x, roi_high_dim, high_slice_z);

    float* slice_low_ra_deData[7] = {};
    float* slice_low_ra_de_sub_block[7] = {};
    float* slice_high_ra_deData[7] = {};
    float* slice_high_ra_de_sub_block[7] = {};
    float* slice_old_high_de_sub_block[7] = {};
    for (int i = 0; i < 7; ++i) {
        slice_low_ra_de_sub_block[i] = new float[low_dim_z * low_dim_y * low_dim_x]();
        slice_high_ra_de_sub_block[i] = new float[dim_z * dim_y * dim_x]();
        slice_old_high_de_sub_block[i] = new float[dim_z * dim_y * dim_x]();
    }

    size_t slice_low_symbols = 0;
    size_t slice_high_symbols = 0;
    auto slice_low_decode_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:slice_low_symbols)
    for (int i = 0; i < static_cast<int>(low_slice_blocks.size()); ++i) {
        const int block = low_slice_blocks[i];
        size_t decodedSymbols = 0;
        size_t unpredZeros = 0;
        slice_low_ra_deData[block] = SZ_decompress_nopred_marked_chunks(
            low_comp[block], low_compressedSize[block],
            low_dim_x, low_dim_y, low_dim_z, roi_low_dim, low_slice_chunks,
            decodedSymbols, unpredZeros);
        slice_low_symbols += decodedSymbols;
    }
    auto slice_low_decode_end = std::chrono::high_resolution_clock::now();
    double slice_low_decode_time = std::chrono::duration_cast<std::chrono::nanoseconds>(slice_low_decode_end - slice_low_decode_start).count() * 1e-9;

    auto slice_low_depre_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(low_slice_blocks.size()); ++i) {
        const int block = low_slice_blocks[i];
        depreprocess_block(block, slice_low_ra_deData[block], decompressed_data, slice_low_ra_de_sub_block[block],
                           low_dim_x, low_dim_y, low_dim_z,
                           low_dim_x, low_dim_y, low_slice_z + 1,
                           0, 0, low_slice_z);
    }
    auto slice_low_depre_end = std::chrono::high_resolution_clock::now();
    double slice_low_depre_time = std::chrono::duration_cast<std::chrono::nanoseconds>(slice_low_depre_end - slice_low_depre_start).count() * 1e-9;

    auto slice_high_decode_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:slice_high_symbols)
    for (int i = 0; i < static_cast<int>(high_slice_blocks.size()); ++i) {
        const int block = high_slice_blocks[i];
        size_t decodedSymbols = 0;
        size_t unpredZeros = 0;
        slice_high_ra_deData[block] = SZ_decompress_nopred_marked_chunks(
            comp[block], compressedSize[block],
            dim_x, dim_y, dim_z, roi_high_dim, high_slice_chunks,
            decodedSymbols, unpredZeros);
        slice_high_symbols += decodedSymbols;
    }
    auto slice_high_decode_end = std::chrono::high_resolution_clock::now();
    double slice_high_decode_time = std::chrono::duration_cast<std::chrono::nanoseconds>(slice_high_decode_end - slice_high_decode_start).count() * 1e-9;

    auto slice_old_high_depre_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(high_slice_blocks.size()); ++i) {
        const int block = high_slice_blocks[i];
        depreprocess_block(block, deData[block], reconstructed_sub_0, slice_old_high_de_sub_block[block],
                           dim_x, dim_y, dim_z,
                           dim_x, dim_y, high_slice_z + 1,
                           0, 0, high_slice_z);
    }
    auto slice_old_high_depre_end = std::chrono::high_resolution_clock::now();
    double slice_old_high_depre_time = std::chrono::duration_cast<std::chrono::nanoseconds>(slice_old_high_depre_end - slice_old_high_depre_start).count() * 1e-9;

    auto slice_high_depre_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(high_slice_blocks.size()); ++i) {
        const int block = high_slice_blocks[i];
        depreprocess_block(block, slice_high_ra_deData[block], reconstructed_sub_0, slice_high_ra_de_sub_block[block],
                           dim_x, dim_y, dim_z,
                           dim_x, dim_y, high_slice_z + 1,
                           0, 0, high_slice_z);
    }
    auto slice_high_depre_end = std::chrono::high_resolution_clock::now();
    double slice_high_depre_time = std::chrono::duration_cast<std::chrono::nanoseconds>(slice_high_depre_end - slice_high_depre_start).count() * 1e-9;

    double slice_max_abs_diff = 0.0;
    for (int block : high_slice_blocks) {
        for (int y = 0; y < dim_y; ++y) {
            for (int x = 0; x < dim_x; ++x) {
                const size_t idx = high_slice_z * dim_y * dim_x + y * dim_x + x;
                slice_max_abs_diff = std::max(slice_max_abs_diff,
                    static_cast<double>(std::abs(slice_old_high_de_sub_block[block][idx] - slice_high_ra_de_sub_block[block][idx])));
            }
        }
    }

    std::cout << "Slice z=" << full_slice_z << " high residual blocks decoded: ";
    for (int block : high_slice_blocks) std::cout << block << " ";
    std::cout << std::endl;
    std::cout << "Slice z=" << full_slice_z << " low residual blocks decoded: ";
    for (int block : low_slice_blocks) std::cout << block << " ";
    std::cout << std::endl;
    std::cout << "Slice marker chunks per decoded low/high block: "
              << low_slice_chunks.size() << " / " << high_slice_chunks.size() << std::endl;
    std::cout << "Slice marker fine decode time: "
              << slice_low_decode_time + slice_high_decode_time << " sec" << std::endl;
    std::cout << "Slice marker fine depreprocess time: "
              << slice_low_depre_time + slice_high_depre_time << " sec" << std::endl;
    std::cout << "Slice old high-only depreprocess time: "
              << slice_old_high_depre_time << " sec" << std::endl;
    std::cout << "Slice marker decoded symbols low/high: "
              << slice_low_symbols << " / " << slice_high_symbols << std::endl;
    std::cout << "Slice marker vs full-decode max abs diff: "
              << slice_max_abs_diff << std::endl;

    for (int i = 0; i < 7; ++i) {
        delete[] slice_low_ra_deData[i];
        delete[] slice_low_ra_de_sub_block[i];
        delete[] slice_high_ra_deData[i];
        delete[] slice_high_ra_de_sub_block[i];
        delete[] slice_old_high_de_sub_block[i];
    }
    }

    auto roi_eval_end = std::chrono::high_resolution_clock::now();
    double roi_eval_time = std::chrono::duration_cast<std::chrono::nanoseconds>(roi_eval_end - roi_eval_start).count() * 1e-9;

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
    double global_decompress_time = std::chrono::duration_cast<std::chrono::nanoseconds>(reconstructed_full_end - global_decompress_start).count() * 1e-9 - roi_eval_time;
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
