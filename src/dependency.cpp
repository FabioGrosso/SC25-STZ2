#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <cfloat>
#include <cmath>
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
constexpr float residual_scale = 1.0f; // Adjust this based on the expected range of residuals
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
    #pragma omp parallel for reduction(max:high) reduction(min:low)
    for (size_t i = 0; i < num_elements; i++) {
        if (data[i] > high)
            high = data[i];
        if (data[i] < low)
            low = data[i];
    }
    double range = high-low;
    return range;
}

void scaleData(float* data, size_t num_elements, float scale) {
    if (scale == 1.0f)
        return;
    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i)
        data[i] *= scale;
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

static inline float interp_axis(const float* ref, int axis, int idx, int x, int y, int z,
                                int dim_x, int dim_y, int dim_z, int dim_xy)
{
    constexpr float W0 = -0.0625f;
    constexpr float W1 = 0.5625f;
    constexpr float W2 = 0.5625f;
    constexpr float W3 = -0.0625f;

    if (axis == 0) {
        if (x == dim_x - 1)
            return ref[idx];
        if (x == dim_x - 2 || x == 0)
            return 0.5f * (ref[idx] + ref[idx + 1]);
        return W0 * ref[idx - 1] + W1 * ref[idx] + W2 * ref[idx + 1] + W3 * ref[idx + 2];
    }
    if (axis == 1) {
        if (y == dim_y - 1)
            return ref[idx];
        if (y == dim_y - 2 || y == 0)
            return 0.5f * (ref[idx] + ref[idx + dim_x]);
        return W0 * ref[idx - dim_x] + W1 * ref[idx] + W2 * ref[idx + dim_x] + W3 * ref[idx + 2 * dim_x];
    }

    if (z == dim_z - 1)
        return ref[idx];
    if (z == dim_z - 2 || z == 0)
        return 0.5f * (ref[idx] + ref[idx + dim_xy]);
    return W0 * ref[idx - dim_xy] + W1 * ref[idx] + W2 * ref[idx + dim_xy] + W3 * ref[idx + 2 * dim_xy];
}

static inline float local_interp_variance(const float* ref, int axis, int idx, int x, int y, int z,
                                          int dim_x, int dim_y, int dim_z, int dim_xy)
{
    constexpr float eps = 1e-12f;

    if (axis == 0) {
        if (x >= 1 && x < dim_x - 2) {
            const float c0 = ref[idx - 1] - 2.0f * ref[idx] + ref[idx + 1];
            const float c1 = ref[idx] - 2.0f * ref[idx + 1] + ref[idx + 2];
            return 0.5f * (c0 * c0 + c1 * c1) + eps;
        }
        if (x + 1 < dim_x) {
            const float e = std::fabs(ref[idx + 1] - ref[idx]);
            return e * e + eps;
        }
        return eps;
    }
    if (axis == 1) {
        if (y >= 1 && y < dim_y - 2) {
            const float c0 = ref[idx - dim_x] - 2.0f * ref[idx] + ref[idx + dim_x];
            const float c1 = ref[idx] - 2.0f * ref[idx + dim_x] + ref[idx + 2 * dim_x];
            return 0.5f * (c0 * c0 + c1 * c1) + eps;
        }
        if (y + 1 < dim_y) {
            const float e = std::fabs(ref[idx + dim_x] - ref[idx]);
            return e * e + eps;
        }
        return eps;
    }

    if (z >= 1 && z < dim_z - 2) {
        const float c0 = ref[idx - dim_xy] - 2.0f * ref[idx] + ref[idx + dim_xy];
        const float c1 = ref[idx] - 2.0f * ref[idx + dim_xy] + ref[idx + 2 * dim_xy];
        return 0.5f * (c0 * c0 + c1 * c1) + eps;
    }
    if (z + 1 < dim_z) {
        const float e = std::fabs(ref[idx + dim_xy] - ref[idx]);
        return e * e + eps;
    }
    return eps;
}

static inline void interp_axis_with_variance(const float* ref, int axis, int idx, int x, int y, int z,
                                             int dim_x, int dim_y, int dim_z, int dim_xy,
                                             float& pred, float& variance)
{
    constexpr float W0 = -0.0625f;
    constexpr float W1 = 0.5625f;
    constexpr float W2 = 0.5625f;
    constexpr float W3 = -0.0625f;
    constexpr float eps = 1e-12f;

    if (axis == 0) {
        if (x == dim_x - 1) {
            pred = ref[idx];
            variance = eps;
            return;
        }
        if (x == dim_x - 2 || x == 0) {
            const float a = ref[idx];
            const float b = ref[idx + 1];
            const float e = std::fabs(b - a);
            pred = 0.5f * (a + b);
            variance = e * e + eps;
            return;
        }
        const float a = ref[idx - 1];
        const float b = ref[idx];
        const float c = ref[idx + 1];
        const float d = ref[idx + 2];
        const float c0 = a - 2.0f * b + c;
        const float c1 = b - 2.0f * c + d;
        pred = W0 * a + W1 * b + W2 * c + W3 * d;
        variance = 0.5f * (c0 * c0 + c1 * c1) + eps;
        return;
    }

    if (axis == 1) {
        if (y == dim_y - 1) {
            pred = ref[idx];
            variance = eps;
            return;
        }
        if (y == dim_y - 2 || y == 0) {
            const float a = ref[idx];
            const float b = ref[idx + dim_x];
            const float e = std::fabs(b - a);
            pred = 0.5f * (a + b);
            variance = e * e + eps;
            return;
        }
        const float a = ref[idx - dim_x];
        const float b = ref[idx];
        const float c = ref[idx + dim_x];
        const float d = ref[idx + 2 * dim_x];
        const float c0 = a - 2.0f * b + c;
        const float c1 = b - 2.0f * c + d;
        pred = W0 * a + W1 * b + W2 * c + W3 * d;
        variance = 0.5f * (c0 * c0 + c1 * c1) + eps;
        return;
    }

    if (z == dim_z - 1) {
        pred = ref[idx];
        variance = eps;
        return;
    }
    if (z == dim_z - 2 || z == 0) {
        const float a = ref[idx];
        const float b = ref[idx + dim_xy];
        const float e = std::fabs(b - a);
        pred = 0.5f * (a + b);
        variance = e * e + eps;
        return;
    }
    const float a = ref[idx - dim_xy];
    const float b = ref[idx];
    const float c = ref[idx + dim_xy];
    const float d = ref[idx + 2 * dim_xy];
    const float c0 = a - 2.0f * b + c;
    const float c1 = b - 2.0f * c + d;
    pred = W0 * a + W1 * b + W2 * c + W3 * d;
    variance = 0.5f * (c0 * c0 + c1 * c1) + eps;
}

static inline float inverse_variance_blend2(float p0, float v0, float p1, float v1)
{
    return (p0 * v1 + p1 * v0) / (v0 + v1);
}

static inline float inverse_variance_blend3(float p0, float v0, float p1, float v1, float p2, float v2)
{
    const float w0 = v1 * v2;
    const float w1 = v0 * v2;
    const float w2 = v0 * v1;
    return (w0 * p0 + w1 * p1 + w2 * p2) / (w0 + w1 + w2);
}

static inline float staged_prediction(int block, const float* ref_x, const float* ref_y, const float* ref_z,
                                      const float* ref_xy, const float* ref_xz, const float* ref_yz,
                                      int idx, int x, int y, int z,
                                      int dim_x, int dim_y, int dim_z, int dim_xy)
{
    switch (block) {
        case 2: {
            float p0, p1, v0, v1;
            interp_axis_with_variance(ref_x, 1, idx, x, y, z, dim_x, dim_y, dim_z, dim_xy, p0, v0);
            interp_axis_with_variance(ref_y, 0, idx, x, y, z, dim_x, dim_y, dim_z, dim_xy, p1, v1);
            return inverse_variance_blend2(p0, v0, p1, v1);
        }
        case 4: {
            float p0, p1, v0, v1;
            interp_axis_with_variance(ref_x, 2, idx, x, y, z, dim_x, dim_y, dim_z, dim_xy, p0, v0);
            interp_axis_with_variance(ref_z, 0, idx, x, y, z, dim_x, dim_y, dim_z, dim_xy, p1, v1);
            return inverse_variance_blend2(p0, v0, p1, v1);
        }
        case 5: {
            float p0, p1, v0, v1;
            interp_axis_with_variance(ref_y, 2, idx, x, y, z, dim_x, dim_y, dim_z, dim_xy, p0, v0);
            interp_axis_with_variance(ref_z, 1, idx, x, y, z, dim_x, dim_y, dim_z, dim_xy, p1, v1);
            return inverse_variance_blend2(p0, v0, p1, v1);
        }
        case 6: {
            float p0, p1, p2, v0, v1, v2;
            interp_axis_with_variance(ref_xy, 2, idx, x, y, z, dim_x, dim_y, dim_z, dim_xy, p0, v0);
            interp_axis_with_variance(ref_xz, 1, idx, x, y, z, dim_x, dim_y, dim_z, dim_xy, p1, v1);
            interp_axis_with_variance(ref_yz, 0, idx, x, y, z, dim_x, dim_y, dim_z, dim_xy, p2, v2);
            return inverse_variance_blend3(p0, v0, p1, v1, p2, v2);
        }
        default:
            return 0.0f;
    }
}

void preprocess_block_staged(int block, const float* sub_block, const float* ref_x, const float* ref_y,
                             const float* ref_z, const float* ref_xy, const float* ref_xz,
                             const float* ref_yz, float* diff, int dim_x, int dim_y, int dim_z)
{
    int dim_xy = dim_x * dim_y;

    #pragma omp parallel for
    for (int z = 0; z < dim_z; ++z) {
        for (int y = 0; y < dim_y; ++y) {
            for (int x = 0; x < dim_x; ++x) {
                const int idx = z * dim_xy + y * dim_x + x;
                diff[idx] = sub_block[idx] - staged_prediction(block, ref_x, ref_y, ref_z, ref_xy, ref_xz, ref_yz,
                                                               idx, x, y, z, dim_x, dim_y, dim_z, dim_xy);
            }
        }
    }
}

void depreprocess_block_staged(int block, const float* deData, const float* ref_x, const float* ref_y,
                               const float* ref_z, const float* ref_xy, const float* ref_xz,
                               const float* ref_yz, float* de_sub, int dim_x, int dim_y, int dim_z)
{
    int dim_xy = dim_x * dim_y;

    #pragma omp parallel for
    for (int z = 0; z < dim_z; ++z) {
        for (int y = 0; y < dim_y; ++y) {
            for (int x = 0; x < dim_x; ++x) {
                const int idx = z * dim_xy + y * dim_x + x;
                de_sub[idx] = deData[idx] + staged_prediction(block, ref_x, ref_y, ref_z, ref_xy, ref_xz, ref_yz,
                                                              idx, x, y, z, dim_x, dim_y, dim_z, dim_xy);
            }
        }
    }
}

//---------------------------------------------------------------------
// Preprocessing: Compute diff = sub_block - reference (sz_out_data)
//---------------------------------------------------------------------
void preprocess_block(int block, const float* sub_block, const float* ref, float* diff,
                        int dim_x, int dim_y, int dim_z)
{
    int dim_xy= dim_x * dim_y;
    const double W[4] = {-0.0625, 0.5625, 0.5625, -0.0625}; // Tensor product weights for cubic spline

    if (block == 0) {
        #pragma omp parallel for
        for (int z = 0; z < dim_z; ++z) {
            for (int y = 0; y < dim_y; ++y) {
                for (int x = 0; x < dim_x; ++x) {
                    const int idx = z * dim_xy + y * dim_x + x;
                    diff[idx] = sub_block[idx] - interp_axis(ref, 0, idx, x, y, z,
                                                             dim_x, dim_y, dim_z, dim_xy);
                }
            }
        }
        return;
    }
    if (block == 1) {
        #pragma omp parallel for
        for (int z = 0; z < dim_z; ++z) {
            for (int y = 0; y < dim_y; ++y) {
                for (int x = 0; x < dim_x; ++x) {
                    const int idx = z * dim_xy + y * dim_x + x;
                    diff[idx] = sub_block[idx] - interp_axis(ref, 1, idx, x, y, z,
                                                             dim_x, dim_y, dim_z, dim_xy);
                }
            }
        }
        return;
    }
    if (block == 3) {
        #pragma omp parallel for
        for (int z = 0; z < dim_z; ++z) {
            for (int y = 0; y < dim_y; ++y) {
                for (int x = 0; x < dim_x; ++x) {
                    const int idx = z * dim_xy + y * dim_x + x;
                    diff[idx] = sub_block[idx] - interp_axis(ref, 2, idx, x, y, z,
                                                             dim_x, dim_y, dim_z, dim_xy);
                }
            }
        }
        return;
    }

    std::cerr << "Unsupported block index in preprocess_block\n";
}

//---------------------------------------------------------------------
// De-preprocessing: Reconstruct de_sub = deData + reference
//---------------------------------------------------------------------
void depreprocess_block(int block, const float* deData, const float* ref, float* de_sub,
                          int dim_x, int dim_y, int dim_z)
{
    int dim_xy= dim_x * dim_y;
    const double W[4] = {-0.0625, 0.5625, 0.5625, -0.0625};

    if (block == 0) {
        #pragma omp parallel for
        for (int z = 0; z < dim_z; ++z) {
            for (int y = 0; y < dim_y; ++y) {
                for (int x = 0; x < dim_x; ++x) {
                    const int idx = z * dim_xy + y * dim_x + x;
                    de_sub[idx] = deData[idx] + interp_axis(ref, 0, idx, x, y, z,
                                                            dim_x, dim_y, dim_z, dim_xy);
                }
            }
        }
        return;
    }
    if (block == 1) {
        #pragma omp parallel for
        for (int z = 0; z < dim_z; ++z) {
            for (int y = 0; y < dim_y; ++y) {
                for (int x = 0; x < dim_x; ++x) {
                    const int idx = z * dim_xy + y * dim_x + x;
                    de_sub[idx] = deData[idx] + interp_axis(ref, 1, idx, x, y, z,
                                                            dim_x, dim_y, dim_z, dim_xy);
                }
            }
        }
        return;
    }
    if (block == 3) {
        #pragma omp parallel for
        for (int z = 0; z < dim_z; ++z) {
            for (int y = 0; y < dim_y; ++y) {
                for (int x = 0; x < dim_x; ++x) {
                    const int idx = z * dim_xy + y * dim_x + x;
                    de_sub[idx] = deData[idx] + interp_axis(ref, 2, idx, x, y, z,
                                                            dim_x, dim_y, dim_z, dim_xy);
                }
            }
        }
        return;
    }

    std::cerr << "Unsupported block index in depreprocess_block\n";
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
    // Note: Returning original data (as in your original code)
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
    // std::cout << "Time taken by sz decompression is: " << std::fixed << std::setprecision(5)
    //         << sz_time_taken_decompress << " sec" << std::endl;
    // writeBinaryData("tur-low-2.raw", decompressed_data, full_dim_z/4 * full_dim_y/4 * full_dim_x/4);

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

    const int single_axis_blocks[3] = {0, 1, 3};
    const int double_axis_blocks[3] = {2, 4, 5};
    const size_t low_num_elements = low_dim_z * low_dim_y * low_dim_x;

    auto low_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:allSize)
    for (int i = 0; i < 3; ++i) {
        int block = single_axis_blocks[i];
        preprocess_block(block, low_block_data[block+1], low_block_data[0], low_diff_data[block],
                           low_dim_x, low_dim_y, low_dim_z);
        scaleData(low_diff_data[block], low_num_elements, residual_scale);
        low_comp[block] = SZ_compress(low_diff_data[block], low_dim_x, low_dim_y, low_dim_z, 2.5 * eb, low_compressedSize[block]);
        allSize += low_compressedSize[block];
        scaleData(low_diff_data[block], low_num_elements, 1.0f / residual_scale);
        depreprocess_block(block, low_diff_data[block], low_block_data[0], low_de_sub_block[block],
                           low_dim_x, low_dim_y, low_dim_z);
    }

    #pragma omp parallel for reduction(+:allSize)
    for (int i = 0; i < 3; ++i) {
        int block = double_axis_blocks[i];
        preprocess_block_staged(block, low_block_data[block+1], low_de_sub_block[0], low_de_sub_block[1],
                                low_de_sub_block[3], low_de_sub_block[2], low_de_sub_block[4],
                                low_de_sub_block[5], low_diff_data[block], low_dim_x, low_dim_y, low_dim_z);
        scaleData(low_diff_data[block], low_num_elements, residual_scale);
        low_comp[block] = SZ_compress(low_diff_data[block], low_dim_x, low_dim_y, low_dim_z, 2.5 * eb, low_compressedSize[block]);
        allSize += low_compressedSize[block];
        scaleData(low_diff_data[block], low_num_elements, 1.0f / residual_scale);
        depreprocess_block_staged(block, low_diff_data[block], low_de_sub_block[0], low_de_sub_block[1],
                                  low_de_sub_block[3], low_de_sub_block[2], low_de_sub_block[4],
                                  low_de_sub_block[5], low_de_sub_block[block], low_dim_x, low_dim_y, low_dim_z);
    }

    preprocess_block_staged(6, low_block_data[7], low_de_sub_block[0], low_de_sub_block[1],
                            low_de_sub_block[3], low_de_sub_block[2], low_de_sub_block[4],
                            low_de_sub_block[5], low_diff_data[6], low_dim_x, low_dim_y, low_dim_z);
    scaleData(low_diff_data[6], low_num_elements, residual_scale);
    low_comp[6] = SZ_compress(low_diff_data[6], low_dim_x, low_dim_y, low_dim_z, 2.5 * eb, low_compressedSize[6]);
    allSize += low_compressedSize[6];
    scaleData(low_diff_data[6], low_num_elements, 1.0f / residual_scale);
    depreprocess_block_staged(6, low_diff_data[6], low_de_sub_block[0], low_de_sub_block[1],
                              low_de_sub_block[3], low_de_sub_block[2], low_de_sub_block[4],
                              low_de_sub_block[5], low_de_sub_block[6], low_dim_x, low_dim_y, low_dim_z);
    auto low_end = std::chrono::high_resolution_clock::now();
    double low_time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(low_end - low_start).count() * 1e-9;
    
    auto reconstructed_low_start = std::chrono::high_resolution_clock::now();
    float* reconstructed_sub_0 = new float[dim_z * dim_x * dim_y];
    float* all_low_blocks[8] = {
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
    auto reconstructed_low_end = std::chrono::high_resolution_clock::now();
    double time_taken_reconstructed_low = std::chrono::duration_cast<std::chrono::nanoseconds>(reconstructed_low_end - reconstructed_low_start).count() * 1e-9;
    
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

    const size_t num_elements = dim_z * dim_y * dim_x;

    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:allSize)
    for (int i = 0; i < 3; ++i) {
        int block = single_axis_blocks[i];
        preprocess_block(block, sub_block_data[block+1], reconstructed_sub_0, diff_data[block],
                           dim_x, dim_y, dim_z);
        scaleData(diff_data[block], num_elements, residual_scale);
        comp[block] = SZ_compress(diff_data[block], dim_x, dim_y, dim_z, 6.25 * eb, compressedSize[block]);
        allSize += compressedSize[block];
        scaleData(diff_data[block], num_elements, 1.0f / residual_scale);
        depreprocess_block(block, diff_data[block], reconstructed_sub_0, de_sub_block[block],
                           dim_x, dim_y, dim_z);
    }

    #pragma omp parallel for reduction(+:allSize)
    for (int i = 0; i < 3; ++i) {
        int block = double_axis_blocks[i];
        preprocess_block_staged(block, sub_block_data[block+1], de_sub_block[0], de_sub_block[1],
                                de_sub_block[3], de_sub_block[2], de_sub_block[4],
                                de_sub_block[5], diff_data[block], dim_x, dim_y, dim_z);
        scaleData(diff_data[block], num_elements, residual_scale);
        comp[block] = SZ_compress(diff_data[block], dim_x, dim_y, dim_z, 6.25 * eb, compressedSize[block]);
        allSize += compressedSize[block];
        scaleData(diff_data[block], num_elements, 1.0f / residual_scale);
        depreprocess_block_staged(block, diff_data[block], de_sub_block[0], de_sub_block[1],
                                  de_sub_block[3], de_sub_block[2], de_sub_block[4],
                                  de_sub_block[5], de_sub_block[block], dim_x, dim_y, dim_z);
    }

    preprocess_block_staged(6, sub_block_data[7], de_sub_block[0], de_sub_block[1],
                            de_sub_block[3], de_sub_block[2], de_sub_block[4],
                            de_sub_block[5], diff_data[6], dim_x, dim_y, dim_z);
    scaleData(diff_data[6], num_elements, residual_scale);
    comp[6] = SZ_compress(diff_data[6], dim_x, dim_y, dim_z, 6.25 * eb, compressedSize[6]);
    allSize += compressedSize[6];
    auto end = std::chrono::high_resolution_clock::now();
    double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;
    double global_compress_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - split_start).count() * 1e-9;

    double original_size = (double)full_dim_z * full_dim_y * full_dim_x * sizeof(float);
    double CR = original_size / (double)allSize;
    std::cout << "CR: " << CR << std::endl;

    auto global_decompress_start = std::chrono::high_resolution_clock::now();
    decompressed_data = SZ_decompress4De(tmp, szcompressedSize, low_dim_x, low_dim_y, low_dim_z);
    auto sz_deend = std::chrono::high_resolution_clock::now();
    sz_time_taken_decompress = std::chrono::duration_cast<std::chrono::nanoseconds>(sz_deend - global_decompress_start).count() * 1e-9;

    auto low_decompress_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int block = 0; block < 7; ++block) {
        low_deData[block] = SZ_decompress_separated(low_comp[block], low_compressedSize[block], low_dim_x, low_dim_y, low_dim_z);
        scaleData(low_deData[block], low_num_elements, 1.0f / residual_scale);
    }
    auto low_decompress_end_sz = std::chrono::high_resolution_clock::now();
    double low_time_taken_decompress_sz = std::chrono::duration_cast<std::chrono::nanoseconds>(low_decompress_end_sz - low_decompress_start).count() * 1e-9;
    #pragma omp parallel for
    for (int i = 0; i < 3; ++i) {
        int block = single_axis_blocks[i];
        depreprocess_block(block, low_deData[block], decompressed_data, low_de_sub_block[block],
                           low_dim_x, low_dim_y, low_dim_z);
    }
    #pragma omp parallel for
    for (int i = 0; i < 3; ++i) {
        int block = double_axis_blocks[i];
        depreprocess_block_staged(block, low_deData[block], low_de_sub_block[0], low_de_sub_block[1],
                                  low_de_sub_block[3], low_de_sub_block[2], low_de_sub_block[4],
                                  low_de_sub_block[5], low_de_sub_block[block], low_dim_x, low_dim_y, low_dim_z);
    }
    depreprocess_block_staged(6, low_deData[6], low_de_sub_block[0], low_de_sub_block[1],
                              low_de_sub_block[3], low_de_sub_block[2], low_de_sub_block[4],
                              low_de_sub_block[5], low_de_sub_block[6], low_dim_x, low_dim_y, low_dim_z);
    auto low_decompress_end = std::chrono::high_resolution_clock::now();
    double low_time_taken_decompress = std::chrono::duration_cast<std::chrono::nanoseconds>(low_decompress_end - low_decompress_start).count() * 1e-9;

    auto reconstructed_low_decompress_start = std::chrono::high_resolution_clock::now();
    all_low_blocks[0] = decompressed_data;
    merge_sub_blocks_to_full(all_low_blocks, reconstructed_sub_0, low_dim_x, low_dim_y, low_dim_z);
    auto reconstructed_low_decompress_end = std::chrono::high_resolution_clock::now();
    double time_taken_reconstructed_low_decompress = std::chrono::duration_cast<std::chrono::nanoseconds>(reconstructed_low_decompress_end - reconstructed_low_decompress_start).count() * 1e-9;

    auto decompress_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int block = 0; block < 7; ++block) {
        deData[block] = SZ_decompress_separated(comp[block], compressedSize[block], dim_x, dim_y, dim_z);
        scaleData(deData[block], num_elements, 1.0f / residual_scale);
    }
    auto decompress_end_sz = std::chrono::high_resolution_clock::now();
    double time_taken_decompress_sz = std::chrono::duration_cast<std::chrono::nanoseconds>(decompress_end_sz - decompress_start).count() * 1e-9;
    #pragma omp parallel for
    for (int i = 0; i < 3; ++i) {
        int block = single_axis_blocks[i];
        depreprocess_block(block, deData[block], reconstructed_sub_0, de_sub_block[block],
                           dim_x, dim_y, dim_z);
    }
    #pragma omp parallel for
    for (int i = 0; i < 3; ++i) {
        int block = double_axis_blocks[i];
        depreprocess_block_staged(block, deData[block], de_sub_block[0], de_sub_block[1],
                                  de_sub_block[3], de_sub_block[2], de_sub_block[4],
                                  de_sub_block[5], de_sub_block[block], dim_x, dim_y, dim_z);
    }
    depreprocess_block_staged(6, deData[6], de_sub_block[0], de_sub_block[1],
                              de_sub_block[3], de_sub_block[2], de_sub_block[4],
                              de_sub_block[5], de_sub_block[6], dim_x, dim_y, dim_z);
    auto decompress_end = std::chrono::high_resolution_clock::now();
    double time_taken_decompress = std::chrono::duration_cast<std::chrono::nanoseconds>(decompress_end - decompress_start).count() * 1e-9;

    auto reconstructed_full_start = std::chrono::high_resolution_clock::now();
    float* reconstructed_full_data = new float[full_dim_z * full_dim_y * full_dim_x];
    float* all_sub_blocks[8] = {
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
    auto reconstructed_full_end = std::chrono::high_resolution_clock::now();
    double time_taken_reconstructed_full = std::chrono::duration_cast<std::chrono::nanoseconds>(reconstructed_full_end - reconstructed_full_start).count() * 1e-9;
    double global_decompress_time = std::chrono::duration_cast<std::chrono::nanoseconds>(reconstructed_full_end - global_decompress_start).count() * 1e-9;

    // writeBinaryData("ours.raw", reconstructed_full_data, full_dim_z * full_dim_y * full_dim_x);

    double mse_full = 0.0;
    double range_high = -FLT_MAX;
    double range_low = FLT_MAX;
    #pragma omp parallel for reduction(+:mse_full) reduction(max:range_high) reduction(min:range_low)
    for (size_t i = 0; i < full_dim_z * full_dim_y * full_dim_x; ++i) {
        const double original = full_data[i];
        if (original > range_high)
            range_high = original;
        if (original < range_low)
            range_low = original;
        double diff = original - reconstructed_full_data[i];
        mse_full += diff * diff;
    }
    mse_full /= (full_dim_z * full_dim_y * full_dim_x);
    double range_full = range_high - range_low;
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
              << sz_time_taken_split + low_time_taken + time_taken_reconstructed_low + time_taken << " sec" << std::endl;
    
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
