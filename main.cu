#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error_ = (call);                                             \
        if (error_ != cudaSuccess)                                               \
            throw std::runtime_error(std::string("CUDA error: ") +              \
                                     cudaGetErrorString(error_) + " at " +       \
                                     __FILE__ + ":" + std::to_string(__LINE__)); \
    } while (false)

constexpr double kPi = 3.14159265358979323846;
constexpr int kThreads = 256;

struct Volume {
    int width = 0, height = 0, depth = 0;
    std::vector<float> data;
    std::size_t voxel_count() const {
        return static_cast<std::size_t>(width) * height * depth;
    }
};

// Transform: rotations are degrees; translations are voxels.
struct Transform {
    double tx = 0, ty = 0, tz = 0;
    double rx = 0, ry = 0, rz = 0;
};

// Kernels receive a precomputed matrix, avoiding trigonometry per voxel.
struct DeviceTransform {
    double m00, m01, m02, m10, m11, m12, m20, m21, m22;
    double tx, ty, tz;
};

struct SearchOptions {
    double max_translation = 12.0;
    double max_rotation = 12.0;
    int max_iterations = 10;
};

struct Evaluation {
    std::vector<double> scores;
    double kernel_ms = 0.0;
};

struct IterationRecord {
    std::string backend;
    int level, iteration;
    double translation_step, rotation_step;
    Transform best;
    double score;
};

struct RegistrationResult {
    Transform transform;
    double score = -std::numeric_limits<double>::infinity();
    double wall_ms = 0.0;
    double kernel_ms = 0.0;
    std::vector<IterationRecord> history;
};

struct Arguments {
    std::string fixed_path, moving_path;
    std::string csv_path = "registration_results.csv";
    int width = 0, height = 0, depth = 0;
    double max_translation = 12.0, max_rotation = 12.0;
    int repeats = 3;
    unsigned seed = 42;
    bool self_test = false, help = false;
};

static std::size_t index3(int x, int y, int z, int w, int h) {
    return (static_cast<std::size_t>(z) * h + y) * w + x;
}

static DeviceTransform prepare(const Transform& p) {
    const double x = p.rx * kPi / 180.0;
    const double y = p.ry * kPi / 180.0;
    const double z = p.rz * kPi / 180.0;
    const double cx = std::cos(x), sx = std::sin(x);
    const double cy = std::cos(y), sy = std::sin(y);
    const double cz = std::cos(z), sz = std::sin(z);

    // Column-vector convention: R = Rz * Ry * Rx.
    return {cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
            sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
            -sy,     cy * sx,                  cy * cx,
            p.tx, p.ty, p.tz};
}

__host__ __device__ static float sample_trilinear(const float* v, int w, int h,
                                                   int d, double x, double y,
                                                   double z) {
    if (x < 0.0 || y < 0.0 || z < 0.0 || x > w - 1.0 || y > h - 1.0 || z > d - 1.0)
        return 0.0f;
    int x0 = static_cast<int>(floor(x)), y0 = static_cast<int>(floor(y));
    int z0 = static_cast<int>(floor(z));
    int x1 = x0 + 1 < w ? x0 + 1 : w - 1;
    int y1 = y0 + 1 < h ? y0 + 1 : h - 1;
    int z1 = z0 + 1 < d ? z0 + 1 : d - 1;
    double fx = x - x0, fy = y - y0, fz = z - z0;
    std::size_t i000 = (static_cast<std::size_t>(z0) * h + y0) * w + x0;
    std::size_t i100 = (static_cast<std::size_t>(z0) * h + y0) * w + x1;
    std::size_t i010 = (static_cast<std::size_t>(z0) * h + y1) * w + x0;
    std::size_t i110 = (static_cast<std::size_t>(z0) * h + y1) * w + x1;
    std::size_t i001 = (static_cast<std::size_t>(z1) * h + y0) * w + x0;
    std::size_t i101 = (static_cast<std::size_t>(z1) * h + y0) * w + x1;
    std::size_t i011 = (static_cast<std::size_t>(z1) * h + y1) * w + x0;
    std::size_t i111 = (static_cast<std::size_t>(z1) * h + y1) * w + x1;
    double c00 = v[i000] * (1 - fx) + v[i100] * fx;
    double c10 = v[i010] * (1 - fx) + v[i110] * fx;
    double c01 = v[i001] * (1 - fx) + v[i101] * fx;
    double c11 = v[i011] * (1 - fx) + v[i111] * fx;
    double c0 = c00 * (1 - fy) + c10 * fy;
    double c1 = c01 * (1 - fy) + c11 * fy;
    return static_cast<float>(c0 * (1 - fz) + c1 * fz);
}

__host__ __device__ static void mapped_coordinate(std::size_t i, int w, int h, int d,
                                                   const DeviceTransform& t,
                                                   double& mx, double& my, double& mz) {
    int x = static_cast<int>(i % w);
    int y = static_cast<int>((i / w) % h);
    int z = static_cast<int>(i / (static_cast<std::size_t>(w) * h));
    double cx = (w - 1) * 0.5, cy = (h - 1) * 0.5, cz = (d - 1) * 0.5;
    double px = x - cx, py = y - cy, pz = z - cz;
    mx = t.m00 * px + t.m01 * py + t.m02 * pz + cx + t.tx;
    my = t.m10 * px + t.m11 * py + t.m12 * pz + cy + t.ty;
    mz = t.m20 * px + t.m21 * py + t.m22 * pz + cz + t.tz;
}

__host__ __device__ static double finish_ncc(double sf, double sm, double sff,
                                              double smm, double sfm, double n) {
    double numerator = sfm - sf * sm / n;
    double vf = sff - sf * sf / n;
    double vm = smm - sm * sm / n;
    double variance_product = vf * vm;
    double denominator = sqrt(variance_product > 0.0 ? variance_product : 0.0);
    return denominator > 1e-15 ? numerator / denominator : -INFINITY;
}

static double ncc_cpu(const Volume& fixed, const Volume& moving,
                      const DeviceTransform& t) {
    double sf = 0, sm = 0, sff = 0, smm = 0, sfm = 0;
    for (std::size_t i = 0; i < fixed.voxel_count(); ++i) {
        double mx, my, mz;
        mapped_coordinate(i, fixed.width, fixed.height, fixed.depth, t, mx, my, mz);
        double f = fixed.data[i];
        double m = sample_trilinear(moving.data.data(), moving.width, moving.height,
                                    moving.depth, mx, my, mz);
        sf += f; sm += m; sff += f * f; smm += m * m; sfm += f * m;
    }
    return finish_ncc(sf, sm, sff, smm, sfm,
                      static_cast<double>(fixed.voxel_count()));
}

__global__ void naive_ncc_kernel(const float* fixed, const float* moving, int w, int h,
                                 int d, const DeviceTransform* transforms, int count,
                                 double* scores) {
    int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= count) return;
    DeviceTransform t = transforms[candidate];
    double sf = 0, sm = 0, sff = 0, smm = 0, sfm = 0;
    std::size_t n = static_cast<std::size_t>(w) * h * d;
    for (std::size_t i = 0; i < n; ++i) {
        double mx, my, mz;
        mapped_coordinate(i, w, h, d, t, mx, my, mz);
        double f = fixed[i], m = sample_trilinear(moving, w, h, d, mx, my, mz);
        sf += f; sm += m; sff += f * f; smm += m * m; sfm += f * m;
    }
    scores[candidate] = finish_ncc(sf, sm, sff, smm, sfm, static_cast<double>(n));
}

__global__ void optimized_ncc_kernel(const float* fixed, const float* moving, int w,
                                     int h, int d, const DeviceTransform* transforms,
                                     int count, double* scores) {
    int candidate = blockIdx.x;
    if (candidate >= count) return;
    extern __shared__ double shared[];
    double* sf = shared;
    double* sm = sf + blockDim.x;
    double* sff = sm + blockDim.x;
    double* smm = sff + blockDim.x;
    double* sfm = smm + blockDim.x;
    double a = 0, b = 0, aa = 0, bb = 0, ab = 0;
    std::size_t n = static_cast<std::size_t>(w) * h * d;
    DeviceTransform t = transforms[candidate];
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
        double mx, my, mz;
        mapped_coordinate(i, w, h, d, t, mx, my, mz);
        double f = fixed[i], m = sample_trilinear(moving, w, h, d, mx, my, mz);
        a += f; b += m; aa += f * f; bb += m * m; ab += f * m;
    }
    int lane = threadIdx.x;
    sf[lane] = a; sm[lane] = b; sff[lane] = aa; smm[lane] = bb; sfm[lane] = ab;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            sf[lane] += sf[lane + stride]; sm[lane] += sm[lane + stride];
            sff[lane] += sff[lane + stride]; smm[lane] += smm[lane + stride];
            sfm[lane] += sfm[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0)
        scores[candidate] = finish_ncc(sf[0], sm[0], sff[0], smm[0], sfm[0],
                                       static_cast<double>(n));
}

class GpuEvaluator {
public:
    GpuEvaluator(const Volume& fixed, const Volume& moving, bool optimized)
        : volume_(fixed), optimized_(optimized) {
        std::size_t bytes = fixed.voxel_count() * sizeof(float);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_fixed_), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_moving_), bytes));
        CUDA_CHECK(cudaMemcpy(d_fixed_, fixed.data.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_moving_, moving.data.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventCreate(&start_)); CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuEvaluator() {
        cudaFree(d_fixed_); cudaFree(d_moving_); cudaFree(d_transforms_); cudaFree(d_scores_);
        cudaEventDestroy(start_); cudaEventDestroy(stop_);
    }
    Evaluation operator()(const std::vector<Transform>& candidates) {
        std::vector<DeviceTransform> prepared;
        prepared.reserve(candidates.size());
        for (const auto& c : candidates) prepared.push_back(prepare(c));
        ensure_capacity(candidates.size());
        CUDA_CHECK(cudaMemcpy(d_transforms_, prepared.data(), prepared.size() * sizeof(DeviceTransform),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start_));
        if (optimized_)
            optimized_ncc_kernel<<<static_cast<int>(candidates.size()), kThreads,
                5 * kThreads * sizeof(double)>>>(d_fixed_, d_moving_, volume_.width,
                volume_.height, volume_.depth, d_transforms_, static_cast<int>(candidates.size()), d_scores_);
        else
            naive_ncc_kernel<<<(static_cast<int>(candidates.size()) + kThreads - 1) / kThreads,
                kThreads>>>(d_fixed_, d_moving_, volume_.width, volume_.height, volume_.depth,
                d_transforms_, static_cast<int>(candidates.size()), d_scores_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_)); CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        Evaluation result;
        result.scores.resize(candidates.size()); result.kernel_ms = ms;
        CUDA_CHECK(cudaMemcpy(result.scores.data(), d_scores_, result.scores.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        return result;
    }
private:
    void ensure_capacity(std::size_t count) {
        if (count <= capacity_) return;
        cudaFree(d_transforms_); cudaFree(d_scores_);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_transforms_), count * sizeof(DeviceTransform)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scores_), count * sizeof(double)));
        capacity_ = count;
    }
    Volume volume_;
    bool optimized_;
    float *d_fixed_ = nullptr, *d_moving_ = nullptr;
    DeviceTransform* d_transforms_ = nullptr;
    double* d_scores_ = nullptr;
    std::size_t capacity_ = 0;
    cudaEvent_t start_{}, stop_{};
};

static Evaluation evaluate_cpu(const Volume& fixed, const Volume& moving,
                               const std::vector<Transform>& candidates) {
    Evaluation e; e.scores.reserve(candidates.size());
    auto begin = std::chrono::steady_clock::now();
    for (const auto& c : candidates) e.scores.push_back(ncc_cpu(fixed, moving, prepare(c)));
    e.kernel_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    return e;
}

static double clamp_value(double x, double limit) {
    return std::max(-limit, std::min(limit, x));
}

static std::vector<Transform> make_candidates(const Transform& center, double ts,
                                              double rs, const SearchOptions& options) {
    std::vector<Transform> result;
    result.reserve(729);
    for (int iz = -1; iz <= 1; ++iz) for (int iy = -1; iy <= 1; ++iy)
    for (int ix = -1; ix <= 1; ++ix) for (int rz = -1; rz <= 1; ++rz)
    for (int ry = -1; ry <= 1; ++ry) for (int rx = -1; rx <= 1; ++rx) {
        Transform p;
        p.tx = clamp_value(center.tx + ix * ts, options.max_translation);
        p.ty = clamp_value(center.ty + iy * ts, options.max_translation);
        p.tz = clamp_value(center.tz + iz * ts, options.max_translation);
        p.rx = clamp_value(center.rx + rx * rs, options.max_rotation);
        p.ry = clamp_value(center.ry + ry * rs, options.max_rotation);
        p.rz = clamp_value(center.rz + rz * rs, options.max_rotation);
        result.push_back(p);
    }
    return result;
}

template <class Evaluator>
static RegistrationResult register_volume(const std::string& name, Evaluator&& evaluate,
                                          const SearchOptions& options) {
    RegistrationResult result;
    auto wall_start = std::chrono::steady_clock::now();
    const std::array<double, 3> steps{4.0, 2.0, 1.0};
    for (int level = 0; level < 3; ++level) {
        for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
            auto candidates = make_candidates(result.transform, steps[level], steps[level], options);
            Evaluation e = evaluate(candidates);
            result.kernel_ms += e.kernel_ms;
            std::size_t best = 0;
            for (std::size_t i = 1; i < e.scores.size(); ++i)
                if (e.scores[i] > e.scores[best]) best = i;
            Transform old = result.transform;
            double previous = result.score;
            result.transform = candidates[best]; result.score = e.scores[best];
            result.history.push_back({name, level, iteration, steps[level], steps[level],
                                      result.transform, result.score});
            bool unchanged = result.transform.tx == old.tx && result.transform.ty == old.ty &&
                result.transform.tz == old.tz && result.transform.rx == old.rx &&
                result.transform.ry == old.ry && result.transform.rz == old.rz;
            if (unchanged || result.score <= previous + 1e-12) break;
        }
    }
    result.wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_start).count();
    return result;
}

static Volume make_synthetic(int size, unsigned seed) {
    Volume v{size, size, size, std::vector<float>(static_cast<std::size_t>(size) * size * size)};
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pos(0.2 * size, 0.8 * size);
    std::uniform_real_distribution<double> sigma(2.0, 6.0);
    std::uniform_real_distribution<double> amplitude(0.3, 1.0);
    struct Blob { double x, y, z, s, a; };
    std::vector<Blob> blobs;
    for (int i = 0; i < 9; ++i) blobs.push_back({pos(rng), pos(rng), pos(rng), sigma(rng), amplitude(rng)});
    for (int z = 0; z < size; ++z) for (int y = 0; y < size; ++y)
    for (int x = 0; x < size; ++x) {
        double value = 0;
        for (const auto& b : blobs) {
            double dx = x - b.x, dy = y - b.y, dz = z - b.z;
            value += b.a * std::exp(-(dx * dx + dy * dy + dz * dz) / (2 * b.s * b.s));
        }
        v.data[index3(x, y, z, size, size)] = static_cast<float>(value);
    }
    float maximum = *std::max_element(v.data.begin(), v.data.end());
    for (float& x : v.data) x /= maximum;
    return v;
}

static DeviceTransform inverse_transform(const Transform& p) {
    DeviceTransform a = prepare(p), b{};
    // Inverse rotation is transpose; inverse translation is -R^T t.
    b.m00 = a.m00; b.m01 = a.m10; b.m02 = a.m20;
    b.m10 = a.m01; b.m11 = a.m11; b.m12 = a.m21;
    b.m20 = a.m02; b.m21 = a.m12; b.m22 = a.m22;
    b.tx = -(b.m00 * p.tx + b.m01 * p.ty + b.m02 * p.tz);
    b.ty = -(b.m10 * p.tx + b.m11 * p.ty + b.m12 * p.tz);
    b.tz = -(b.m20 * p.tx + b.m21 * p.ty + b.m22 * p.tz);
    return b;
}

static Volume warp(const Volume& input, const Transform& registration_transform) {
    Volume out{input.width, input.height, input.depth,
               std::vector<float>(input.voxel_count())};
    DeviceTransform inverse = inverse_transform(registration_transform);
    for (std::size_t i = 0; i < out.voxel_count(); ++i) {
        double x, y, z;
        mapped_coordinate(i, out.width, out.height, out.depth, inverse, x, y, z);
        out.data[i] = sample_trilinear(input.data.data(), input.width, input.height,
                                       input.depth, x, y, z);
    }
    return out;
}

static Volume load_raw(const std::string& path, int w, int h, int d) {
    Volume v{w, h, d, std::vector<float>(static_cast<std::size_t>(w) * h * d)};
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open raw volume: " + path);
    auto actual = file.tellg();
    auto expected = static_cast<std::streamoff>(v.voxel_count() * sizeof(float));
    if (actual != expected)
        throw std::runtime_error("Raw file size mismatch for " + path + ": expected " +
                                 std::to_string(static_cast<long long>(expected)) + " bytes, got " +
                                 std::to_string(static_cast<long long>(actual)));
    file.seekg(0); file.read(reinterpret_cast<char*>(v.data.data()), expected);
    if (!file) throw std::runtime_error("Failed while reading: " + path);
    return v;
}

static Arguments parse_arguments(int argc, char** argv) {
    Arguments a;
    auto value = [&](int& i) -> std::string {
        if (++i >= argc) throw std::runtime_error(std::string("Missing value after ") + argv[i - 1]);
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--fixed") a.fixed_path = value(i);
        else if (key == "--moving") a.moving_path = value(i);
        else if (key == "--width") a.width = std::stoi(value(i));
        else if (key == "--height") a.height = std::stoi(value(i));
        else if (key == "--depth") a.depth = std::stoi(value(i));
        else if (key == "--max-translation") a.max_translation = std::stod(value(i));
        else if (key == "--max-rotation") a.max_rotation = std::stod(value(i));
        else if (key == "--repeats") a.repeats = std::stoi(value(i));
        else if (key == "--seed") a.seed = static_cast<unsigned>(std::stoul(value(i)));
        else if (key == "--csv") a.csv_path = value(i);
        else if (key == "--self-test") a.self_test = true;
        else if (key == "--help" || key == "-h") a.help = true;
        else throw std::runtime_error("Unknown argument: " + key);
    }
    return a;
}

static void print_help() {
    std::cout << "3-D rigid NCC registration: CPU vs naive CUDA vs optimized CUDA\n\n"
              << "Synthetic demo:\n  cuda_registration [--seed 42] [--repeats 3]\n\n"
              << "Raw float32 volumes:\n  cuda_registration --fixed fixed.raw --moving moving.raw"
                 " --width 64 --height 64 --depth 64\n\n"
              << "Options:\n  --max-translation N   Search bound in voxels (default 12)\n"
              << "  --max-rotation N      Search bound in degrees (default 12)\n"
              << "  --csv PATH            History CSV (default registration_results.csv)\n"
              << "  --self-test           Run quick interpolation/NCC tests\n";
}

static double transform_error(const Transform& a, const Transform& b) {
    double dt = std::sqrt((a.tx-b.tx)*(a.tx-b.tx)+(a.ty-b.ty)*(a.ty-b.ty)+(a.tz-b.tz)*(a.tz-b.tz));
    double dr = std::sqrt((a.rx-b.rx)*(a.rx-b.rx)+(a.ry-b.ry)*(a.ry-b.ry)+(a.rz-b.rz)*(a.rz-b.rz));
    return dt + dr;
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    std::size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) * 0.5;
}

static double maximum_score_difference(const std::vector<double>& a,
                                       const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::runtime_error("score array sizes differ");
    double difference = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::isinf(a[i]) && std::isinf(b[i])) continue;
        difference = std::max(difference, std::abs(a[i] - b[i]));
    }
    return difference;
}

static void self_test() {
    Volume v{3, 3, 3, std::vector<float>(27)};
    for (int i = 0; i < 27; ++i) v.data[i] = static_cast<float>(i);
    if (std::abs(sample_trilinear(v.data.data(), 3, 3, 3, 1, 1, 1) - 13.0f) > 1e-6)
        throw std::runtime_error("integer interpolation test failed");
    if (sample_trilinear(v.data.data(), 3, 3, 3, -0.1, 1, 1) != 0.0f)
        throw std::runtime_error("out-of-bounds interpolation test failed");
    double identity = ncc_cpu(v, v, prepare({}));
    if (std::abs(identity - 1.0) > 1e-12) throw std::runtime_error("identity NCC test failed");
    int devices = 0; cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaSuccess && devices > 0) {
        std::vector<Transform> candidates{{}, {1, 0, 0, 0, 0, 0}};
        auto cpu = evaluate_cpu(v, v, candidates);
        GpuEvaluator naive(v, v, false), optimized(v, v, true);
        auto n = naive(candidates), o = optimized(candidates);
        for (std::size_t i = 0; i < candidates.size(); ++i)
            if (std::abs(cpu.scores[i] - n.scores[i]) > 1e-9 ||
                std::abs(cpu.scores[i] - o.scores[i]) > 1e-9)
                throw std::runtime_error("CPU/CUDA agreement test failed");
    } else {
        std::cout << "CUDA test skipped: no CUDA device available.\n";
        cudaGetLastError();
    }
    std::cout << "All self-tests passed.\n";
}

static void write_csv(const std::string& path, const std::vector<RegistrationResult>& results) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot create CSV: " + path);
    out << "backend,level,iteration,translation_step,rotation_step,tx,ty,tz,rx,ry,rz,ncc\n";
    out << std::setprecision(12);
    for (const auto& result : results) for (const auto& r : result.history)
        out << r.backend << ',' << r.level << ',' << r.iteration << ','
            << r.translation_step << ',' << r.rotation_step << ','
            << r.best.tx << ',' << r.best.ty << ',' << r.best.tz << ','
            << r.best.rx << ',' << r.best.ry << ',' << r.best.rz << ',' << r.score << '\n';
}

static void print_result(const RegistrationResult& r, const Transform* truth) {
    std::cout << std::fixed << std::setprecision(4)
              << r.history.front().backend << ": (tx,ty,tz)=(" << r.transform.tx << ','
              << r.transform.ty << ',' << r.transform.tz << ")  (rx,ry,rz)=("
              << r.transform.rx << ',' << r.transform.ry << ',' << r.transform.rz
              << ")  NCC=" << r.score << "  wall=" << r.wall_ms << " ms";
    if (r.kernel_ms > 0) std::cout << "  kernels=" << r.kernel_ms << " ms";
    if (truth) std::cout << "  parameter error=" << transform_error(r.transform, *truth);
    std::cout << '\n';
}

int main(int argc, char** argv) {
    try {
        Arguments args = parse_arguments(argc, argv);
        if (args.help) { print_help(); return 0; }
        if (args.self_test) { self_test(); return 0; }
        bool any_path = !args.fixed_path.empty() || !args.moving_path.empty();
        if (any_path && (args.fixed_path.empty() || args.moving_path.empty()))
            throw std::runtime_error("--fixed and --moving must be supplied together");
        if (any_path && (args.width <= 0 || args.height <= 0 || args.depth <= 0))
            throw std::runtime_error("positive --width, --height, and --depth are required for raw files");
        if (args.max_translation < 0 || args.max_rotation < 0 || args.repeats <= 0)
            throw std::runtime_error("search bounds must be nonnegative and repeats must be positive");

        Volume fixed, moving;
        Transform truth{3, -2, 1, 4, -3, 2};
        Transform* truth_ptr = nullptr;
        if (any_path) {
            fixed = load_raw(args.fixed_path, args.width, args.height, args.depth);
            moving = load_raw(args.moving_path, args.width, args.height, args.depth);
        } else {
            fixed = make_synthetic(64, args.seed);
            moving = warp(fixed, truth);
            truth_ptr = &truth;
            std::cout << "Synthetic truth: (tx,ty,tz)=(3,-2,1), (rx,ry,rz)=(4,-3,2)\n";
        }

        int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices == 0) throw std::runtime_error("No CUDA-capable GPU was found");
        cudaDeviceProp properties{}; CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        std::cout << "GPU: " << properties.name << "\nVolume: " << fixed.width << 'x'
                  << fixed.height << 'x' << fixed.depth << "\n";

        SearchOptions options{args.max_translation, args.max_rotation, 10};
        // Warm-up removes CUDA context creation and JIT overhead from measurements.
        std::vector<Transform> warmup{{}};
        { GpuEvaluator warm(fixed, moving, true); warm(warmup); }

        auto cpu = register_volume("CPU", [&](const auto& c) {
            return evaluate_cpu(fixed, moving, c); }, options);
        GpuEvaluator naive_eval(fixed, moving, false);
        auto naive = register_volume("Naive CUDA", [&](const auto& c) {
            Evaluation best;
            std::vector<double> timings;
            for (int i = 0; i < args.repeats; ++i) {
                Evaluation current = naive_eval(c);
                if (i == 0) best = current;
                timings.push_back(current.kernel_ms);
            }
            best.kernel_ms = median(timings);
            return best;
        }, options);
        GpuEvaluator optimized_eval(fixed, moving, true);
        auto optimized = register_volume("Optimized CUDA", [&](const auto& c) {
            Evaluation best;
            std::vector<double> timings;
            for (int i = 0; i < args.repeats; ++i) {
                Evaluation current = optimized_eval(c);
                if (i == 0) best = current;
                timings.push_back(current.kernel_ms);
            }
            best.kernel_ms = median(timings);
            return best;
        }, options);

        print_result(cpu, truth_ptr); print_result(naive, truth_ptr); print_result(optimized, truth_ptr);
        // Evaluate an identical final neighborhood to compare numerical results directly.
        auto check_candidates = make_candidates(cpu.transform, 1.0, 1.0, options);
        auto check_cpu = evaluate_cpu(fixed, moving, check_candidates);
        auto check_naive = naive_eval(check_candidates);
        auto check_optimized = optimized_eval(check_candidates);
        std::cout << "CPU / optimized wall speedup: " << cpu.wall_ms / optimized.wall_ms << "x\n"
                  << "Naive / optimized kernel speedup: " << naive.kernel_ms / optimized.kernel_ms << "x\n"
                  << "Max score difference, CPU vs naive: "
                  << maximum_score_difference(check_cpu.scores, check_naive.scores) << "\n"
                  << "Max score difference, CPU vs optimized: "
                  << maximum_score_difference(check_cpu.scores, check_optimized.scores) << "\n";
        write_csv(args.csv_path, {cpu, naive, optimized});
        std::cout << "History written to " << args.csv_path << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
