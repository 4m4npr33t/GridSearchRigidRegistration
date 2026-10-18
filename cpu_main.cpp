// Simple CPU-only 3-D rigid registration using normalized cross-correlation.
// This file needs only a C++17 compiler: no CUDA and no third-party libraries.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

constexpr double PI = 3.14159265358979323846;

struct Volume {
    int width = 0;
    int height = 0;
    int depth = 0;
    std::vector<float> voxels;

    std::size_t size() const {
        return static_cast<std::size_t>(width) * height * depth;
    }
};

// Translation is measured in voxels and rotation in degrees.
struct Transform {
    double tx = 0.0, ty = 0.0, tz = 0.0;
    double rx = 0.0, ry = 0.0, rz = 0.0;
};

struct MatrixTransform {
    double m00, m01, m02;
    double m10, m11, m12;
    double m20, m21, m22;
    double tx, ty, tz;
};

struct SearchOptions {
    double max_translation = 12.0;
    double max_rotation = 12.0;
    int max_iterations = 10;
};

struct HistoryRow {
    int level = 0;
    int iteration = 0;
    double step = 0.0;
    Transform transform;
    double ncc = -std::numeric_limits<double>::infinity();
};

struct Result {
    Transform transform;
    double ncc = -std::numeric_limits<double>::infinity();
    double milliseconds = 0.0;
    std::vector<HistoryRow> history;
};

struct Arguments {
    std::string fixed_path;
    std::string moving_path;
    std::string csv_path = "cpu_registration_results.csv";
    int width = 0, height = 0, depth = 0;
    double max_translation = 12.0;
    double max_rotation = 12.0;
    int repeats = 3;
    unsigned seed = 42;
    bool self_test = false;
    bool help = false;
};

std::size_t index_3d(int x, int y, int z, int width, int height) {
    return (static_cast<std::size_t>(z) * height + y) * width + x;
}

MatrixTransform make_matrix(const Transform& p) {
    const double x = p.rx * PI / 180.0;
    const double y = p.ry * PI / 180.0;
    const double z = p.rz * PI / 180.0;
    const double cx = std::cos(x), sx = std::sin(x);
    const double cy = std::cos(y), sy = std::sin(y);
    const double cz = std::cos(z), sz = std::sin(z);

    // Column vectors, with R = Rz * Ry * Rx.
    return {cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
            sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
            -sy,     cy * sx,                  cy * cx,
            p.tx, p.ty, p.tz};
}

float sample_trilinear(const Volume& volume, double x, double y, double z) {
    if (x < 0.0 || y < 0.0 || z < 0.0 || x > volume.width - 1.0 ||
        y > volume.height - 1.0 || z > volume.depth - 1.0) {
        return 0.0f;
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int z0 = static_cast<int>(std::floor(z));
    const int x1 = std::min(x0 + 1, volume.width - 1);
    const int y1 = std::min(y0 + 1, volume.height - 1);
    const int z1 = std::min(z0 + 1, volume.depth - 1);
    const double fx = x - x0, fy = y - y0, fz = z - z0;

    auto at = [&](int xi, int yi, int zi) {
        return volume.voxels[index_3d(xi, yi, zi, volume.width, volume.height)];
    };

    const double c00 = at(x0, y0, z0) * (1.0 - fx) + at(x1, y0, z0) * fx;
    const double c10 = at(x0, y1, z0) * (1.0 - fx) + at(x1, y1, z0) * fx;
    const double c01 = at(x0, y0, z1) * (1.0 - fx) + at(x1, y0, z1) * fx;
    const double c11 = at(x0, y1, z1) * (1.0 - fx) + at(x1, y1, z1) * fx;
    const double c0 = c00 * (1.0 - fy) + c10 * fy;
    const double c1 = c01 * (1.0 - fy) + c11 * fy;
    return static_cast<float>(c0 * (1.0 - fz) + c1 * fz);
}

void map_fixed_to_moving(int x, int y, int z, const Volume& volume,
                         const MatrixTransform& t, double& mx, double& my, double& mz) {
    const double cx = (volume.width - 1) * 0.5;
    const double cy = (volume.height - 1) * 0.5;
    const double cz = (volume.depth - 1) * 0.5;
    const double px = x - cx, py = y - cy, pz = z - cz;

    mx = t.m00 * px + t.m01 * py + t.m02 * pz + cx + t.tx;
    my = t.m10 * px + t.m11 * py + t.m12 * pz + cy + t.ty;
    mz = t.m20 * px + t.m21 * py + t.m22 * pz + cz + t.tz;
}

double normalized_cross_correlation(const Volume& fixed, const Volume& moving,
                                    const Transform& parameters) {
    const MatrixTransform transform = make_matrix(parameters);
    double sum_fixed = 0.0, sum_moving = 0.0;
    double sum_fixed_squared = 0.0, sum_moving_squared = 0.0;
    double sum_product = 0.0;

    for (int z = 0; z < fixed.depth; ++z) {
        for (int y = 0; y < fixed.height; ++y) {
            for (int x = 0; x < fixed.width; ++x) {
                double mx, my, mz;
                map_fixed_to_moving(x, y, z, fixed, transform, mx, my, mz);
                const double f = fixed.voxels[index_3d(x, y, z, fixed.width, fixed.height)];
                const double m = sample_trilinear(moving, mx, my, mz);
                sum_fixed += f;
                sum_moving += m;
                sum_fixed_squared += f * f;
                sum_moving_squared += m * m;
                sum_product += f * m;
            }
        }
    }

    const double count = static_cast<double>(fixed.size());
    const double numerator = sum_product - sum_fixed * sum_moving / count;
    const double fixed_variance = sum_fixed_squared - sum_fixed * sum_fixed / count;
    const double moving_variance = sum_moving_squared - sum_moving * sum_moving / count;
    const double product = fixed_variance * moving_variance;
    const double denominator = std::sqrt(product > 0.0 ? product : 0.0);
    return denominator > 1e-15
               ? numerator / denominator
               : -std::numeric_limits<double>::infinity();
}

double clamp_parameter(double value, double limit) {
    return std::max(-limit, std::min(limit, value));
}

std::vector<Transform> make_candidates(const Transform& center, double step,
                                       const SearchOptions& options) {
    std::vector<Transform> candidates;
    candidates.reserve(729); // 3 choices for each of 6 parameters = 3^6.

    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
    for (int drz = -1; drz <= 1; ++drz)
    for (int dry = -1; dry <= 1; ++dry)
    for (int drx = -1; drx <= 1; ++drx) {
        Transform p;
        p.tx = clamp_parameter(center.tx + dx * step, options.max_translation);
        p.ty = clamp_parameter(center.ty + dy * step, options.max_translation);
        p.tz = clamp_parameter(center.tz + dz * step, options.max_translation);
        p.rx = clamp_parameter(center.rx + drx * step, options.max_rotation);
        p.ry = clamp_parameter(center.ry + dry * step, options.max_rotation);
        p.rz = clamp_parameter(center.rz + drz * step, options.max_rotation);
        candidates.push_back(p);
    }
    return candidates;
}

Result register_volumes(const Volume& fixed, const Volume& moving,
                        const SearchOptions& options) {
    Result result;
    const std::array<double, 3> steps{4.0, 2.0, 1.0};
    const auto start = std::chrono::steady_clock::now();

    for (int level = 0; level < static_cast<int>(steps.size()); ++level) {
        for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
            const auto candidates = make_candidates(result.transform, steps[level], options);
            std::size_t best_index = 0;
            double best_ncc = -std::numeric_limits<double>::infinity();

            for (std::size_t i = 0; i < candidates.size(); ++i) {
                const double score = normalized_cross_correlation(fixed, moving, candidates[i]);
                if (score > best_ncc) {
                    best_ncc = score;
                    best_index = i;
                }
            }

            const Transform previous_transform = result.transform;
            const double previous_ncc = result.ncc;
            result.transform = candidates[best_index];
            result.ncc = best_ncc;
            result.history.push_back({level, iteration, steps[level],
                                      result.transform, result.ncc});

            const Transform& a = previous_transform;
            const Transform& b = result.transform;
            const bool unchanged = a.tx == b.tx && a.ty == b.ty && a.tz == b.tz &&
                                   a.rx == b.rx && a.ry == b.ry && a.rz == b.rz;
            if (unchanged || result.ncc <= previous_ncc + 1e-12) {
                break;
            }
        }
    }

    result.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

Volume make_synthetic_volume(int size, unsigned seed) {
    Volume volume{size, size, size,
                  std::vector<float>(static_cast<std::size_t>(size) * size * size)};
    std::mt19937 random(seed);
    std::uniform_real_distribution<double> position(0.2 * size, 0.8 * size);
    std::uniform_real_distribution<double> sigma(2.0, 6.0);
    std::uniform_real_distribution<double> amplitude(0.3, 1.0);
    struct Blob { double x, y, z, sigma, amplitude; };
    std::vector<Blob> blobs;
    for (int i = 0; i < 9; ++i) {
        blobs.push_back({position(random), position(random), position(random),
                         sigma(random), amplitude(random)});
    }

    for (int z = 0; z < size; ++z)
    for (int y = 0; y < size; ++y)
    for (int x = 0; x < size; ++x) {
        double value = 0.0;
        for (const Blob& blob : blobs) {
            const double dx = x - blob.x, dy = y - blob.y, dz = z - blob.z;
            value += blob.amplitude * std::exp(
                -(dx * dx + dy * dy + dz * dz) / (2.0 * blob.sigma * blob.sigma));
        }
        volume.voxels[index_3d(x, y, z, size, size)] = static_cast<float>(value);
    }

    const float maximum = *std::max_element(volume.voxels.begin(), volume.voxels.end());
    for (float& value : volume.voxels) value /= maximum;
    return volume;
}

MatrixTransform inverse_matrix(const Transform& p) {
    const MatrixTransform forward = make_matrix(p);
    MatrixTransform inverse{};
    // R^-1 = R^T and the inverse translation is -R^T t.
    inverse.m00 = forward.m00; inverse.m01 = forward.m10; inverse.m02 = forward.m20;
    inverse.m10 = forward.m01; inverse.m11 = forward.m11; inverse.m12 = forward.m21;
    inverse.m20 = forward.m02; inverse.m21 = forward.m12; inverse.m22 = forward.m22;
    inverse.tx = -(inverse.m00 * p.tx + inverse.m01 * p.ty + inverse.m02 * p.tz);
    inverse.ty = -(inverse.m10 * p.tx + inverse.m11 * p.ty + inverse.m12 * p.tz);
    inverse.tz = -(inverse.m20 * p.tx + inverse.m21 * p.ty + inverse.m22 * p.tz);
    return inverse;
}

Volume create_moving_volume(const Volume& fixed, const Transform& known_transform) {
    Volume moving{fixed.width, fixed.height, fixed.depth, std::vector<float>(fixed.size())};
    const MatrixTransform inverse = inverse_matrix(known_transform);
    for (int z = 0; z < fixed.depth; ++z)
    for (int y = 0; y < fixed.height; ++y)
    for (int x = 0; x < fixed.width; ++x) {
        double fx, fy, fz;
        map_fixed_to_moving(x, y, z, fixed, inverse, fx, fy, fz);
        moving.voxels[index_3d(x, y, z, fixed.width, fixed.height)] =
            sample_trilinear(fixed, fx, fy, fz);
    }
    return moving;
}

Volume load_raw_volume(const std::string& path, int width, int height, int depth) {
    Volume volume{width, height, depth,
                  std::vector<float>(static_cast<std::size_t>(width) * height * depth)};
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open volume: " + path);
    const std::streamoff actual_bytes = input.tellg();
    const std::streamoff expected_bytes =
        static_cast<std::streamoff>(volume.size() * sizeof(float));
    if (actual_bytes != expected_bytes) {
        throw std::runtime_error("Wrong file size for " + path + ": expected " +
                                 std::to_string(static_cast<long long>(expected_bytes)) +
                                 " bytes, received " +
                                 std::to_string(static_cast<long long>(actual_bytes)));
    }
    input.seekg(0);
    input.read(reinterpret_cast<char*>(volume.voxels.data()), expected_bytes);
    if (!input) throw std::runtime_error("Failed to read volume: " + path);
    return volume;
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    auto next_value = [&](int& i) {
        if (++i >= argc) throw std::runtime_error(std::string("Missing value after ") + argv[i - 1]);
        return std::string(argv[i]);
    };
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--fixed") args.fixed_path = next_value(i);
        else if (option == "--moving") args.moving_path = next_value(i);
        else if (option == "--width") args.width = std::stoi(next_value(i));
        else if (option == "--height") args.height = std::stoi(next_value(i));
        else if (option == "--depth") args.depth = std::stoi(next_value(i));
        else if (option == "--max-translation") args.max_translation = std::stod(next_value(i));
        else if (option == "--max-rotation") args.max_rotation = std::stod(next_value(i));
        else if (option == "--repeats") args.repeats = std::stoi(next_value(i));
        else if (option == "--seed") args.seed = static_cast<unsigned>(std::stoul(next_value(i)));
        else if (option == "--csv") args.csv_path = next_value(i);
        else if (option == "--self-test") args.self_test = true;
        else if (option == "--help" || option == "-h") args.help = true;
        else throw std::runtime_error("Unknown option: " + option);
    }
    return args;
}

void print_help() {
    std::cout
        << "CPU-only 3-D rigid NCC registration\n\n"
        << "Synthetic example:\n  cpu_registration [--seed 42] [--repeats 3]\n\n"
        << "Raw float32 input:\n  cpu_registration --fixed fixed.raw --moving moving.raw"
           " --width 64 --height 64 --depth 64\n\n"
        << "Options:\n"
        << "  --max-translation N   Bound in voxels (default 12)\n"
        << "  --max-rotation N      Bound in degrees (default 12)\n"
        << "  --repeats N           Complete timing repetitions (default 3)\n"
        << "  --csv PATH            History output file\n"
        << "  --self-test           Run small CPU tests\n";
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 1
               ? values[middle]
               : (values[middle - 1] + values[middle]) * 0.5;
}

double parameter_error(const Transform& a, const Transform& b) {
    const double translation = std::sqrt((a.tx-b.tx)*(a.tx-b.tx) +
                                         (a.ty-b.ty)*(a.ty-b.ty) +
                                         (a.tz-b.tz)*(a.tz-b.tz));
    const double rotation = std::sqrt((a.rx-b.rx)*(a.rx-b.rx) +
                                      (a.ry-b.ry)*(a.ry-b.ry) +
                                      (a.rz-b.rz)*(a.rz-b.rz));
    return translation + rotation;
}

void write_history(const std::string& path, const Result& result) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot create CSV: " + path);
    output << "backend,level,iteration,translation_step,rotation_step,tx,ty,tz,rx,ry,rz,ncc\n";
    output << std::setprecision(12);
    for (const HistoryRow& row : result.history) {
        output << "CPU," << row.level << ',' << row.iteration << ','
               << row.step << ',' << row.step << ','
               << row.transform.tx << ',' << row.transform.ty << ',' << row.transform.tz << ','
               << row.transform.rx << ',' << row.transform.ry << ',' << row.transform.rz << ','
               << row.ncc << '\n';
    }
}

void run_self_tests() {
    Volume volume{3, 3, 3, std::vector<float>(27)};
    for (int i = 0; i < 27; ++i) volume.voxels[i] = static_cast<float>(i);
    if (std::abs(sample_trilinear(volume, 1.0, 1.0, 1.0) - 13.0f) > 1e-6f)
        throw std::runtime_error("Integer interpolation test failed");
    if (sample_trilinear(volume, -0.1, 1.0, 1.0) != 0.0f)
        throw std::runtime_error("Out-of-bounds interpolation test failed");
    if (std::abs(normalized_cross_correlation(volume, volume, {}) - 1.0) > 1e-12)
        throw std::runtime_error("Identity NCC test failed");

    Volume constant{3, 3, 3, std::vector<float>(27, 1.0f)};
    if (!std::isinf(normalized_cross_correlation(constant, constant, {})))
        throw std::runtime_error("Constant-volume test failed");
    std::cout << "All CPU self-tests passed.\n";
}

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        if (args.help) { print_help(); return 0; }
        if (args.self_test) { run_self_tests(); return 0; }

        const bool has_path = !args.fixed_path.empty() || !args.moving_path.empty();
        if (has_path && (args.fixed_path.empty() || args.moving_path.empty()))
            throw std::runtime_error("--fixed and --moving must be provided together");
        if (has_path && (args.width <= 0 || args.height <= 0 || args.depth <= 0))
            throw std::runtime_error("Raw input requires positive width, height, and depth");
        if (args.max_translation < 0.0 || args.max_rotation < 0.0 || args.repeats <= 0)
            throw std::runtime_error("Bounds must be nonnegative and repeats must be positive");

        Volume fixed;
        Volume moving;
        const Transform truth{3, -2, 1, 4, -3, 2};
        const Transform* truth_pointer = nullptr;
        if (has_path) {
            fixed = load_raw_volume(args.fixed_path, args.width, args.height, args.depth);
            moving = load_raw_volume(args.moving_path, args.width, args.height, args.depth);
        } else {
            fixed = make_synthetic_volume(64, args.seed);
            moving = create_moving_volume(fixed, truth);
            truth_pointer = &truth;
            std::cout << "Synthetic truth: (tx,ty,tz)=(3,-2,1), "
                      << "(rx,ry,rz)=(4,-3,2)\n";
        }

        const SearchOptions options{args.max_translation, args.max_rotation, 10};
        Result result;
        std::vector<double> times;
        for (int repeat = 0; repeat < args.repeats; ++repeat) {
            Result current = register_volumes(fixed, moving, options);
            times.push_back(current.milliseconds);
            if (repeat == 0) result = std::move(current);
        }
        result.milliseconds = median(times);

        std::cout << std::fixed << std::setprecision(4)
                  << "CPU result: (tx,ty,tz)=(" << result.transform.tx << ','
                  << result.transform.ty << ',' << result.transform.tz
                  << "), (rx,ry,rz)=(" << result.transform.rx << ','
                  << result.transform.ry << ',' << result.transform.rz
                  << "), NCC=" << result.ncc
                  << ", median time=" << result.milliseconds << " ms\n";
        if (truth_pointer)
            std::cout << "Combined parameter error: "
                      << parameter_error(result.transform, *truth_pointer) << '\n';

        write_history(args.csv_path, result);
        std::cout << "History written to " << args.csv_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
