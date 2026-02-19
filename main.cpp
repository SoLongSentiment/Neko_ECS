#include "Neko_StringSort.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip> // For formatting
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono;

// System prepare
void optimize_system_for_bench()
{
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
    //SetProcessAffinityMask(GetCurrentProcess(), 1);
}

std::string format_size(size_t bytes)
{
    if (bytes < 1024)
        return std::to_string(bytes) + " B";
    if (bytes < 1024 * 1024)
        return std::to_string(bytes / 1024) + " KB";
    return std::to_string(bytes / 1024 / 1024) + " MB";
}

double get_median(std::vector<double> &times)
{
    if (times.empty())
        return 0;
    std::sort(times.begin(), times.end());
    if (times.size() % 2 == 0)
        return (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2;
    return times[times.size() / 2];
}

double get_stable_average(std::vector<double> &times)
{
    if (times.empty())
        return 0;
    std::sort(times.begin(), times.end());

    size_t trim = times.size() / 10;
    double sum = 0;
    int count = 0;
    for (size_t i = trim; i < times.size() - trim; ++i)
    {
        sum += times[i];
        count++;
    }
    return sum / count;
}

void print_benchmark_header(int id, const std::string &title,
                            const std::string &description)
{
    std::cout << "\n[" << id << "] BENCHMARK: " << title << "\n";
    std::cout << "Description: " << description << "\n";
    std::cout << std::string(60, '-') << "\n";
}

void print_result(const std::string &label, double value,
                  const std::string &unit = "ms")
{

    std::cout << std::left << std::setw(20) << "  " + label << ": "
              << std::right << std::setw(10) << std::fixed
              << std::setprecision(4) << value << " " << unit << "\n";
}

void bench_asset_sorting()
{
    print_benchmark_header(
        2, "2M ASSET TICKETS SORT",
        "Radix (Tickets) vs std::sort (Strings) | REVERSE ORDER");

    const int N = 2000000;
    const int RUN_COUNT = 1;

    String_lib::UniversalArena<16> string_arena(N * 64);
    std::vector<SoftTicket> tickets_master;
    tickets_master.reserve(N);

    srand(42);
    for (int i = 0; i < N; ++i)
    {
        std::string raw = "Asset_Name_Stress_Test_Long_Prefix_" +
                          std::to_string(rand() % 1000) + "_" +
                          std::to_string(i);
        char *ptr = static_cast<char *>(string_arena.alloc(raw.size() + 1));
        memcpy(ptr, raw.c_str(), raw.size());
        ptr[raw.size()] = '\0';
        tickets_master.push_back(
            {String_lib::StringView(ptr, raw.size()), (size_t)i});
    }

    std::vector<double> engine_times;
    {
        auto test_container = tickets_master;
        auto t1 = high_resolution_clock::now();
        Radix_Internal::RadSort(test_container.data(),
                                test_container.data() + test_container.size(),
                                0);
        auto t2 = high_resolution_clock::now();
        engine_times.push_back(duration<double, std::milli>(t2 - t1).count());
    }

    std::vector<double> std_times;
    {
        auto test_container = tickets_master;
        auto t1 = high_resolution_clock::now();
        std::sort(test_container.begin(), test_container.end(),
                  std::greater<SoftTicket>());
        auto t2 = high_resolution_clock::now();
        std_times.push_back(duration<double, std::milli>(t2 - t1).count());
    }

    auto get_stable_avg = [](std::vector<double> &v)
    {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    double my_avg = get_stable_avg(engine_times);
    double std_avg = get_stable_avg(std_times);

    print_result("RADIX (AoS)", my_avg, "ms");
    print_result("STD Sort (AoS)", std_avg, "ms");

    String_lib::UniversalArena<16> data_arena(1024 * 1024 * 1024);
    Registry<16> reg_stub(data_arena);
    auto &soa_set = reg_stub.get_set<SoftTicket>(N + 1);
    auto ticket_group = reg_stub.group<SoftTicket>();
    ticket_group.set_explicit_size(N);

    for (const auto &t : tickets_master)
    {
        soa_set.add(static_cast<uint32_t>(t.original_idx), t.view,
                    t.original_idx);
    }

    auto reset_soa = [&]()
    {
        for (uint32_t i = 0; i < N; ++i)
        {
            soa_set.update_sparse(
                static_cast<uint32_t>(tickets_master[i].original_idx), i);
            soa_set.get_dense_ptr()[i] =
                static_cast<uint32_t>(tickets_master[i].original_idx);
        }
    };

    std::vector<double> soa_512_times;
    reset_soa();
    auto ts1 = high_resolution_clock::now();
    Radix_Internal::SortGroup<SoftTicket, 0>(ticket_group, reg_stub);
    auto ts2 = high_resolution_clock::now();
    soa_512_times.push_back(duration<double, std::milli>(ts2 - ts1).count());

    std::vector<double> soa_256_times;
    reset_soa();
    auto ta1 = high_resolution_clock::now();
    Radix_Internal::SortGroup_AVX2<SoftTicket, 0>(ticket_group, reg_stub);
    auto ta2 = high_resolution_clock::now();
    soa_256_times.push_back(duration<double, std::milli>(ta2 - ta1).count());

    double s512_avg = get_stable_avg(soa_512_times);
    double s256_avg = get_stable_avg(soa_256_times);

    print_result("RADIX SoA (AVX2)", s256_avg, "ms");
    print_result("RADIX SoA (AVX-512)", s512_avg, "ms");

    std::cout << "------------------------------------------" << "\n";
    print_result("AVX-512 vs AVX2 Speedup", s256_avg / s512_avg, "x");
    print_result("SoA AVX-512 vs STD Sort", std_avg / s512_avg, "x");
    std::cout << "------------------------------------------" << "\n";
}

void bench_ecs_performance()
{
    print_benchmark_header(3, "ECS CORE SYSTEMS",
                           "1M Entities | Update + Sparse View + Lifecycle");

    const int N = 1000000;

    static String_lib::UniversalArena<64> data_arena(1024 * 1024 * 1024);
    Registry<64> reg(data_arena);

    auto t_create_1 = high_resolution_clock::now();
    auto group = reg.group<Position3D, Physics3D>();
    reg.mass_add_to_group(N, group);
    auto t_create_2 = high_resolution_clock::now();
    print_result("Mass Create (1M Grouped)",
                 duration<double, std::milli>(t_create_2 - t_create_1).count(),
                 "ms");

    float dt = 0.016f;
    auto t_phys_1 = high_resolution_clock::now();

    group.each_fast(
        [&](float &x, float &y, float &z,                       // Position3D
            float &vx, float &vy, float &vz, float &m, float &s // Physics3D
        )
        {
            x += vx * dt;
            y += vy * dt;
            z += vz * dt;
            vx *= 0.99f;
        });

    auto t_phys_2 = high_resolution_clock::now();
    print_result("System Update (each_fast)",
                 duration<double, std::milli>(t_phys_2 - t_phys_1).count(),
                 "ms");

    for (int i = 0; i < N; i += 100)
    {
        reg.emplace_component<SoftTicket>(i, String_lib::StringView("Asset", 5),
                                          (size_t)i);
    }

    auto t_view_1 = high_resolution_clock::now();
    auto v = reg.view<SoftTicket>();
    v.each([&](String_lib::StringView &view, size_t &idx) { idx += 1; });
    auto t_view_2 = high_resolution_clock::now();
    print_result("Sparse View (1% fill)",
                 duration<double, std::milli>(t_view_2 - t_view_1).count(),
                 "ms");

    auto t_dest_1 = high_resolution_clock::now();
    for (uint32_t i = 0; i < 100000; ++i)
    {
        reg.queue_destruction(i);
    }
    reg.flush_destroyed(group);
    auto t_dest_2 = high_resolution_clock::now();
    print_result("Flush Destruction (100k)",
                 duration<double, std::milli>(t_dest_2 - t_dest_1).count(),
                 "ms");

    std::cout << "------------------------------------------" << "\n";
}

void bench_ecs_extreme()
{
    print_benchmark_header(4, "ECS",
                           "AVX-512 Linear Access");
    const int N = 10000000;
    static String_lib::UniversalArena<64> data_arena(1024 * 1024 * 1024);
    Registry<64> reg(data_arena);
    auto group = reg.group<Position3D, Physics3D>();
    reg.mass_add_to_group(N, group);

    float *px = group.get_ptr<Position3D, 0>();
    float *py = group.get_ptr<Position3D, 1>();
    float *pz = group.get_ptr<Position3D, 2>();
    float *vx = group.get_ptr<Physics3D, 0>();
    float *vy = group.get_ptr<Physics3D, 1>();
    float *vz = group.get_ptr<Physics3D, 2>();

    float dt = 0.016f;

    auto t1 = high_resolution_clock::now();

#pragma clang loop vectorize(enable) interleave(enable)
    for (uint32_t i = 0; i < N; ++i)
    {
        px[i] += vx[i] * dt;
        py[i] += vy[i] * dt;
        pz[i] += vz[i] * dt;
    }

    auto t2 = high_resolution_clock::now();
    double extreme_ms = duration<double, std::milli>(t2 - t1).count();
    print_result("Raw Pointer Access (SIMD)",
                 duration<double, std::milli>(t2 - t1).count(), "ms");
    double entities_per_sec = (N / (extreme_ms / 1000.0)) / 1000000.0;
    std::cout << "  - Throughput: " << entities_per_sec
              << " Million Entities/sec" << "\n";
    std::cout << "------------------------------------------" << "\n";
}

void bench_ecs_extreme_c()
{
    print_benchmark_header(5, "ECS",
                           "AVX-512 Parallel + Prefetch + Raw Pointers");
    const int N = 10000000;

    static String_lib::UniversalArena<64> extreme_arena(1024 * 1024 * 512);
    Registry<64> reg(extreme_arena);

    auto group = reg.group<Position3D, Physics3D>();
    reg.mass_add_to_group(N, group);

    float *__restrict px = group.get_ptr<Position3D, 0>();
    float *__restrict py = group.get_ptr<Position3D, 1>();
    float *__restrict pz = group.get_ptr<Position3D, 2>();
    float *__restrict vx = group.get_ptr<Physics3D, 0>();
    float *__restrict vy = group.get_ptr<Physics3D, 1>();
    float *__restrict vz = group.get_ptr<Physics3D, 2>();

    const float dt = 0.016f;

    auto t1 = high_resolution_clock::now();

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
    {
        if (i + 64 < N)
        {
            _mm_prefetch((const char *)&px[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&vx[i + 64], _MM_HINT_T0);
        }

        px[i] += vx[i] * dt;
        py[i] += vy[i] * dt;
        pz[i] += vz[i] * dt;
    }

    auto t2 = high_resolution_clock::now();

    double extreme_ms = duration<double, std::milli>(t2 - t1).count();
    print_result("Parallel SIMD + Prefetch (1M)", extreme_ms, "ms");

    double entities_per_sec = (N / (extreme_ms / 1000.0)) / 1000000.0;
    std::cout << "  - Throughput: " << entities_per_sec
              << " Million Entities/sec" << "\n";
    std::cout << "------------------------------------------" << "\n";
}

int main()
{
    optimize_system_for_bench();
    bench_asset_sorting();
    bench_ecs_performance();
    bench_ecs_extreme();
    bench_ecs_extreme_c();
    std::cin.clear();
    while (std::cin.peek() != EOF && std::cin.get() != '\n')
        ;

    std::cin.get();
    std::cin.get();
    system("pause");
    return 0;

}
