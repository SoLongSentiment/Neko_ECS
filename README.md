# Neko_ECS 

A data-oriented Entity Component System implementation for C++20, focusing on **Structure of Arrays (SoA)** memory layout and **SIMD**-accelerated operations.

requires **AVX2** or **AVX-512**

## Key Technical Features
*   **SoA Memory Layout**: Component data is stored in contiguous streams.
*   **SIMD Radix Sort**: Custom MSD Radix sort with **AVX-2** and **AVX-512** paths. Includes "Common Prefix Skip" optimization for string-based keys.
*   **Zero-Allocation Pipeline**: Uses thread-local `UniversalArena` allocators.
*   **Type Reflection**: Automatic struct-to-tuple mapping using C++20 structured bindings.

## Performance Benchmarks
*Tested on 1,000,000 entities (Position + Physics components).*

| Operation | Implementation | Latency / Throughput |
| :--- | :--- | :--- |
| **System Update** | AVX-512 Parallel + Prefetch | **~24.6 ms** |
| **System Throughput** | Linear Stream Access | **405.7M entities/sec** |
| **Asset Sorting (2M)** | `std::sort` (Strings) | ~718.7 ms |
| **Asset Sorting (2M)** | **Radix SoA (AVX-512)** | **~183.8 ms** |
| **Sparse View (1%)** | Sparse Set Iterator | **0.04 ms** |

tested on Intel Core i5-11400H
