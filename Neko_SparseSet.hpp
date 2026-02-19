#ifndef SPARSE_SET_HPP
#define SPARSE_SET_HPP

#include <cstdint>
#include <tuple>
#include <utility>

#include "String.hpp"
#include <algorithm>
#include <type_traits>
#include <vector>

#include <atomic>
#include <cmath>
#include <omp.h>

struct Any
{
    template <typename T> operator T() const;
};

template <typename T, typename... Args>
concept is_braces_constructible = requires { T{std::declval<Args>()...}; };

// ECS REFLECTION
template <typename T>
auto struct_to_tuple(T &s) {
    if constexpr (is_braces_constructible<T, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any>) {
        auto &[a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p] = s;
        return std::forward_as_tuple(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    } else if constexpr (is_braces_constructible<T, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any>) {
        auto &[a, b, c, d, e, f, g, h, i, j, k, l, m, n, o] = s;
        return std::forward_as_tuple(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o);
    } else if constexpr (is_braces_constructible<T, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any>) {
        auto &[a, b, c, d, e, f, g, h, i, j, k, l, m, n] = s;
        return std::forward_as_tuple(a, b, c, d, e, f, g, h, i, j, k, l, m, n);
    } else if constexpr (is_braces_constructible<T, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any>) {
        auto &[a, b, c, d, e, f, g, h, i, j, k, l, m] = s;
        return std::forward_as_tuple(a, b, c, d, e, f, g, h, i, j, k, l, m);
    } else if constexpr (is_braces_constructible<T, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any>) {
        auto &[a, b, c, d, e, f, g, h, i, j, k, l] = s;
        return std::forward_as_tuple(a, b, c, d, e, f, g, h, i, j, k, l);
    } else if constexpr (is_braces_constructible<T, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any>) {
        auto &[a, b, c, d, e, f, g, h, i, j, k] = s;
        return std::forward_as_tuple(a, b, c, d, e, f, g, h, i, j, k);
    } else if constexpr (is_braces_constructible<T, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any>) {
        auto &[a, b, c, d, e, f, g, h, i, j] = s;
        return std::forward_as_tuple(a, b, c, d, e, f, g, h, i, j);
    } else if constexpr (is_braces_constructible<T, Any, Any, Any, Any, Any, Any, Any, Any, Any>) {
        auto &[a, b, c, d, e, f, g, h, i] = s;
        return std::forward_as_tuple(a, b, c, d, e, f, g, h, i);
    } else if constexpr (is_braces_constructible<T, Any, Any, Any, Any, Any, Any, Any, Any>) {
        auto &[a, b, c, d, e, f, g, h] = s;
        return std::forward_as_tuple(a, b, c, d, e, f, g, h);
    } else if constexpr (is_braces_constructible<T, Any, Any, Any, Any, Any, Any, Any>) {
        auto &[a, b, c, d, e, f, g] = s;
        return std::forward_as_tuple(a, b, c, d, e, f, g);
    } else if constexpr (is_braces_constructible<T, Any, Any, Any, Any, Any, Any>) {
        auto &[a, b, c, d, e, f] = s;
        return std::forward_as_tuple(a, b, c, d, e, f);
    } else if constexpr (is_braces_constructible<T, Any, Any, Any, Any, Any>) {
        auto &[a, b, c, d, e] = s;
        return std::forward_as_tuple(a, b, c, d, e);
    } else if constexpr (is_braces_constructible<T, Any, Any, Any, Any>) {
        auto &[a, b, c, d] = s;
        return std::forward_as_tuple(a, b, c, d);
    } else if constexpr (is_braces_constructible<T, Any, Any, Any>) {
        auto &[a, b, c] = s;
        return std::forward_as_tuple(a, b, c);
    } else if constexpr (is_braces_constructible<T, Any, Any>) {
        auto &[a, b] = s;
        return std::forward_as_tuple(a, b);
    } else if constexpr (is_braces_constructible<T, Any>) {
        auto &[a] = s;
        return std::forward_as_tuple(a);
    }
}

// thread_local uint8_t local_bitset[262144]; //  16384 - 48kb

struct Position3D
{
    float x, y, z;
};
struct alignas(16) Physics3D
{
    float vx, vy, vz, mass, shift;
};

struct Attractor
{
    float x, y, z, force;
};

template <typename T> struct ComponentTraits;

template <> struct ComponentTraits<Position3D>
{
    using Types = std::tuple<float, float, float>; // x, y, z
};
template <> struct ComponentTraits<Physics3D>
{
    using Types = std::tuple<float, float, float, float, float>;
};
template <> struct ComponentTraits<Attractor>
{
    using Types = std::tuple<float, float, float, float>; // x, y, z, force
};

// TODO: Blocks -d
__attribute__((always_inline)) inline bool
LexisReverseCompare(const String_lib::StringView &lhs,
                    const String_lib::StringView &rhs)
{
    size_t len1 = lhs.GetLength();
    size_t len2 = rhs.GetLength();
    size_t min_len = (len1 < len2) ? len1 : len2;

    const char *p1 = lhs.c_str();
    const char *p2 = rhs.c_str();

    // Prepass
    if (LIKELY(min_len >= 8))
    {
        uint64_t u1 =
            __builtin_bswap64(*reinterpret_cast<const uint64_t *>(p1));
        uint64_t u2 =
            __builtin_bswap64(*reinterpret_cast<const uint64_t *>(p2));
        if (u1 != u2)
            return u1 > u2;

        int res = String_lib::c_strcmp_unaligned(p1 + 8, p2 + 8, min_len - 8);
        if (res != 0)
            return res > 0;
    }
    else
    {

        int res = String_lib::c_strcmp_unaligned(p1, p2, min_len);
        if (res != 0)
            return res > 0;
    }

    return len1 > len2;
}

struct SoftTicket
{
    String_lib::StringView view;
    size_t original_idx;

    bool operator>(const SoftTicket &other) const
    {
        return LexisReverseCompare(view, other.view);
    }
};

template <> struct ComponentTraits<SoftTicket>
{

    using Types = std::tuple<String_lib::StringView, size_t>;
};

template <size_t Align, typename... Components> class MultiView;

template <size_t Align, typename... Components> class Group;

template <size_t Align, typename... Fields> class SparseSetSoA
{
  private:
    uint32_t *sparse;
    uint32_t *dense;

    std::tuple<Fields *...> streams;

    uint32_t count = 0;
    uint32_t max_entity_id;
    static constexpr uint32_t NOT_FOUND = 0xFFFFFFFF;
    uint32_t capacity;

    template <size_t... Is>
    void remap_impl(const uint32_t *sorted_entities, uint32_t count_to_reorder,
                    std::index_sequence<Is...> seq)
    {
        uint32_t i = 0;
        uint32_t limit = (count_to_reorder > 8) ? (count_to_reorder - 8) : 0;

        for (; i < limit; i += 8)
        {
#define STEP(idx)                                                              \
    {                                                                          \
        uint32_t target = sorted_entities[idx];                                \
        uint32_t curr_idx = sparse[target];                                    \
        if (LIKELY(curr_idx != idx))                                           \
        {                                                                      \
            uint32_t ent_at_i = dense[idx];                                    \
            std::swap(dense[idx], dense[curr_idx]);                            \
                                                                               \
            ((std::swap(std::get<Is>(streams)[idx],                            \
                        std::get<Is>(streams)[curr_idx])),                     \
             ...);                                                             \
            sparse[target] = idx;                                              \
            sparse[ent_at_i] = curr_idx;                                       \
        }                                                                      \
    }

            STEP(i);
            STEP(i + 1);
            STEP(i + 2);
            STEP(i + 3);
            STEP(i + 4);
            STEP(i + 5);
            STEP(i + 6);
            STEP(i + 7);
#undef STEP
        }

        for (; i < count_to_reorder; ++i)
        {
            uint32_t target = sorted_entities[i];
            uint32_t curr_idx = sparse[target];
            if (curr_idx != i)
            {
                uint32_t ent_at_i = dense[i];
                std::swap(dense[i], dense[curr_idx]);
                ((std::swap(std::get<Is>(streams)[i],
                            std::get<Is>(streams)[curr_idx])),
                 ...);
                sparse[target] = i;
                sparse[ent_at_i] = curr_idx;
            }
        }
    }

  public:
    SparseSetSoA(uint32_t max_entities, uint32_t initial_capacity,
                 String_lib::UniversalArena<Align> &arena)
        : max_entity_id(max_entities), capacity(initial_capacity)
    {
        sparse = static_cast<uint32_t *>(
            arena.alloc(max_entities * sizeof(uint32_t)));
        dense = static_cast<uint32_t *>(
            arena.alloc(initial_capacity * sizeof(uint32_t)));

        alloc_streams(arena, initial_capacity,
                      std::make_index_sequence<sizeof...(Fields)>{});

        for (uint32_t i = 0; i != max_entities; ++i)
            sparse[i] = NOT_FOUND;
    }

    template <size_t... Is>
    ALWAYS_INLINE void swap_all_fields_unrolled(uint32_t a, uint32_t b,
                                                std::index_sequence<Is...>)
    {

        ((std::swap(std::get<Is>(streams)[a], std::get<Is>(streams)[b])), ...);
    }

    ALWAYS_INLINE void swap_all_fields_manual(uint32_t a, uint32_t b)
    {

        swap_all_fields_unrolled(a, b,
                                 std::make_index_sequence<sizeof...(Fields)>{});
    }

    void remap_dense(const uint32_t *sorted_entities, uint32_t count_to_reorder)
    {
        if (count_to_reorder == 0)
            return;

        remap_impl(sorted_entities, count_to_reorder,
                   std::make_index_sequence<sizeof...(Fields)>{});
    }

    ALWAYS_INLINE void add(uint32_t entity_id, Fields... args)
    {

        if (UNLIKELY(entity_id >= max_entity_id))
            return;

        if (UNLIKELY(count >= capacity))
            return;

        sparse[entity_id] = count;
        dense[count] = entity_id;

        auto tp = std::forward_as_tuple(args...);
        [&]<size_t... Is>(std::index_sequence<Is...>)
        {
            ((std::get<Is>(streams)[count] = std::get<Is>(tp)), ...);
        }(std::make_index_sequence<sizeof...(Fields)>{});

        count++;
    }

    ALWAYS_INLINE void add_empty(uint32_t entity_id)
    {
        if (UNLIKELY(entity_id >= max_entity_id || count >= capacity))
            return;
        sparse[entity_id] = count;
        dense[count] = entity_id;
        count++;
    }
    void swap_entities(uint32_t e1, uint32_t e2)
    {
        if (e1 == e2)
            return;
        uint32_t idx1 = sparse[e1];
        uint32_t idx2 = sparse[e2];
        std::swap(dense[idx1], dense[idx2]);

        [&]<size_t... Is>(std::index_sequence<Is...>)
        {
            ((std::swap(std::get<Is>(streams)[idx1],
                        std::get<Is>(streams)[idx2])),
             ...);
        }(std::make_index_sequence<sizeof...(Fields)>{});

        sparse[e1] = idx2;
        sparse[e2] = idx1;
    }
    ALWAYS_INLINE auto get(uint32_t entity_id)
    {
        uint32_t idx = sparse[entity_id];

        return [&]<size_t... Is>(std::index_sequence<Is...>)
        {
            return std::forward_as_tuple(std::get<Is>(streams)[idx]...);
        }(std::make_index_sequence<sizeof...(Fields)>{});
    }

    ALWAYS_INLINE void prefetch_sparse(uint32_t entity_id) const
    {
        if (LIKELY(entity_id < max_entity_id))
        {
            __builtin_prefetch(&sparse[entity_id], 0, 3);
        }
    }
    void remove(uint32_t entity_id)
    {
        if (!contains(entity_id))
            return;

        uint32_t idx_to_remove = sparse[entity_id];
        uint32_t last_idx = count - 1;
        uint32_t last_entity = dense[last_idx];

        [&]<size_t... Is>(std::index_sequence<Is...>)
        {
            ((std::get<Is>(streams)[idx_to_remove] =
                  std::move(std::get<Is>(streams)[last_idx])),
             ...);
        }(std::make_index_sequence<sizeof...(Fields)>{});

        dense[idx_to_remove] = last_entity;
        sparse[last_entity] = idx_to_remove;
        sparse[entity_id] = NOT_FOUND;

        count--;
    }
    template <size_t I> ALWAYS_INLINE auto *get_stream()
    {
        return std::get<I>(streams);
    }

    ALWAYS_INLINE uint32_t size() const
    {
        return count;
    }
    ALWAYS_INLINE uint32_t get_entity_at(uint32_t i) const
    {
        return dense[i];
    }
    ALWAYS_INLINE bool contains(uint32_t id) const
    {
        return id < max_entity_id && sparse[id] != NOT_FOUND;
    }

    template <size_t... Is>
    void alloc_streams(String_lib::UniversalArena<Align> &arena, uint32_t cap,
                       std::index_sequence<Is...>)
    {

        ((std::get<Is>(streams) = static_cast<
              typename std::tuple_element<Is, std::tuple<Fields *...>>::type>(
              arena.alloc(cap * sizeof(typename std::tuple_element<
                                       Is, std::tuple<Fields...>>::type)))),
         ...);
    }

    ALWAYS_INLINE uint32_t get_index(uint32_t entity_id) const
    {
        return sparse[entity_id];
    }
    ALWAYS_INLINE auto &get_streams()
    {
        return streams;
    }
    ALWAYS_INLINE uint32_t *get_dense_ptr()
    {
        return dense;
    }
    ALWAYS_INLINE void update_sparse(uint32_t eid, uint32_t dense_idx)
    {
        sparse[eid] = dense_idx;
    }
    static constexpr size_t FieldsCount = sizeof...(Fields);
};

template <typename Set, size_t Align>
void reorder_set_simd(Set &set, const uint32_t *sorted_entities,
                      const uint32_t *local_indices, uint32_t count,
                      String_lib::UniversalArena<Align> &arena)
{

    [&]<size_t... Is>(std::index_sequence<Is...>)
    {
        ((
             [&]()
             {
                 auto *current_stream = set.template get_stream<Is>();

                 using T = std::remove_pointer_t<decltype(current_stream)>;

                 T *tmp_buf = static_cast<T *>(arena.alloc(count * sizeof(T)));

                 uint32_t i = 0;

                 if constexpr (std::is_trivially_copyable_v<T> &&
                               (sizeof(T) == 4 || sizeof(T) == 8))
                 {
                     while (i + 16 <= count)
                     {
                         __m512i idx = _mm512_loadu_si512(&local_indices[i]);
                         if constexpr (sizeof(T) == 4)
                         {
                             _mm512_storeu_si512(
                                 &tmp_buf[i],
                                 _mm512_i32gather_epi32(
                                     idx, (const int *)current_stream, 4));
                         }
                         else
                         {
                             _mm512_storeu_si512(
                                 &tmp_buf[i],
                                 _mm512_i32gather_epi64(
                                     _mm512_castsi512_si256(idx),
                                     (const long long *)current_stream, 8));
                             _mm512_storeu_si512(
                                 &tmp_buf[i + 8],
                                 _mm512_i32gather_epi64(
                                     _mm512_extracti32x8_epi32(idx, 1),
                                     (const long long *)current_stream, 8));
                         }
                         i += 16;
                     }
                 }

                 while (i < count)
                 {

                     tmp_buf[i] = std::move(current_stream[local_indices[i]]);
                     i++;
                 }

                 if constexpr (std::is_trivially_copyable_v<T>)
                 {
                     memcpy(current_stream, tmp_buf, count * sizeof(T));
                 }
                 else
                 {
                     for (uint32_t j = 0; j < count; ++j)
                     {
                         current_stream[j] = std::move(tmp_buf[j]);
                         tmp_buf[j].~T();
                     }
                 }
             }()),
         ...);
    }(std::make_index_sequence<Set::FieldsCount>{}); // FieldsCount =
                                                     // sizeof...(Fields)

    memcpy(set.get_dense_ptr(), sorted_entities, count * sizeof(uint32_t));

    uint32_t s = 0;
    while (s < count)
    {
        set.update_sparse(sorted_entities[s], s);
        ++s;
    }
}
template <typename Set, size_t Align>
void reorder_set_avx2(Set &set, const uint32_t *sorted_entities,
                      const uint32_t *local_indices, uint32_t count,
                      String_lib::UniversalArena<Align> &arena)
{

    [&]<size_t... Is>(std::index_sequence<Is...>)
    {
        ((
             [&]()
             {
                 auto *current_stream = set.template get_stream<Is>();
                 using T = std::remove_pointer_t<decltype(current_stream)>;

                 T *tmp_buf = static_cast<T *>(arena.alloc(count * sizeof(T)));
                 uint32_t i = 0;

                 if constexpr (std::is_trivially_copyable_v<T> &&
                               (sizeof(T) == 4 || sizeof(T) == 8))
                 {
                     while (i + 8 <= count)
                     {
                         __m256i idx = _mm256_loadu_si256(
                             (const __m256i *)&local_indices[i]);

                         if constexpr (sizeof(T) == 4)
                         {

                             __m256i gathered = _mm256_i32gather_epi32(
                                 (const int *)current_stream, idx, 4);
                             _mm256_storeu_si256((__m256i *)&tmp_buf[i],
                                                 gathered);
                         }
                         else
                         {

                             __m256i g1 = _mm256_i32gather_epi64(
                                 (const long long *)current_stream,
                                 _mm256_extracti128_si256(idx, 0), 8);
                             __m256i g2 = _mm256_i32gather_epi64(
                                 (const long long *)current_stream,
                                 _mm256_extracti128_si256(idx, 1), 8);
                             _mm256_storeu_si256((__m256i *)&tmp_buf[i], g1);
                             _mm256_storeu_si256((__m256i *)&tmp_buf[i + 4],
                                                 g2);
                         }
                         i += 8;
                     }
                 }

                 while (i < count)
                 {
                     tmp_buf[i] = std::move(current_stream[local_indices[i]]);
                     i++;
                 }

                 if constexpr (std::is_trivially_copyable_v<T>)
                 {
                     memcpy(current_stream, tmp_buf, count * sizeof(T));
                 }
                 else
                 {
                     for (uint32_t j = 0; j < count; ++j)
                     {
                         current_stream[j] = std::move(tmp_buf[j]);
                         tmp_buf[j].~T();
                     }
                 }
             }()),
         ...);
    }(std::make_index_sequence<Set::FieldsCount>{});

    // Update Dense and Sparse arrays
    memcpy(set.get_dense_ptr(), sorted_entities, count * sizeof(uint32_t));
    for (uint32_t s = 0; s < count; ++s)
    {
        set.update_sparse(sorted_entities[s], s);
    }
}

struct ComponentTypeCounter
{
    static inline uint32_t next_id = 0;
};

template <typename T> struct ComponentID
{
    static inline const uint32_t value = ComponentTypeCounter::next_id++;
};

template <size_t Align> class Registry
{
  private:
    String_lib::UniversalArena<Align> &arena;
    uint32_t *destruction_queue;

    static constexpr uint32_t M_D = 131072;

    alignas(64) void *sets[128];

    alignas(64) uint32_t entity_counter = 0;

    alignas(64) std::atomic<uint32_t> destruction_count{0};

  public:
    explicit Registry(String_lib::UniversalArena<Align> &a) : arena(a)
    {
        for (int i = 0; i < 128; ++i)
            sets[i] = nullptr;
        destruction_queue =
            static_cast<uint32_t *>(arena.alloc(M_D * sizeof(uint32_t)));
    }

    template <typename... Comps>
    void reorder_to_grid(const uint32_t *sorted_entities, uint32_t count)
    {

        auto sets_tuple = std::tie(get_set<Comps>()...);

        std::apply([&](auto &...s)
                   { (s.remap_dense(sorted_entities, count), ...); },
                   sets_tuple);
    }

    template <typename T> auto &get_set(uint32_t reserve_count = 1000000)
    {
        using T_Traits = typename ComponentTraits<T>::Types;

        return get_set_soa_impl<T>(T_Traits{}, reserve_count);
    }

  private:
    template <typename T, typename... Fields>
    auto &get_set_soa_impl(std::tuple<Fields...>, uint32_t count)
    {
        uint32_t id = ComponentID<T>::value;
        if (UNLIKELY(!sets[id]))
        {
            using SetType = SparseSetSoA<Align, Fields...>;
            void *mem = arena.alloc(sizeof(SetType));
            sets[id] = new (mem) SetType(count, count, arena);
        }
        return *static_cast<SparseSetSoA<Align, Fields...> *>(sets[id]);
    }

  public:
    uint32_t create_entity()
    {
        return entity_counter++;
    }

    template <typename T> void add_component(uint32_t entity, T &&c)
    {
        auto &s = get_set<T>();
        auto tp = struct_to_tuple(std::forward<T>(c));
        // s.add(entity, tp);
        std::apply([&](auto &&...args)
                   { s.add(entity, std::forward<decltype(args)>(args)...); },
                   tp);
    }

    template <typename T, typename... GroupComps>
    void add_component(uint32_t entity, T &&c, Group<Align, GroupComps...> &g)
    {
        add_component<T>(entity, std::move(c));
        g.sync(entity);
    }

    template <typename T, typename... Args>
    void emplace_component(uint32_t entity, Args &&...args)
    {
        auto &s = get_set<T>();

        s.add(entity, std::forward<Args>(args)...);
    }

    template <typename... Comps>
    void mass_add_to_group(uint32_t count, Group<Align, Comps...> &g)
    {

        auto sets_tuple = std::tie(get_set<Comps>()...);

        for (uint32_t i = 0; i < count; ++i)
        {
            uint32_t entity_id = create_entity();

            std::apply([&](auto &...s) { (s.add_empty(entity_id), ...); },
                       sets_tuple);
        }

        g.set_explicit_size(count);
    }

    template <typename T, typename... Args>
    ALWAYS_INLINE void set_component_data(uint32_t entity, Args &&...args)
    {
        auto &s = get_set<T>();

        auto tp = std::forward_as_tuple(args...);
        uint32_t idx = s.get_index(entity);

        [&]<size_t... Is>(std::index_sequence<Is...>)
        {
            ((std::get<Is>(s.get_streams())[idx] = std::get<Is>(tp)), ...);
        }(std::make_index_sequence<sizeof...(Args)>{});
    }

    ALWAYS_INLINE uint32_t get_entity_count() const
    {
        return entity_counter;
    }

    template <typename T, typename... GroupComps>
    void remove_component(uint32_t entity, Group<Align, GroupComps...> &g)
    {
        g.desync(entity);
        get_set<T>().remove(entity);
    }

    template <typename... Components> ALWAYS_INLINE auto view()
    {
        return MultiView<Align, Components...>(*this);
    }

    template <typename... Components> auto group()
    {
        return Group<Align, Components...>(*this);
    }

    void queue_destruction(uint32_t entity_id)
    {
        uint32_t idx =
            destruction_count.fetch_add(1, std::memory_order_relaxed);
        if (LIKELY(idx < M_D))
        {
            destruction_queue[idx] = entity_id;
        }
    }
    template <typename... Comps> void flush_destroyed(Group<Align, Comps...> &g)
    {
        uint32_t total = destruction_count.load(std::memory_order_acquire);
        if (total == 0)
            return;

        uint32_t actual_count = (total > M_D) ? M_D : total;

        for (uint32_t i = 0; i < actual_count; ++i)
        {
            uint32_t id = destruction_queue[i];

            using FirstComp =
                typename std::tuple_element<0, std::tuple<Comps...>>::type;
            if (this->get_set<FirstComp>().contains(id))
            {

                (this->remove_component<Comps>(id, g), ...);
            }
        }

        destruction_count.store(0, std::memory_order_release);
    }
};

template <size_t Align, typename... Components> class MultiView
{
    std::tuple<decltype(std::declval<Registry<Align>>()
                            .template get_set<Components>())...>
        sets;

  public:
    MultiView(Registry<Align> &reg)
        : sets(reg.template get_set<Components>()...)
    {
    }

    template <typename Func> ALWAYS_INLINE void each(Func &&func)
    {
        uint32_t min_size = 0xFFFFFFFF;
        uint32_t leader_idx = 0;
        uint32_t current_idx = 0;

        std::apply(
            [&](auto &...s)
            {
                (((s.size() < min_size
                       ? (min_size = s.size(), leader_idx = current_idx)
                       : 0),
                  ++current_idx),
                 ...);
            },
            sets);

        if (min_size == 0)
            return;

        iterate(func, std::make_index_sequence<sizeof...(Components)>{},
                leader_idx, min_size);
    }

  private:
    template <typename Func, size_t... Is>
    ALWAYS_INLINE void iterate(Func &&func, std::index_sequence<Is...>,
                               uint32_t leader_idx, uint32_t sz)
    {
        for (uint32_t i = 0; i != sz; ++i)
        {
            if (LIKELY(i + 4 < sz))
            {
                uint32_t next_id = 0;
                uint32_t k_pre = 0;
                ((k_pre++ == leader_idx
                      ? (next_id = std::get<Is>(sets).get_entity_at(i + 4))
                      : 0),
                 ...);
                (std::get<Is>(sets).prefetch_sparse(next_id), ...);
            }

            uint32_t entity = 0;
            uint32_t k = 0;
            ((k++ == leader_idx ? (entity = std::get<Is>(sets).get_entity_at(i))
                                : 0),
             ...);

            if ((std::get<Is>(sets).contains(entity) && ...))
            {

                std::apply(func,
                           std::tuple_cat(std::get<Is>(sets).get(entity)...));
            }
        }
    }
};

template <size_t Align, typename... Components> class Group
{
    Registry<Align> &registry;
    uint32_t group_size = 0;

  public:
    Group(Registry<Align> &reg) : registry(reg)
    {
    }

    template <typename T, size_t FieldIdx> ALWAYS_INLINE auto *get_stream()
    {

        using Fields = typename ComponentTraits<T>::Types;
        return get_stream_impl<T, FieldIdx>(Fields{});
    }

  private:
    template <typename T, size_t FieldIdx, typename... Fields>
    ALWAYS_INLINE auto *get_stream_impl(std::tuple<Fields...>)
    {
        return registry.template get_set<T>().template get_stream<FieldIdx>();
    }

  public:
    ALWAYS_INLINE uint32_t size() const
    {
        return group_size;
    }

    void set_explicit_size(uint32_t size)
    {
        group_size = size;
    }

    template <typename T, size_t FieldIdx> ALWAYS_INLINE auto *get_ptr()
    {

        return this->template get_stream<T, FieldIdx>();
    }
    ALWAYS_INLINE const uint32_t *get_entity_ids() const
    {

        using LeaderComponent =
            typename std::tuple_element<0, std::tuple<Components...>>::type;
        return registry.template get_set<LeaderComponent>().get_dense_ptr();
    }

    void sync(uint32_t entity)
    {

        if (LIKELY((registry.template get_set<Components>().contains(entity) &&
                    ...)))
        {

            (registry.template get_set<Components>().swap_entities(
                 entity, registry.template get_set<Components>().get_entity_at(
                             group_size)),
             ...);
            group_size++;
        }
    }

    void desync(uint32_t entity)
    {
        if (LIKELY((registry.template get_set<Components>().contains(entity) &&
                    ...)))
        {

            uint32_t last_in_group =
                registry
                    .template get_set<typename std::tuple_element<
                        0, std::tuple<Components...>>::type>()
                    .get_entity_at(group_size - 1);

            (registry.template get_set<Components>().swap_entities(
                 entity, last_in_group),
             ...);

            group_size--;
        }
    }

    template <typename Func> ALWAYS_INLINE void each(Func &&func)
    {
        auto &leader = registry.template get_set<
            typename std::tuple_element<0, std::tuple<Components...>>::type>();

        for (uint32_t i = 0; i != group_size; ++i)
        {
            uint32_t entity = leader.get_entity_at(i);

            auto call_lambda = [&](auto &&...args)
            {
                std::apply(
                    func,
                    std::tuple_cat(registry.template get_set<Components>().get(
                        entity)...));
            };
            call_lambda();
        }
    }

    template <typename Func> ALWAYS_INLINE void each_fast(Func &&func)
    {
        if (group_size == 0) [[unlikely]]
            return;

        auto flat_streams = std::tuple_cat(
            registry.template get_set<Components>().get_streams()...);

        using FT = decltype(flat_streams);
        static constexpr size_t TF = std::tuple_size_v<FT>;

        [&]<size_t... Is>(std::index_sequence<Is...>)
        {
            for (uint32_t i = 0; i < group_size; ++i)
            {
                if (LIKELY(i + 32 < group_size))
                {

                    (_mm_prefetch(
                         (const char *)&std::get<Is>(flat_streams)[i + 32],
                         _MM_HINT_T0),
                     ...);
                }
                func(std::get<Is>(flat_streams)[i]...);
            }
        }(std::make_index_sequence<TF>{});
    }
};


#endif
