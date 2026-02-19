#include "Neko_SparseSet.hpp"

namespace Radix_Internal
{

struct alignas(16) SortKey
{
    uint64_t head;
    uint32_t entity;
};

template <typename T> struct is_pair : std::false_type
{
};
template <typename T1, typename T2>
struct is_pair<std::pair<T1, T2>> : std::true_type
{
};

ALWAYS_INLINE uint32_t get_length(const String_lib::StringView &v)
{
    return static_cast<uint32_t>(v.GetLength());
}

ALWAYS_INLINE const char *get_c_str(const String_lib::StringView &v)
{
    return v.c_str();
}

template <typename V1, typename V2>
ALWAYS_INLINE uint32_t get_length(const std::pair<V1, V2> &p)
{
    return get_length(p.first);
}

template <typename V1, typename V2>
ALWAYS_INLINE const char *get_c_str(const std::pair<V1, V2> &p)
{
    return get_c_str(p.first);
}

ALWAYS_INLINE uint32_t get_length(const SoftTicket &t)
{
    return static_cast<uint32_t>(t.view.GetLength());
}

ALWAYS_INLINE const char *get_c_str(const SoftTicket &t)
{
    return t.view.c_str();
}

template <typename T>
ALWAYS_INLINE uint8_t get_byte(const T &ticket, size_t offset)
{
    if (offset >= get_length(ticket))
        return 0;

    if constexpr (std::is_same_v<T, String_lib::StringView>)
    {
        return static_cast<uint8_t>(ticket.c_str()[offset]);
    }
    else if constexpr (is_pair<T>::value)
    {
        return static_cast<uint8_t>(ticket.first.c_str()[offset]);
    }
    else
    {

        return static_cast<uint8_t>(ticket.view.c_str()[offset]);
    }
}

ALWAYS_INLINE const String_lib::StringView &
get_view(const String_lib::StringView &v)
{
    return v;
}

template <typename K, typename V>
ALWAYS_INLINE const String_lib::StringView &get_view(const std::pair<K, V> &p)
{
    return p.first;
}

template <typename T>
ALWAYS_INLINE const String_lib::StringView &get_view(const T &t)
{
    return t.view;
}

template <typename T>
void RadSortIdxInternal(uint32_t *idx_begin, uint32_t *idx_end, T *base_data,
                        uint8_t *bytes_pool, size_t offset)
{
restart:
    const size_t count = idx_end - idx_begin;
    if (count < 64)
    {
        for (uint32_t *i = idx_begin + 1; i < idx_end; ++i)
        {
            uint32_t val = *i;
            uint32_t *j = i;
            while (j > idx_begin &&
                   LexisReverseCompare(get_view(base_data[val]),
                                       get_view(base_data[*(j - 1)])))
            {
                *j = *(j - 1);
                --j;
            }
            *j = val;
        }
        return;
    }

    uint8_t *bytes = bytes_pool;

    uint32_t c0[256] = {0}, c1[256] = {0}, c2[256] = {0}, c3[256] = {0};

    size_t i = 0;
    for (; i + 3 < count; i += 4)
    {
        uint8_t b0 = get_byte(base_data[idx_begin[i]], offset);
        uint8_t b1 = get_byte(base_data[idx_begin[i + 1]], offset);
        uint8_t b2 = get_byte(base_data[idx_begin[i + 2]], offset);
        uint8_t b3 = get_byte(base_data[idx_begin[i + 3]], offset);
        bytes[i] = b0;
        c0[b0]++;
        bytes[i + 1] = b1;
        c1[b1]++;
        bytes[i + 2] = b2;
        c2[b2]++;
        bytes[i + 3] = b3;
        c3[b3]++;
    }
    for (; i < count; ++i)
    {
        uint8_t b = get_byte(base_data[idx_begin[i]], offset);
        bytes[i] = b;
        c0[b]++;
    }

    uint32_t counts[256];
    for (int j = 0; j < 256; ++j)
        counts[j] = c0[j] + c1[j] + c2[j] + c3[j];

    for (int j = 0; j < 256; ++j)
    {
        if (counts[j] == count)
        {
            if (offset < 63 && j != 0)
            {
                offset++;
                goto restart;
            }
            return;
        }
    }

    uint32_t offsets[256], pos = 0;
    for (int j = 255; j >= 0; --j)
    {
        offsets[j] = pos;
        pos += counts[j];
    }
    uint32_t active_offsets[256];
    std::copy(std::begin(offsets), std::end(offsets),
              std::begin(active_offsets));

    for (int j = 255; j >= 0; --j)
    {
        if (counts[j] == 0)
            continue;
        uint32_t limit = offsets[j] + counts[j];
        while (active_offsets[j] < limit)
        {
            uint32_t curr_idx = active_offsets[j];
            uint32_t current_val = idx_begin[curr_idx];
            uint8_t b = bytes[curr_idx];
            while (b != j)
            {
                uint32_t dest_idx = active_offsets[b]++;
                std::swap(current_val, idx_begin[dest_idx]);
                std::swap(b, bytes[dest_idx]);
            }
            idx_begin[active_offsets[j]++] = current_val;
        }
    }

    if (offset < 64)
    {
        uint32_t current_pos = 0;
        for (int j = 255; j >= 1; --j)
        {
            if (counts[j] > 1)
            {

                RadSortIdxInternal(idx_begin + current_pos,
                                   idx_begin + current_pos + counts[j],
                                   base_data, bytes_pool + current_pos,
                                   offset + 1);
            }
            current_pos += counts[j];
        }
    }
}

static thread_local String_lib::UniversalArena<8> RadixTempArena(1024 * 1024 *
                                                                 256);
static thread_local String_lib::UniversalArena<16>
    RadixGroupTempArena(1024 * 1024 * 256);

template <typename T_Comp, size_t FieldIdx, size_t Align,
          typename... GroupComps>
void RadixMSDStep(uint32_t *idx_begin, uint32_t *idx_end,
                  Group<Align, GroupComps...> &group, size_t offset,
                  uint8_t *chars_buffer)
{
    const size_t count = idx_end - idx_begin;
    if (count < 32)
    {
        auto *stream = group.template get_stream<T_Comp, FieldIdx>();
        for (uint32_t *i = idx_begin + 1; i < idx_end; ++i)
        {
            uint32_t val = *i;
            uint32_t *j = i;
            while (j > idx_begin &&
                   LexisReverseCompare(stream[val], stream[*(j - 1)]))
            {
                *j = *(j - 1);
                --j;
            }
            *j = val;
        }
        return;
    }

    auto *stream = group.template get_stream<T_Comp, FieldIdx>();
    uint32_t counts[256] = {0};

    for (size_t i = 0; i < count; ++i)
    {
        const auto &sv = stream[idx_begin[i]];
        uint8_t b = (offset < sv.GetLength()) ? (uint8_t)sv.c_str()[offset] : 0;
        counts[b]++;
    }

    uint32_t offsets[256], active_offsets[256];
    uint32_t pos = 0;
    for (int j = 255; j >= 0; --j)
    {
        offsets[j] = pos;
        active_offsets[j] = pos;
        pos += counts[j];
    }

    for (int j = 255; j >= 0; --j)
    {
        while (active_offsets[j] < offsets[j] + counts[j])
        {
            uint32_t cur_idx = idx_begin[active_offsets[j]];
            const auto &sv = stream[cur_idx];
            uint8_t b =
                (offset < sv.GetLength()) ? (uint8_t)sv.c_str()[offset] : 0;

            if (b == j)
            {
                active_offsets[j]++;
            }
            else
            {
                std::swap(idx_begin[active_offsets[j]],
                          idx_begin[active_offsets[b]++]);
            }
        }
    }

    uint32_t current_pos = 0;
    for (int j = 255; j >= 1; --j)
    {
        if (counts[j] > 1 && offset < 128)
        {
            RadixMSDStep<T_Comp, FieldIdx>(idx_begin + current_pos,
                                           idx_begin + current_pos + counts[j],
                                           group, offset + 1, chars_buffer);
        }
        current_pos += counts[j];
    }
}

template <typename T_Comp, size_t FieldIdx, size_t Align,
          typename... GroupComps>
void SortGroup_AVX512(Group<Align, GroupComps...> &group, Registry<Align> &reg)
{
    const uint32_t count = group.size();
    if (UNLIKELY(count < 2))
        return;
    const size_t start_offset = RadixGroupTempArena.current_offset();

    auto *string_stream = group.template get_stream<T_Comp, FieldIdx>();
    const uint32_t *entities = group.get_entity_ids();

    uint32_t common_skip = 0;
    uint32_t min_len = 0xFFFFFFFF;

    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t l = string_stream[i].GetLength();
        if (l < min_len)
            min_len = l;
    }

    bool difference_found = false;
    while (common_skip + 32 <= min_len)
    {
        __m256i first_chunk = _mm256_loadu_si256(
            (const __m256i *)(string_stream[0].c_str() + common_skip));
        uint32_t total_diff_mask = 0;

        for (uint32_t i = 1; i < count; ++i)
        {
            __m256i current_chunk = _mm256_loadu_si256(
                (const __m256i *)(string_stream[i].c_str() + common_skip));

            __m256i cmp = _mm256_cmpeq_epi8(first_chunk, current_chunk);

            uint32_t match_mask = (uint32_t)_mm256_movemask_epi8(cmp);
            if (match_mask != 0xFFFFFFFF)
            {

                total_diff_mask = ~match_mask;
                common_skip += __builtin_ctz(total_diff_mask);
                difference_found = true;
                break;
            }
        }
        if (difference_found)
            break;
        common_skip += 32;
    }

    if (!difference_found && common_skip < min_len)
    {
        const char *first_ptr = string_stream[0].c_str();
        while (common_skip < min_len)
        {
            char c = first_ptr[common_skip];
            bool match = true;
            for (uint32_t i = 1; i < count; ++i)
            {
                if (string_stream[i].c_str()[common_skip] != c)
                {
                    match = false;
                    break;
                }
            }
            if (!match)
                break;
            common_skip++;
        }
    }

    struct Key
    {
        uint64_t head;
        uint32_t ent;
    };
    Key *src = (Key *)RadixGroupTempArena.alloc(count * sizeof(Key));
    Key *dst = (Key *)RadixGroupTempArena.alloc(count * sizeof(Key));

    for (uint32_t i = 0; i < count; ++i)
    {
        const auto &sv = string_stream[i];
        const char *p = sv.c_str() + common_skip;
        uint32_t rem = sv.GetLength() - common_skip;

        uint32_t load_cnt = (rem < 8) ? rem : 8;
        __mmask8 load_mask = (1U << load_cnt) - 1;

        uint64_t val = __builtin_bswap64(
            _mm_cvtsi128_si64(_mm_maskz_loadu_epi8(load_mask, p)));

        src[i] = {~val, i};
    }

    for (int shift = 0; shift < 64; shift += 11)
    {
        uint32_t counts[2048] = {0};
        uint32_t offsets[2048];
        uint32_t mask = (shift > 50) ? 511 : 2047;

        for (uint32_t i = 0; i < count; ++i)
            counts[(src[i].head >> shift) & mask]++;

        uint32_t pos = 0;
        for (int j = 0; j <= (int)mask; ++j)
        {
            offsets[j] = pos;
            pos += counts[j];
        }

        struct alignas(64) BucketBuffer
        {
            Key data[16];
            uint32_t count;
        };

        void *raw_mem =
            RadixGroupTempArena.alloc((mask + 1) * sizeof(BucketBuffer) + 64);
        BucketBuffer *buffers =
            (BucketBuffer *)(((uintptr_t)raw_mem + 63) & ~63);
        for (int j = 0; j <= (int)mask; ++j)
            buffers[j].count = 0;

        for (uint32_t i = 0; i < count; ++i)
        {
            uint32_t bucket = (src[i].head >> shift) & mask;
            BucketBuffer &b = buffers[bucket];
            b.data[b.count++] = src[i];

            if (UNLIKELY(b.count == 16))
            {
                memcpy(&dst[offsets[bucket]], b.data, 16 * sizeof(Key));
                offsets[bucket] += 16;
                b.count = 0;
            }
        }

        for (int j = 0; j <= (int)mask; ++j)
        {
            if (buffers[j].count > 0)
            {
                memcpy(&dst[offsets[j]], buffers[j].data,
                       buffers[j].count * sizeof(Key));
                offsets[j] += buffers[j].count;
            }
        }
        std::swap(src, dst);
    }

    for (uint32_t i = 0; i < count;)
    {
        uint32_t j = i + 1;

        while (j < count && src[j].head == src[i].head)
            j++;

        uint32_t n = j - i;
        if (n > 1)
        {

            if (n <= 32)
            {
                for (uint32_t k = i + 1; k < j; ++k)
                {
                    Key pivot = src[k];
                    uint32_t m = k;

                    while (m > i &&
                           LexisReverseCompare(string_stream[pivot.ent],
                                               string_stream[src[m - 1].ent]))
                    {
                        src[m] = src[m - 1];
                        --m;
                    }
                    src[m] = pivot;
                }
            }
            else
            {

                std::sort(src + i, src + j,
                          [&](const Key &a, const Key &b)
                          {
                              return LexisReverseCompare(string_stream[a.ent],
                                                         string_stream[b.ent]);
                          });
            }
        }
        i = j;
    }

    uint32_t *final_map =
        (uint32_t *)RadixGroupTempArena.alloc(count * sizeof(uint32_t));

    uint32_t *local_indices =
        (uint32_t *)RadixGroupTempArena.alloc(count * sizeof(uint32_t));

    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t orig_idx = src[i].ent;
        final_map[i] = entities[orig_idx];
        local_indices[i] = orig_idx;
    }

    (
        [&]()
        {
            auto &set = reg.template get_set<GroupComps>();
            reorder_set_simd(set, final_map, local_indices, count,
                             RadixGroupTempArena);
        }(),
        ...);

    RadixGroupTempArena.set_offset(start_offset);
}

template <typename T_Comp, size_t FieldIdx, size_t Align,
          typename... GroupComps>
void SortGroup_AVX2(Group<Align, GroupComps...> &group, Registry<Align> &reg)
{
    const uint32_t count = group.size();
    if (UNLIKELY(count < 2))
        return;
    const size_t start_offset = RadixGroupTempArena.current_offset();

    auto *string_stream = group.template get_stream<T_Comp, FieldIdx>();
    const uint32_t *entities = group.get_entity_ids();

    uint32_t common_skip = 0;
    uint32_t min_len = 0xFFFFFFFF;

    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t l = string_stream[i].GetLength();
        if (l < min_len)
            min_len = l;
    }

    bool difference_found = false;
    while (common_skip + 32 <= min_len)
    {
        __m256i first_chunk = _mm256_loadu_si256(
            (const __m256i *)(string_stream[0].c_str() + common_skip));
        uint32_t total_diff_mask = 0;

        for (uint32_t i = 1; i < count; ++i)
        {
            __m256i current_chunk = _mm256_loadu_si256(
                (const __m256i *)(string_stream[i].c_str() + common_skip));

            __m256i cmp = _mm256_cmpeq_epi8(first_chunk, current_chunk);

            uint32_t match_mask = (uint32_t)_mm256_movemask_epi8(cmp);
            if (match_mask != 0xFFFFFFFF)
            {

                total_diff_mask = ~match_mask;
                common_skip += __builtin_ctz(total_diff_mask);
                difference_found = true;
                break;
            }
        }
        if (difference_found)
            break;
        common_skip += 32;
    }

    if (!difference_found && common_skip < min_len)
    {
        const char *first_ptr = string_stream[0].c_str();
        while (common_skip < min_len)
        {
            char c = first_ptr[common_skip];
            bool match = true;
            for (uint32_t i = 1; i < count; ++i)
            {
                if (string_stream[i].c_str()[common_skip] != c)
                {
                    match = false;
                    break;
                }
            }
            if (!match)
                break;
            common_skip++;
        }
    }

    struct Key
    {
        uint64_t head;
        uint32_t ent;
    };
    Key *src = (Key *)RadixGroupTempArena.alloc(count * sizeof(Key));
    Key *dst = (Key *)RadixGroupTempArena.alloc(count * sizeof(Key));

    for (uint32_t i = 0; i < count; ++i)
    {
        const auto &sv = string_stream[i];
        const char *p = sv.c_str() + common_skip;
        uint32_t rem = sv.GetLength() - common_skip;

        uint64_t val = 0;
        if (rem >= 8)
        {

            __builtin_memcpy(&val, p, 8);
        }
        else
        {

            for (uint32_t k = 0; k < rem; ++k)
            {
                ((char *)&val)[k] = p[k];
            }
        }

        val = __builtin_bswap64(val);
        src[i] = {~val, i};
    }

    for (int shift = 0; shift < 64; shift += 11)
    {
        uint32_t counts[2048] = {0};
        uint32_t offsets[2048];
        uint32_t mask = (shift > 50) ? 511 : 2047;

        for (uint32_t i = 0; i < count; ++i)
            counts[(src[i].head >> shift) & mask]++;

        uint32_t pos = 0;
        for (int j = 0; j <= (int)mask; ++j)
        {
            offsets[j] = pos;
            pos += counts[j];
        }

        struct alignas(64) BucketBuffer
        {
            Key data[16];
            uint32_t count;
        };

        void *raw_mem =
            RadixGroupTempArena.alloc((mask + 1) * sizeof(BucketBuffer) + 64);
        BucketBuffer *buffers =
            (BucketBuffer *)(((uintptr_t)raw_mem + 63) & ~63);
        for (int j = 0; j <= (int)mask; ++j)
            buffers[j].count = 0;

        for (uint32_t i = 0; i < count; ++i)
        {
            uint32_t bucket = (src[i].head >> shift) & mask;
            BucketBuffer &b = buffers[bucket];
            b.data[b.count++] = src[i];

            if (UNLIKELY(b.count == 16))
            {
                memcpy(&dst[offsets[bucket]], b.data, 16 * sizeof(Key));
                offsets[bucket] += 16;
                b.count = 0;
            }
        }

        for (int j = 0; j <= (int)mask; ++j)
        {
            if (buffers[j].count > 0)
            {
                memcpy(&dst[offsets[j]], buffers[j].data,
                       buffers[j].count * sizeof(Key));
                offsets[j] += buffers[j].count;
            }
        }
        std::swap(src, dst);
    }

    for (uint32_t i = 0; i < count;)
    {
        uint32_t j = i + 1;

        while (j < count && src[j].head == src[i].head)
            j++;

        uint32_t n = j - i;
        if (n > 1)
        {

            if (n <= 32)
            {
                for (uint32_t k = i + 1; k < j; ++k)
                {
                    Key pivot = src[k];
                    uint32_t m = k;

                    while (m > i &&
                           LexisReverseCompare(string_stream[pivot.ent],
                                               string_stream[src[m - 1].ent]))
                    {
                        src[m] = src[m - 1];
                        --m;
                    }
                    src[m] = pivot;
                }
            }
            else
            {

                std::sort(src + i, src + j,
                          [&](const Key &a, const Key &b)
                          {
                              return LexisReverseCompare(string_stream[a.ent],
                                                         string_stream[b.ent]);
                          });
            }
        }
        i = j;
    }

    uint32_t *final_map =
        (uint32_t *)RadixGroupTempArena.alloc(count * sizeof(uint32_t));

    uint32_t *local_indices =
        (uint32_t *)RadixGroupTempArena.alloc(count * sizeof(uint32_t));

    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t orig_idx = src[i].ent;
        final_map[i] = entities[orig_idx];
        local_indices[i] = orig_idx;
    }

    (
        [&]()
        {
            auto &set = reg.template get_set<GroupComps>();
            reorder_set_avx2(set, final_map, local_indices, count,
                             RadixGroupTempArena);
        }(),
        ...);

    RadixGroupTempArena.set_offset(start_offset);
}

template <typename T_Comp, size_t FieldIdx, size_t Align,
          typename... GroupComps>
void SortGroup(Group<Align, GroupComps...> &group, Registry<Align> &reg)
{

    static bool has512 = String_lib::CPU_Features::has_512();

    if (has512)
    {

        SortGroup_AVX512<T_Comp, FieldIdx, Align, GroupComps...>(group, reg);
        return;
    }

    SortGroup_AVX2<T_Comp, FieldIdx, Align, GroupComps...>(group, reg);
}

/*template <typename T_Comp, size_t FieldIdx, size_t Align, typename... Fields>
    void SortGroup(SparseSetSoA<Align, Fields...>& set, Registry<Align>& reg) {
    const uint32_t count = set.size();
    if (UNLIKELY(count < 2)) return;
    const size_t start_offset = RadixGroupTempArena.current_offset();

    auto* string_stream = set.template get_stream<FieldIdx>();
    const uint32_t* entities = set.get_dense_ptr();


    uint32_t common_skip = 0;
    uint32_t min_len = 0xFFFFFFFF;


    for (uint32_t i = 0; i < count; ++i) {
        uint32_t l = string_stream[i].GetLength();
        if (l < min_len) min_len = l;
    }


    bool difference_found = false;
    while (common_skip + 64 <= min_len) {
        __m512i first_chunk = _mm512_loadu_si512(string_stream[0].c_str() +
common_skip); uint64_t total_diff_mask = 0;

        for (uint32_t i = 1; i < count; ++i) {
            __m512i current_chunk = _mm512_loadu_si512(string_stream[i].c_str()
+ common_skip); total_diff_mask |= _mm512_cmpneq_epi8_mask(first_chunk,
current_chunk); if (total_diff_mask) break;
        }

        if (total_diff_mask) {
            common_skip += __builtin_ctzll(total_diff_mask);
            difference_found = true;
            break;
        }
        common_skip += 64;
    }


    if (!difference_found && common_skip < min_len) {
        const char* first_ptr = string_stream[0].c_str();
        while (common_skip < min_len) {
            char c = first_ptr[common_skip];
            bool match = true;
            for (uint32_t i = 1; i < count; ++i) {
                if (string_stream[i].c_str()[common_skip] != c) {
                    match = false;
                    break;
                }
            }
            if (!match) break;
            common_skip++;
        }
    }

    struct Key { uint64_t head; uint32_t ent; };
    Key* src = (Key*)RadixGroupTempArena.alloc(count * sizeof(Key));
    Key* dst = (Key*)RadixGroupTempArena.alloc(count * sizeof(Key));

    for (uint32_t i = 0; i < count; ++i) {
    const auto& sv = string_stream[i];
    const char* p = sv.c_str() + common_skip;
    uint32_t rem = sv.GetLength() - common_skip;

    uint32_t load_cnt = (rem < 8) ? rem : 8;
    __mmask8 load_mask = (1U << load_cnt) - 1;

    uint64_t val =
__builtin_bswap64(_mm_cvtsi128_si64(_mm_maskz_loadu_epi8(load_mask, p)));



    src[i] = { ~val, i };
}


    for (int shift = 0; shift < 64; ) {
        uint32_t counts[2048] = {0}, offsets[2048];
        uint32_t mask = (shift > 50) ? 511 : 2047;
        for (uint32_t i = 0; i < count; ++i) counts[(src[i].head >> shift) &
mask]++; offsets[0] = 0; for (int j = 1; j <= mask; ++j) offsets[j] =
offsets[j-1] + counts[j-1]; for (uint32_t i = 0; i < count; ++i)
dst[offsets[(src[i].head >> shift) & mask]++] = src[i]; std::swap(src, dst);
        shift += 11;
    }


for (uint32_t i = 0; i < count; ) {
    uint32_t j = i + 1;

    while (j < count && src[j].head == src[i].head) j++;

    uint32_t n = j - i;
    if (n > 1) {

        if (n <= 32) {
            for (uint32_t k = i + 1; k < j; ++k) {
                Key pivot = src[k];
                uint32_t m = k;



                while (m > i && LexisReverseCompare(string_stream[pivot.ent],
string_stream[src[m - 1].ent])) { src[m] = src[m - 1];
                    --m;
                }
                src[m] = pivot;
            }
        } else {


            std::sort(src + i, src + j, [&](const Key& a, const Key& b) {
                return LexisReverseCompare(string_stream[a.ent],
string_stream[b.ent]);
            });
        }
    }
    i = j;
}


uint32_t* final_map = (uint32_t*)RadixGroupTempArena.alloc(count *
sizeof(uint32_t));

uint32_t* local_indices = (uint32_t*)RadixGroupTempArena.alloc(count *
sizeof(uint32_t));

for (uint32_t i = 0; i < count; ++i) {
    uint32_t orig_idx = src[i].ent;
    final_map[i] = entities[orig_idx];
    local_indices[i] = orig_idx;
}

reorder_set_simd(set, final_map, local_indices, count, RadixGroupTempArena);

    RadixGroupTempArena.set_offset(start_offset);
}*/
template <typename T_Comp, size_t FieldIdx, size_t Align, typename Set>
void SortGroupNumeric(Set &set, Registry<Align> &reg,
                      String_lib::UniversalArena<Align> &temp_arena)
{
    const uint32_t count = set.size();
    if (count < 2)
        return;

    auto *sort_stream = set.template get_stream<FieldIdx>();

    struct Key
    {
        uint32_t head;
        uint32_t ent;
    };
    Key *src = (Key *)temp_arena.alloc(count * sizeof(Key));
    Key *dst = (Key *)temp_arena.alloc(count * sizeof(Key));

    for (uint32_t i = 0; i < count; ++i)
    {
        T_Comp val = sort_stream[i];
        uint32_t u_val;
        static_assert(sizeof(T_Comp) == 4,
                      "Only 4-byte types supported for now");
        memcpy(&u_val, &val, 4);

        src[i] = {~u_val, i};
    }

    for (int shift = 0; shift < 32; shift += 11)
    {
        uint32_t counts[2048] = {0}, offsets[2048];
        uint32_t mask = (shift > 22) ? 1023 : 2047;
        for (uint32_t i = 0; i < count; ++i)
            counts[(src[i].head >> shift) & mask]++;
        offsets[0] = 0;
        for (int j = 1; j <= mask; ++j)
            offsets[j] = offsets[j - 1] + counts[j - 1];
        for (uint32_t i = 0; i < count; ++i)
            dst[offsets[(src[i].head >> shift) & mask]++] = src[i];
        std::swap(src, dst);
    }

    uint32_t *final_map =
        (uint32_t *)temp_arena.alloc(count * sizeof(uint32_t));
    uint32_t *local_indices =
        (uint32_t *)temp_arena.alloc(count * sizeof(uint32_t));
    uint32_t *original_entities = set.get_dense_ptr();

    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t orig_idx = src[i].ent;
        final_map[i] = original_entities[orig_idx];
        local_indices[i] = orig_idx;
    }

    reorder_set_simd(set, final_map, local_indices, count, temp_arena);
}

// not use!
template <typename Set, size_t Align>
void reorder_sparse_set(Set &set, uint32_t *sorted_entities, uint32_t count,
                        String_lib::UniversalArena<Align> &arena)
{

    for (uint32_t i = 0; i < count; ++i)
    {
        set.swap_entities(sorted_entities[i], set.get_entity_at(i));
    }
}

template <typename T> void RadSortRecursive(T *begin, T *end, size_t offset)
{
    const size_t count = end - begin;

    size_t start_offset = RadixTempArena.current_offset();

    uint32_t *indices =
        static_cast<uint32_t *>(RadixTempArena.alloc(count * sizeof(uint32_t)));
#pragma clang loop vectorize(enable) interleave(enable)
    for (uint32_t i = 0; i < count; ++i)
        indices[i] = i;

    uint8_t *bytes_pool =
        static_cast<uint8_t *>(RadixTempArena.alloc(count * sizeof(uint8_t)));

    RadSortIdxInternal(indices, indices + count, begin, bytes_pool, offset);

    T *temp = static_cast<T *>(RadixTempArena.alloc(count * sizeof(T)));

    for (size_t i = 0; i < count; ++i)
    {
        if (LIKELY(i + 32 < count))
        {
            __builtin_prefetch(&begin[indices[i + 32]], 0, 3);

            __builtin_prefetch(&indices[i + 64], 0, 0);
        }

        if constexpr (std::is_trivially_copyable_v<T>)
        {
            memcpy(&temp[i], &begin[indices[i]], sizeof(T));
        }
        else
        {
            new (&temp[i]) T(std::move(begin[indices[i]]));
        }
    }

    if constexpr (std::is_trivially_copyable_v<T>)
    {
        memcpy(begin, temp, count * sizeof(T));
    }
    else
    {
        for (size_t i = 0; i < count; ++i)
        {
            begin[i] = std::move(temp[i]);
            temp[i].~T();
        }
    }

    RadixTempArena.set_offset(start_offset);
}
/*

ALWAYS_INLINE uint32_t get_length(const SoftTicket& t) {
    return static_cast<uint32_t>(t.view.GetLength());
}

ALWAYS_INLINE const char* get_c_str(const SoftTicket& t) {
    return t.view.c_str();
}*/

template <typename T>
void RadSortRecursiveSkipPrefix(T *begin, T *end, size_t offset)
{
    const size_t count = end - begin;

    size_t start_offset = RadixTempArena.current_offset();

    size_t common_skip = offset;
    uint32_t min_len = 0xFFFFFFFF;

    for (size_t i = 0; i < count; ++i)
    {
        uint32_t l = get_length(begin[i]);
        if (l < min_len)
            min_len = l;
    }

    if (min_len > offset)
    {
        bool diff_found = false;
        const char *first_ptr = get_c_str(begin[0]);

        while (common_skip + 64 <= min_len)
        {
            __m512i base = _mm512_loadu_si512(first_ptr + common_skip);
            uint64_t total_mask = 0;
            for (size_t i = 1; i < count && i < 128; ++i)
            {
                __m512i curr =
                    _mm512_loadu_si512(get_c_str(begin[i]) + common_skip);
                total_mask |= _mm512_cmpneq_epi8_mask(base, curr);
                if (total_mask)
                    break;
            }
            if (total_mask)
            {
                common_skip += __builtin_ctzll(total_mask);
                diff_found = true;
                break;
            }
            common_skip += 64;
        }

        if (!diff_found)
        {
            while (common_skip < min_len)
            {
                char c = first_ptr[common_skip];
                bool match = true;
                for (size_t i = 1; i < count; ++i)
                {
                    if (get_c_str(begin[i])[common_skip] != c)
                    {
                        match = false;
                        break;
                    }
                }
                if (!match)
                    break;
                common_skip++;
            }
        }
    }

    uint32_t *indices =
        static_cast<uint32_t *>(RadixTempArena.alloc(count * sizeof(uint32_t)));
#pragma clang loop vectorize(enable) interleave(enable)
    for (uint32_t i = 0; i < count; ++i)
        indices[i] = i;

    uint8_t *bytes_pool =
        static_cast<uint8_t *>(RadixTempArena.alloc(count * sizeof(uint8_t)));

    RadSortIdxInternal(indices, indices + count, begin, bytes_pool,
                       common_skip);

    T *temp = static_cast<T *>(RadixTempArena.alloc(count * sizeof(T)));

    for (size_t i = 0; i < count; ++i)
    {
        if (LIKELY(i + 64 < count))
        {
            __builtin_prefetch(&begin[indices[i + 64]], 0, 3);

            __builtin_prefetch(&indices[i + 128], 0, 0);
        }

        if constexpr (std::is_trivially_copyable_v<T>)
        {
            memcpy(&temp[i], &begin[indices[i]], sizeof(T));
        }
        else
        {
            new (&temp[i]) T(std::move(begin[indices[i]]));
        }
    }

    if constexpr (std::is_trivially_copyable_v<T>)
    {
        memcpy(begin, temp, count * sizeof(T));
    }
    else
    {
        for (size_t i = 0; i < count; ++i)
        {
            begin[i] = std::move(temp[i]);
            temp[i].~T();
        }
    }

    RadixTempArena.set_offset(start_offset);
}

// interface
template <typename T> void RadSort(T *begin, T *end, size_t offset)
{
    if (UNLIKELY(end - begin < 2))
        return;
    RadSortRecursiveSkipPrefix(begin, end, offset);
}

template <typename T_Comp, size_t FieldIdx, typename... GroupComps,
          size_t Align>
void RadSortGroup(Group<Align, GroupComps...> &group, Registry<Align> &reg)
{

    const size_t start_offset = RadixTempArena.current_offset();

    SortGroup<T_Comp, FieldIdx>(group, reg, RadixTempArena);

    RadixTempArena.set_offset(start_offset);
}
} // namespace Radix_Internal