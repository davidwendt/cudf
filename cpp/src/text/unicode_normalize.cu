/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text/unicode_normalize.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/unicode_normalize.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_resource>
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/std/span>
#include <cuda/stream>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/uninitialized_fill.h>

#include <cstdint>

namespace nvtext {
namespace detail {
namespace {

// Composition exclusion: ~70 codepoints explicitly excluded from NFC/NFKC
// composition (Unicode 15, DerivedNormalizationProps.txt).
// Must be sorted ascending for binary_search.
// 0x2ADC (Supplemental Mathematical Operators) is placed after 0x0FB9 (Tibetan),
// not adjacent to the Hebrew block where it was previously listed out of order.
// clang-format off
__device__ __constant__ cuda::std::array COMPOSITION_EXCLUSIONS{
  0x0958u, 0x0959u, 0x095Au, 0x095Bu, 0x095Cu, 0x095Du, 0x095Eu, 0x095Fu, // Devanagari
  0x09DCu, 0x09DDu, 0x09DFu, // Bengali
  0x0A33u, 0x0A36u, // Gurmukhi
  0x0A59u, 0x0A5Au, 0x0A5Bu, 0x0A5Cu, 0x0A5Eu, // Gujarati
  0x0B5Cu, 0x0B5Du, // Oriya
  0x0F43u, 0x0F4Du, 0x0F52u, 0x0F57u, 0x0F5Cu, 0x0F69u, 0x0F76u, 0x0F78u, // Tibetan
  0x0F80u, 0x0F93u, 0x0F9Du, 0x0FA2u, 0x0FA7u, 0x0FACu, 0x0FB9u,
  0x2ADCu, // Supplemental Mathematical Operators
  0xFB1Du, 0xFB1Fu, 0xFB2Au, 0xFB2Bu, 0xFB2Cu, 0xFB2Du, 0xFB2Eu, // Hebrew Presentation Forms
  0xFB2Fu, 0xFB30u, 0xFB31u, 0xFB32u, 0xFB33u, 0xFB34u, 0xFB35u,
  0xFB36u, 0xFB38u, 0xFB39u, 0xFB3Au, 0xFB3Bu, 0xFB3Cu, 0xFB3Eu,
  0xFB40u, 0xFB41u, 0xFB43u, 0xFB44u, 0xFB46u, 0xFB47u, 0xFB48u,
  0xFB49u, 0xFB4Au, 0xFB4Bu, 0xFB4Cu, 0xFB4Du, 0xFB4Eu,
  0x1D15Eu, 0x1D15Fu, 0x1D160u, 0x1D161u, 0x1D162u, 0x1D163u, 0x1D164u, // Musical Symbols
  0x1D1BBu, 0x1D1BCu, 0x1D1BDu, 0x1D1BEu, 0x1D1BFu, 0x1D1C0u,
};
// clang-format on

/**
 * Invoke `fn` for each space-separated hex token in a decomp mapping string.
 * Returns immediately for empty strings or, when `apply_compat==false`, for
 * compatibility mappings (strings that begin with '<').  When `apply_compat==true`
 * the leading "<tag> " prefix is consumed before the iteration starts.
 */
template <typename Fn>
__device__ void for_each_decomp_token(cudf::string_view d_str, bool apply_compat, Fn fn)
{
  auto const size = d_str.size_bytes();
  if (size == 0) { return; }
  char const* const ptr = d_str.data();
  bool const is_compat  = (ptr[0] == '<');
  cudf::size_type pos   = 0;
  if (is_compat) {
    if (!apply_compat) { return; }
    while (pos < size && ptr[pos] != '>') {
      ++pos;
    }
    pos += 2;  // skip '>' and the following space
  }
  while (pos < size) {
    while (pos < size && ptr[pos] == ' ') {
      ++pos;
    }
    cudf::size_type const tok_start = pos;
    while (pos < size && ptr[pos] != ' ') {
      ++pos;
    }
    if (pos > tok_start) { fn(ptr + tok_start, pos - tok_start); }
  }
}

/**
 * Fused per-row setup kernel: scatter CCC, scatter decomposition token count,
 * and set NFC/NFKC quick-check flags — all in one pass over the unicode_data rows.
 *
 * Each thread owns one row exclusively (no cross-row read dependencies), so the
 * fusion is data-race-free.  compat_flags writes use cudf::set_bit (atomic)
 * because different rows may flag the same bit.
 *
 * One invocation per UnicodeData.txt row.
 */
struct setup_row_fn {
  cudf::column_device_view ccc_col;
  cudf::column_device_view decomp_map;
  cuda::std::span<uint32_t const> d_codepoints;
  bool apply_compat;
  cuda::std::span<uint8_t> ccc_table;                // output: CCC indexed by codepoint
  cuda::std::span<uint32_t> decomp_offsets;          // output: token count per codepoint
  cuda::std::span<cudf::bitmask_type> compat_flags;  // output: quick-check bits (empty=skip)

  __device__ void operator()(cudf::size_type idx) const
  {
    uint32_t const cp = d_codepoints[idx];

    // Scatter CCC
    if (cp <= MAX_CODEPOINT) {
      ccc_table[cp] = static_cast<uint8_t>(ccc_col.element<int32_t>(idx));
    }

    // Count apply_compat-aware tokens and scatter count to decomp_offsets
    auto const sv = decomp_map.element<cudf::string_view>(idx);
    auto count    = cudf::size_type{0};
    for_each_decomp_token(sv, apply_compat, [&count](char const*, cudf::size_type) { ++count; });
    if (cp <= MAX_CODEPOINT) { decomp_offsets[cp] = count; }

    // Set quick-check flags (NFC/NFKC only; compat_flags is empty for NFD/NFKD)
    if (!compat_flags.empty() && cp <= MAX_CODEPOINT) {
      bool const is_compat_row = sv.size_bytes() > 0 && sv.data()[0] == '<';
      // Flag compat decompositions (NFKC-unstable only; compat rows are NFC-stable
      // unless they are also singleton canonical decompositions, covered below) and
      // singleton canonical decompositions like U+212B -> U+00C5 (NFC-unstable).
      if ((is_compat_row && apply_compat) || count == 1) {
        cudf::set_bit(compat_flags.data(), static_cast<cudf::size_type>(cp));
      }
    }
  }
};

/**
 * Propagate quick-check flags to canonical multi-token decompositions whose
 * expansion contains at least one already-flagged codepoint.
 *
 * Handles indirect NFKC_QC=No codepoints, e.g. U+0385 GREEK DIALYTIKA TONOS,
 * which canonically decomposes to U+00A8 (compat-flagged) + U+0301.
 */
struct propagate_compat_flag_fn {
  cudf::column_device_view decomp_map;
  cuda::std::span<uint32_t const> d_codepoints;
  cuda::std::span<cudf::bitmask_type> compat_flags;

  __device__ void operator()(cudf::size_type idx) const
  {
    auto const sv = decomp_map.element<cudf::string_view>(idx);
    if (sv.size_bytes() == 0) { return; }
    if (sv.data()[0] == '<') { return; }  // compat-tagged: already handled
    uint32_t const cp = d_codepoints[idx];
    if (cp > MAX_CODEPOINT) { return; }
    if (cudf::bit_is_set(compat_flags.data(), static_cast<cudf::size_type>(cp))) { return; }

    bool needs_flag   = false;
    auto const& flags = compat_flags;
    auto fn           = [&needs_flag, &flags](char const* ptr, cudf::size_type size) {
      uint32_t const token_cp = hex_to_cp(ptr, size);
      if (token_cp <= MAX_CODEPOINT &&
          cudf::bit_is_set(flags.data(), static_cast<cudf::size_type>(token_cp))) {
        needs_flag = true;
      }
    };
    for_each_decomp_token(sv, /*apply_compat=*/false, fn);
    if (needs_flag) { cudf::set_bit(compat_flags.data(), static_cast<cudf::size_type>(cp)); }
  }
};

/**
 * Write decomposition codepoints into the flat decomp_table.
 * One invocation per row; uses pre-computed per-codepoint offsets for placement.
 */
struct write_decomp_tokens_fn {
  cudf::column_device_view decomp_map;
  bool apply_compat;
  cuda::std::span<uint32_t const> d_codepoints;       // parsed codepoint per row
  cuda::std::span<uint32_t const> decomp_cp_offsets;  // write-start per codepoint
  cuda::std::span<uint32_t> decomp_table;             // flat output decomp codepoints

  __device__ void operator()(cudf::size_type idx) const
  {
    auto const cp = d_codepoints[idx];
    if (cp > MAX_CODEPOINT) { return; }
    auto write_pos = decomp_cp_offsets[cp];
    auto fn        = [this, &write_pos](char const* ptr, cudf::size_type size) {
      decomp_table[write_pos++] = hex_to_cp(ptr, size);
    };
    for_each_decomp_token(decomp_map.element<cudf::string_view>(idx), apply_compat, fn);
  }
};

/**
 * Composition table key for a (starter, combining) codepoint pair.
 * The starter occupies the upper 32 bits so keys sort by starter first.
 */
__device__ inline uint64_t composition_key(uint32_t starter, uint32_t combining)
{
  constexpr uint32_t starter_shift = 32;
  return (static_cast<uint64_t>(starter) << starter_shift) | combining;
}

/**
 * Build composition table entries from canonical two-token decompositions.
 * Writes a (key, value) pair per qualifying row; zero for non-qualifying rows.
 */
struct build_comp_table_fn {
  cudf::column_device_view decomp_map;
  cuda::std::span<uint32_t const> d_codepoints;      // parsed codepoint per row
  cuda::std::span<uint8_t const> ccc_table;          // CCC indexed by codepoint
  cuda::std::span<cudf::bitmask_type> compat_flags;  // NFC/NFKC quick-check bitset
  cuda::std::span<uint64_t> d_comp_keys;             // output: composition key
  cuda::std::span<uint32_t> d_comp_values;           // output: composed codepoint

  __device__ void operator()(cudf::size_type idx) const
  {
    d_comp_keys[idx]   = 0;
    d_comp_values[idx] = 0;
    // Extract canonical tokens (apply_compat=false skips compat mappings).
    // Count beyond 2 so rows with more than two tokens are correctly rejected.
    uint32_t tokens[2] = {0, 0};
    int32_t tok        = 0;
    auto fn            = [&tokens, &tok](char const* ptr, cudf::size_type size) {
      if (tok < 2) { tokens[tok] = hex_to_cp(ptr, size); }
      ++tok;
    };
    for_each_decomp_token(decomp_map.element<cudf::string_view>(idx), false, fn);
    if (tok != 2) { return; }
    auto const composed = d_codepoints[idx];
    if (composed > MAX_CODEPOINT) { return; }
    auto const starter   = tokens[0];
    auto const combining = tokens[1];
    if (cuda::std::binary_search(
          COMPOSITION_EXCLUSIONS.begin(), COMPOSITION_EXCLUSIONS.end(), composed)) {
      // Script/explicit exclusion: NFC_QC=No, flag it so quick check catches it
      cudf::set_bit(compat_flags.data(), static_cast<cudf::size_type>(composed));
      return;
    }

    if (starter > MAX_CODEPOINT || combining > MAX_CODEPOINT) { return; }
    if (ccc_table[starter] != 0) {
      // Non-starter decomposition: NFC_QC=No, flag it so quick check catches it
      cudf::set_bit(compat_flags.data(), static_cast<cudf::size_type>(composed));
      return;
    }
    d_comp_keys[idx]   = composition_key(starter, combining);
    d_comp_values[idx] = composed;
    // CCC=0 second operands are not caught by the ccc_table quick-check path;
    // flag them explicitly so nfc_quick_check_fn triggers the full pipeline.
    if (combining <= MAX_CODEPOINT && ccc_table[combining] == 0) {
      cudf::set_bit(compat_flags.data(), static_cast<cudf::size_type>(combining));
    }
  }
};

struct is_zero_comp_key_fn {
  __device__ bool operator()(cuda::std::tuple<uint64_t, uint32_t> const& kv) const
  {
    return cuda::std::get<0>(kv) == uint64_t{0};
  }
};

}  // namespace
}  // namespace detail

struct unicode_normalizer::unicode_normalizer_impl {
  rmm::device_uvector<uint32_t> decomp_offsets;  // size DECOMP_OFFSETS_SIZE
  rmm::device_uvector<uint32_t> decomp_table;    // flat replacement codepoints
  rmm::device_uvector<uint8_t> ccc_table;        // size CODEPOINT_TABLE_SIZE
  rmm::device_uvector<cudf::bitmask_type> compat_decomp_flags;
  rmm::device_uvector<uint64_t> comp_keys;    // sorted (starter<<32|combining)
  rmm::device_uvector<uint32_t> comp_values;  // parallel composed codepoints
  unicode_normalization_form form;

  unicode_normalizer_impl(rmm::device_uvector<uint32_t>&& decomp_offsets,
                          rmm::device_uvector<uint32_t>&& decomp_table,
                          rmm::device_uvector<uint8_t>&& ccc_table,
                          rmm::device_uvector<cudf::bitmask_type>&& compat_decomp_flags,
                          rmm::device_uvector<uint64_t>&& comp_keys,
                          rmm::device_uvector<uint32_t>&& comp_values,
                          unicode_normalization_form form)
    : decomp_offsets(std::move(decomp_offsets)),
      decomp_table(std::move(decomp_table)),
      ccc_table(std::move(ccc_table)),
      compat_decomp_flags(std::move(compat_decomp_flags)),
      comp_keys(std::move(comp_keys)),
      comp_values(std::move(comp_values)),
      form(form)
  {
  }
};

unicode_normalizer::unicode_normalizer(cudf::table_view const& unicode_data,
                                       unicode_normalization_form form,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(unicode_data.num_columns() == 3,
               "unicode_data table must have exactly 3 columns",
               std::invalid_argument);
  CUDF_EXPECTS(unicode_data.column(0).type().id() == cudf::type_id::STRING,
               "unicode_data column[0] must be STRING",
               std::invalid_argument);
  CUDF_EXPECTS(unicode_data.column(1).type().id() == cudf::type_id::INT32,
               "unicode_data column[1] must be INT32",
               std::invalid_argument);
  CUDF_EXPECTS(unicode_data.column(2).type().id() == cudf::type_id::STRING,
               "unicode_data column[2] must be STRING",
               std::invalid_argument);
  CUDF_EXPECTS(!cudf::has_nulls(unicode_data),
               "unicode_data table must not contain nulls",
               std::invalid_argument);

  cudf::size_type const num_rows = unicode_data.num_rows();
  CUDF_EXPECTS(num_rows > 0, "unicode_data table must not be empty", std::invalid_argument);

  auto temp_mr = cudf::get_current_device_resource_ref();
  auto codepoints_col =
    cudf::strings::detail::hex_to_integers(cudf::strings_column_view(unicode_data.column(0)),
                                           cudf::data_type{cudf::type_id::UINT32},
                                           stream,
                                           temp_mr);
  auto d_codepoints = cuda::std::span<uint32_t const>(codepoints_col->view().data<uint32_t>(),
                                                      static_cast<std::size_t>(num_rows));

  auto const d_ccc_col    = cudf::column_device_view::create(unicode_data.column(1), stream);
  auto const d_decomp_map = cudf::column_device_view::create(unicode_data.column(2), stream);
  bool const apply_compat =
    (form == unicode_normalization_form::NFKD || form == unicode_normalization_form::NFKC);
  auto const policy   = rmm::exec_policy_nosync(stream, temp_mr);
  auto const row_iter = cuda::make_counting_iterator(cudf::size_type{0});

  // Build Canonical Combining Class (CCC) table
  auto ccc_table = cudf::detail::make_zeroed_device_uvector_async<uint8_t>(
    detail::CODEPOINT_TABLE_SIZE, stream, mr);

  // Allocate compat_decomp_flags only for NFC/NFKC (NFD/NFKD never run the quick check).
  bool const need_compat_flags =
    (form == unicode_normalization_form::NFC || form == unicode_normalization_form::NFKC);
  auto compat_decomp_flags = rmm::device_uvector<cudf::bitmask_type>(
    need_compat_flags ? cudf::num_bitmask_words(detail::CODEPOINT_TABLE_SIZE) : 0, stream, mr);
  if (need_compat_flags) {
    thrust::uninitialized_fill(
      policy, compat_decomp_flags.begin(), compat_decomp_flags.end(), uint32_t{0});
  }

  // Fused single-pass kernel: scatter CCC values, scatter per-codepoint decomposition
  // token counts into decomp_offsets, and (for NFC/NFKC) set initial quick-check flags
  // for compat decompositions and singleton canonical decompositions.
  auto decomp_offsets = cudf::detail::make_zeroed_device_uvector_async<uint32_t>(
    detail::DECOMP_OFFSETS_SIZE, stream, mr);
  thrust::for_each_n(policy,
                     row_iter,
                     num_rows,
                     detail::setup_row_fn{*d_ccc_col,
                                          *d_decomp_map,
                                          d_codepoints,
                                          apply_compat,
                                          ccc_table,
                                          decomp_offsets,
                                          compat_decomp_flags});

  // Propagate quick-check flags to canonical decompositions whose expansion contains
  // an already-flagged codepoint (e.g. U+0385 -> U+00A8 + U+0301 where U+00A8 is
  // compat-flagged). Must follow setup_row_fn so all direct flags are visible.
  if (need_compat_flags) {
    auto prop_flag_fn =
      detail::propagate_compat_flag_fn{*d_decomp_map, d_codepoints, compat_decomp_flags};
    thrust::for_each_n(policy, row_iter, num_rows, prop_flag_fn);
  }

  // In-place exclusive scan of decomp_offsets: each codepoint's slot becomes
  // its start offset in the flat decomp_table.  The extra sentinel slot at
  // MAX_CODEPOINT+1 accumulates the total via the scan.
  auto const total_decomp_size = cudf::detail::sizes_to_offsets(
    decomp_offsets.begin(), decomp_offsets.end(), decomp_offsets.begin(), 0, stream, temp_mr);

  // Fill decomp_table
  auto decomp_table    = rmm::device_uvector<uint32_t>(total_decomp_size, stream, mr);
  auto write_tokens_fn = detail::write_decomp_tokens_fn{
    *d_decomp_map, apply_compat, d_codepoints, decomp_offsets, decomp_table};
  thrust::for_each_n(policy, row_iter, num_rows, write_tokens_fn);

  if (!need_compat_flags) {
    _impl = std::make_unique<unicode_normalizer_impl>(
      std::move(decomp_offsets),
      std::move(decomp_table),
      std::move(ccc_table),
      rmm::device_uvector<cudf::bitmask_type>(0, stream, mr),  // unused for NFD/NFKD
      rmm::device_uvector<uint64_t>(0, stream, mr),
      rmm::device_uvector<uint32_t>(0, stream, mr),
      form);
    return;
  }

  // Build composition table (NFC/NFKC only)
  auto d_comp_keys    = rmm::device_uvector<uint64_t>(num_rows, stream, temp_mr);
  auto d_comp_values  = rmm::device_uvector<uint32_t>(num_rows, stream, temp_mr);
  auto build_table_fn = detail::build_comp_table_fn{
    *d_decomp_map, d_codepoints, ccc_table, compat_decomp_flags, d_comp_keys, d_comp_values};
  thrust::for_each_n(policy, row_iter, num_rows, build_table_fn);

  // Compact keys and values together in one pass: remove any (key, value) pair
  // where the key is 0 (rows that build_comp_table_fn left empty).
  auto kv_begin        = cuda::make_zip_iterator(d_comp_keys.begin(), d_comp_values.begin());
  auto kv_end          = cuda::make_zip_iterator(d_comp_keys.end(), d_comp_values.end());
  auto const end_itr   = thrust::remove_if(policy, kv_begin, kv_end, detail::is_zero_comp_key_fn{});
  auto const comp_size = static_cast<std::size_t>(end_itr - kv_begin);

  // Copy only the compacted prefix into exact-size allocations; the tail past
  // comp_size holds unspecified leftovers from remove_if.
  auto comp_keys = cudf::detail::make_device_uvector_async(
    cudf::device_span<uint64_t const>(d_comp_keys.data(), comp_size), stream, mr);
  auto comp_values = cudf::detail::make_device_uvector_async(
    cudf::device_span<uint32_t const>(d_comp_values.data(), comp_size), stream, mr);

  thrust::sort_by_key(policy, comp_keys.begin(), comp_keys.end(), comp_values.begin());

  _impl = std::make_unique<unicode_normalizer_impl>(std::move(decomp_offsets),
                                                    std::move(decomp_table),
                                                    std::move(ccc_table),
                                                    std::move(compat_decomp_flags),
                                                    std::move(comp_keys),
                                                    std::move(comp_values),
                                                    form);
}

unicode_normalizer::~unicode_normalizer() {}

std::unique_ptr<unicode_normalizer> create_unicode_normalizer(cudf::table_view const& unicode_data,
                                                              unicode_normalization_form form,
                                                              cuda::stream_ref stream,
                                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return std::make_unique<unicode_normalizer>(unicode_data, form, stream, mr);
}

namespace detail {
namespace {

// Packed codepoint slot layout (one uint32_t per expanded codepoint):
//   bits 20:0  — Unicode codepoint (21 bits, range 0x000000–0x10FFFF)
//   bits 28:21 — Canonical Combining Class (8 bits, range 0–254)
//   bit  29    — consumed-by-composition flag (set when the slot is eliminated)
//   bits 31:30 — UTF-8 encoded width of the codepoint minus 1
// A consumed slot holds only the consumed flag so it contributes no output bytes.
constexpr uint32_t PACKED_CP_MASK      = 0x001F'FFFFu;  // bits 20:0
constexpr uint32_t PACKED_CCC_SHIFT    = 21u;
constexpr uint32_t PACKED_CCC_MASK     = 0xFFu;
constexpr uint32_t PACKED_CONSUMED_BIT = 1u << 29;  // bit 29
constexpr uint32_t PACKED_WIDTH_SHIFT  = 30u;

__device__ __forceinline__ uint32_t pack_cp_ccc(uint32_t cp, uint8_t ccc)
{
  auto const width =
    cudf::strings::detail::bytes_in_char_utf8(cudf::strings::detail::codepoint_to_utf8(cp));
  return (static_cast<uint32_t>(width - 1) << PACKED_WIDTH_SHIFT) |
         (static_cast<uint32_t>(ccc) << PACKED_CCC_SHIFT) | (cp & PACKED_CP_MASK);
}
__device__ __forceinline__ uint32_t cp_of(uint32_t packed) { return packed & PACKED_CP_MASK; }
__device__ __forceinline__ uint8_t ccc_of(uint32_t packed)
{
  return static_cast<uint8_t>((packed >> PACKED_CCC_SHIFT) & PACKED_CCC_MASK);
}
__device__ __forceinline__ bool is_consumed(uint32_t packed)
{
  return (packed & PACKED_CONSUMED_BIT) != 0u;
}
__device__ __forceinline__ int64_t utf8_width_of(uint32_t packed)
{
  return is_consumed(packed) ? 0 : static_cast<int64_t>(packed >> PACKED_WIDTH_SHIFT) + 1;
}

/**
 * Transitively decompose the codepoint whose UTF-8 encoding starts at @p idx.
 *
 * Runs the full NFD/NFKD ping-pong expansion loop and calls `fn(i, cp)` for each
 * of the resulting codepoints in order.
 *
 * @return The number of codepoints passed to `fn` (0 for intermediate UTF-8 bytes)
 */
template <typename Fn>
__device__ int32_t for_each_decomposed_cp(int64_t idx,
                                          cuda::std::span<char const> chars,
                                          cuda::std::span<uint32_t const> decomp_offsets,
                                          cuda::std::span<uint32_t const> decomp_table,
                                          Fn fn)
{
  if (!cudf::strings::detail::is_begin_utf8_char(chars[idx])) { return 0; }
  cudf::char_utf8 ch = 0;
  cudf::strings::detail::to_char_utf8(chars.data() + idx, ch);
  uint32_t const cp = cudf::strings::detail::utf8_to_codepoint(ch);

  // Fast path: most codepoints have no decomposition so the expansion buffers are not needed
  bool const is_hangul = (cp >= HANGUL_SBASE && cp <= HANGUL_SEND);
  if (!is_hangul && (cp > MAX_CODEPOINT || decomp_offsets[cp] == decomp_offsets[cp + 1])) {
    fn(0, cp);
    return 1;
  }

  uint32_t buf_a[MAX_DECOMP_EXPAND];
  uint32_t buf_b[MAX_DECOMP_EXPAND];
  int32_t count_a = 1;
  buf_a[0]        = cp;
  for (int32_t depth = 0; depth < MAX_DECOMP_DEPTH; ++depth) {
    int32_t count_b = 0;
    bool expanded   = false;
    for (int32_t i = 0; i < count_a; ++i) {
      auto const cp = buf_a[i];
      if (cp >= HANGUL_SBASE && cp <= HANGUL_SEND) {
        if (count_b + 3 <= MAX_DECOMP_EXPAND) {
          count_b += hangul_decompose(cp, buf_b + count_b);
          expanded = true;
        }
      } else if (cp > MAX_CODEPOINT) {
        if (count_b < MAX_DECOMP_EXPAND) { buf_b[count_b++] = cp; }  // out-of-range: pass through
      } else {
        auto const start = decomp_offsets[cp];
        auto const end   = decomp_offsets[cp + 1];
        if (start == end) {
          buf_b[count_b++] = cp;
        } else {
          auto copy_size =
            cuda::std::min(end - start, static_cast<uint32_t>(MAX_DECOMP_EXPAND - count_b));
          cuda::std::memcpy(
            buf_b + count_b, decomp_table.data() + start, copy_size * sizeof(uint32_t));
          count_b += copy_size;
          expanded = true;
        }
      }
    }
    cuda::std::memcpy(buf_a, buf_b, count_b * sizeof(uint32_t));
    count_a = count_b;
    if (!expanded) { break; }
  }
  for (int32_t i = 0; i < count_a; ++i) {
    fn(i, buf_a[i]);
  }
  return count_a;
}

/**
 * Decomposes the input bytes into codepoints using the normalizer's tables
 */
struct decompose_fn {
  cuda::std::span<char const> d_input_chars;
  cuda::std::span<uint32_t const> decomp_offsets;
  cuda::std::span<uint32_t const> decomp_table;
  cuda::std::span<uint8_t const> ccc_table;

  /// Returns the number of codepoints for the input byte at @p idx (at most MAX_DECOMP_EXPAND)
  __device__ int32_t count(int64_t idx) const
  {
    return for_each_decomposed_cp(
      idx, d_input_chars, decomp_offsets, decomp_table, [](int32_t, uint32_t) {});
  }

  /// Writes the packed slots for the input byte at @p idx to @p d_out
  __device__ void fill(int64_t idx, uint32_t* d_out) const
  {
    for_each_decomposed_cp(
      idx, d_input_chars, decomp_offsets, decomp_table, [this, d_out](int32_t i, uint32_t cp) {
        auto const ccc = (cp <= MAX_CODEPOINT) ? ccc_table[cp] : uint8_t{0};
        d_out[i]       = pack_cp_ccc(cp, ccc);
      });
  }
};

constexpr int32_t decompose_block_size = 256;

/**
 * @brief Computes the number of decomposed codepoints for each input byte
 *
 * Launched as a thread per input byte.
 *
 * @param fn Decomposes each input byte
 * @param d_counts Number of codepoints for each input byte
 * @param d_block_counts Total number of codepoints for each block
 */
CUDF_KERNEL void decompose_count_kernel(decompose_fn fn, uint8_t* d_counts, int64_t* d_block_counts)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  int32_t count  = 0;
  if (idx < static_cast<int64_t>(fn.d_input_chars.size())) {
    count         = fn.count(idx);
    d_counts[idx] = static_cast<uint8_t>(count);
  }
  using block_reduce = cub::BlockReduce<int32_t, decompose_block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;
  auto const block_count = block_reduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) { d_block_counts[blockIdx.x] = block_count; }
}

/**
 * @brief Writes the packed slots of the decomposed codepoints for each input byte
 *
 * Launched as a thread per input byte. Each thread's slots are located using the
 * exclusive scan of the block's counts added to the block's offset.
 *
 * @param fn Decomposes each input byte
 * @param d_counts Number of codepoints for each input byte
 * @param d_block_offsets Offset of the first slot for each block
 * @param d_cps Output packed slots
 */
CUDF_KERNEL void decompose_fill_kernel(decompose_fn fn,
                                       uint8_t const* d_counts,
                                       int64_t const* d_block_offsets,
                                       uint32_t* d_cps)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  int32_t const count =
    idx < static_cast<int64_t>(fn.d_input_chars.size()) ? d_counts[idx] : int32_t{0};
  using block_scan = cub::BlockScan<int32_t, decompose_block_size>;
  __shared__ typename block_scan::TempStorage temp_storage;
  int32_t offset = 0;
  block_scan(temp_storage).ExclusiveSum(count, offset);
  if (count > 0) { fn.fill(idx, d_cps + d_block_offsets[blockIdx.x] + offset); }
}

/**
 * Stable-sort combining mark runs within a string's codepoint slice.
 * One invocation per string; insertion-sort each maximal run of CCC>0 marks.
 * d_cps holds packed (cp | ccc) slots; CCC is extracted from the packed value.
 */
struct reorder_fn {
  cuda::std::span<uint32_t> d_cps;  // packed cp+ccc slots
  cuda::std::span<int64_t const> d_str_cp_offsets;

  __device__ void operator()(cudf::size_type str_idx) const
  {
    auto const cp_start = d_str_cp_offsets[str_idx];
    auto const cp_end   = d_str_cp_offsets[str_idx + 1];
    auto run_start      = cp_start;
    for (int64_t i = cp_start; i <= cp_end; ++i) {
      bool const is_combining = (i < cp_end) && (ccc_of(d_cps[i]) > 0);
      if (is_combining) { continue; }
      auto const run_len = i - run_start;
      if (run_len > 1) {
        // Insertion sort: upper_bound locates the insertion point by CCC, then
        // a single rotate on the packed array moves both cp and ccc together.
        for (int64_t j = run_start + 1; j < i; ++j) {
          auto const ccc_j = ccc_of(d_cps[j]);
          // upper_bound(begin, end, value, comp): returns first element where comp(value, elem)
          // is true, i.e., first packed slot whose CCC exceeds ccc_j.
          auto const ins = cuda::std::upper_bound(
                             d_cps.begin() + run_start,
                             d_cps.begin() + j,
                             ccc_j,
                             [](uint8_t val, uint32_t packed) { return val < ccc_of(packed); }) -
                           d_cps.begin();
          if (ins < j) {
            cuda::std::rotate(d_cps.begin() + ins, d_cps.begin() + j, d_cps.begin() + j + 1);
          }
        }
      }
      run_start = i + 1;
    }
  }
};

/**
 * Canonical composition pass (NFC/NFKC only).
 * One invocation per string.  The composition table is small (~600 entries,
 * ~7 KB) and accessed by all strings, so it stays L2-hot throughout execution.
 * Consumed slots are marked with PACKED_CONSUMED_BIT and skipped by output_fn.
 * Composed starters always have CCC=0, so pack_cp_ccc(composed, 0) needs no
 * additional CCC table lookup.
 */
struct compose_fn {
  cuda::std::span<uint32_t> d_cps;  // packed cp+ccc slots
  cuda::std::span<int64_t const> d_str_cp_offsets;
  cuda::std::span<uint64_t const> comp_keys;
  cuda::std::span<uint32_t const> comp_values;
  cuda::std::span<cudf::bitmask_type const> compat_flags;  // NFC/NFKC quick-check bitset

  __device__ void operator()(cudf::size_type str_idx) const
  {
    auto const cp_start  = d_str_cp_offsets[str_idx];
    auto const cp_end    = d_str_cp_offsets[str_idx + 1];
    int64_t last_starter = -1;
    uint8_t last_class   = 0;

    for (int64_t i = cp_start; i < cp_end; ++i) {
      auto const packed_i = d_cps[i];
      if (is_consumed(packed_i)) { continue; }
      uint8_t const ccc = ccc_of(packed_i);
      if (last_starter < 0) {
        last_starter = ccc == 0 ? i : last_starter;
        last_class   = ccc;
        continue;
      }
      if (ccc == 0) {
        // New starter — attempt composition only when unblocked (last_class == 0).
        // Try Hangul algorithmic composition first, then the canonical table for
        // starter+starter pairs (e.g. Bengali U+09C7 + U+09BE → U+09CB).
        if (last_class == 0) {
          auto const composed_hangul = hangul_compose(cp_of(d_cps[last_starter]), cp_of(packed_i));
          if (composed_hangul != 0) {
            d_cps[last_starter] = pack_cp_ccc(composed_hangul, 0);
            d_cps[i]            = PACKED_CONSUMED_BIT;
            continue;
          }
          // Only codepoints flagged as CCC=0 second operands can match a table key here
          auto const cp = cp_of(packed_i);
          if (cp <= MAX_CODEPOINT &&
              cudf::bit_is_set(compat_flags.data(), static_cast<cudf::size_type>(cp))) {
            auto const key = composition_key(cp_of(d_cps[last_starter]), cp);
            auto const it  = cuda::std::lower_bound(comp_keys.begin(), comp_keys.end(), key);
            if (it != comp_keys.end() && *it == key) {
              d_cps[last_starter] =
                pack_cp_ccc(comp_values[cuda::std::distance(comp_keys.begin(), it)], 0);
              d_cps[i] = PACKED_CONSUMED_BIT;
              continue;
            }
          }
        }
        last_starter = i;
      } else {
        // Combining mark: compose with last_starter
        if (last_class < ccc) {
          auto const key = composition_key(cp_of(d_cps[last_starter]), cp_of(packed_i));
          auto const it  = cuda::std::lower_bound(comp_keys.begin(), comp_keys.end(), key);
          if (it != comp_keys.end() && *it == key) {
            d_cps[last_starter] =
              pack_cp_ccc(comp_values[cuda::std::distance(comp_keys.begin(), it)], 0);
            d_cps[i] = PACKED_CONSUMED_BIT;
            continue;
          }
        }
      }
      last_class = ccc;
    }
  }
};

/**
 * Fused canonical reorder + composition for NFC/NFKC.
 * One thread per string reorders and then immediately composes its codepoint
 * interval, eliminating a kernel launch and giving composition a warm L2 cache
 * for the row just touched by reorder.
 */
struct reorder_and_compose_fn {
  reorder_fn reorder;
  compose_fn compose;

  __device__ void operator()(cudf::size_type str_idx) const
  {
    reorder(str_idx);
    compose(str_idx);
  }
};

/**
 * NFC/NFKC quick-check predicate.
 *
 * Returns true for the first byte of any UTF-8 sequence whose codepoint
 * requires the full normalization pipeline:
 *   - Non-zero CCC (combining mark): may need reorder or table-based composition.
 *   - Hangul V jamo (U+1161–U+1175) or T jamo (U+11A8–U+11C2): NFC_QC=Maybe;
 *     can compose algorithmically with a preceding L or LV syllable.
 *   - Compat-decomp or singleton-canonical flag: unstable under NFC/NFKC.
 *
 * If no such byte exists the column is already in NFC/NFKC form and the
 * early-return copy path fires.
 */
struct nfc_quick_check_fn {
  cuda::std::span<char const> chars;
  cuda::std::span<uint8_t const> ccc_table;
  cuda::std::span<cudf::bitmask_type const> compat_flags;

  __device__ bool operator()(int64_t idx) const
  {
    if (!cudf::strings::detail::is_begin_utf8_char(chars[idx])) { return false; }
    cudf::char_utf8 ch = 0;
    cudf::strings::detail::to_char_utf8(chars.data() + idx, ch);
    auto const cp = cudf::strings::detail::utf8_to_codepoint(ch);
    if (cp > MAX_CODEPOINT) { return false; }
    if (ccc_table[cp] > 0) { return true; }
    if ((cp >= HANGUL_VBASE && cp <= HANGUL_VEND) || (cp >= HANGUL_TSTART && cp <= HANGUL_TEND)) {
      return true;
    }
    return !compat_flags.empty() &&
           cudf::bit_is_set(compat_flags.data(), static_cast<cudf::size_type>(cp));
  }
};

/**
 * Write the UTF-8 bytes of the packed slot at @p idx starting at @p out_pos.
 *
 * Called through a tabulate output iterator by the exclusive scan of the slot widths.
 * The rows are contiguous in the slots so the scan gives each slot's position in the
 * output chars directly.
 */
struct write_utf8_fn {
  uint32_t const* d_cps;
  char* d_chars;

  __device__ void operator()(int64_t idx, int64_t out_pos) const
  {
    auto const packed = d_cps[idx];
    if (is_consumed(packed)) { return; }
    auto const utf8 = cudf::strings::detail::codepoint_to_utf8(cp_of(packed));
    cudf::strings::detail::from_char_utf8(utf8, d_chars + out_pos);
  }
};

struct utf8_width_fn {
  __device__ int64_t operator()(uint32_t packed) const { return utf8_width_of(packed); }
};

/**
 * @brief Sums `d_in` over the segments [d_begin[i], d_end[i]) for each of the `size` rows
 */
template <typename InputIterator, typename OffsetIterator, typename OutputType>
void segmented_sum(InputIterator d_in,
                   OutputType* d_out,
                   cudf::size_type size,
                   OffsetIterator d_begin,
                   OffsetIterator d_end,
                   cuda::stream_ref stream)
{
  auto const env =
    cuda::std::execution::env{cuda::std::execution::prop{cuda::get_stream_t{}, stream},
                              cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                                                         cudf::get_current_device_resource_ref()}};
  CUDF_CUDA_TRY(cub::DeviceSegmentedReduce::Sum(d_in, d_out, size, d_begin, d_end, env));
}

/**
 * @brief In-place exclusive sum of `d_data[0, num_items)`
 */
template <typename T>
void exclusive_sum(T* d_data, int64_t num_items, cuda::stream_ref stream)
{
  auto const env =
    cuda::std::execution::env{cuda::std::execution::prop{cuda::get_stream_t{}, stream},
                              cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                                                         cudf::get_current_device_resource_ref()}};
  CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(d_data, num_items, env));
}

/**
 * @brief Exclusive sum of `d_in` written through a tabulate output iterator calling `fn`
 */
template <typename InputIterator, typename Fn>
void scan_with(InputIterator d_in, Fn fn, int64_t num_items, cuda::stream_ref stream)
{
  auto const env =
    cuda::std::execution::env{cuda::std::execution::prop{cuda::get_stream_t{}, stream},
                              cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                                                         cudf::get_current_device_resource_ref()}};
  CUDF_CUDA_TRY(
    cub::DeviceScan::ExclusiveSum(d_in, cuda::tabulate_output_iterator(fn), num_items, env));
}

}  // namespace

std::unique_ptr<cudf::column> normalize_unicode(cudf::strings_column_view const& input,
                                                unicode_normalizer const& normalizer,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}); }

  auto const [first_offset, last_offset] =
    cudf::strings::detail::get_first_and_last_offset(input, stream);
  auto const chars_size = last_offset - first_offset;
  if (chars_size == 0) { return std::make_unique<cudf::column>(input.parent(), stream, mr); }

  auto const& p          = *normalizer._impl;
  auto const temp_mr     = cudf::get_current_device_resource_ref();
  auto const policy      = rmm::exec_policy_nosync(stream, temp_mr);
  auto const byte_iter   = cuda::make_counting_iterator(int64_t{0});
  auto const d_raw_chars = input.chars_begin(stream) + first_offset;
  auto const chars_span  = cuda::std::span<char const>(d_raw_chars, chars_size);

  // NFC/NFKC quick check: scan for any codepoint that is NFC_QC=No or NFC_QC=Maybe
  // (non-zero Canonical Combining Class (CCC), Hangul V/T jamo, compat decomp, singleton canonical,
  // script exclusion, or non-starter decomposition).
  // If none found the column is already normalized and we can just return a copy.
  if (p.form == unicode_normalization_form::NFC || p.form == unicode_normalization_form::NFKC) {
    auto nfc_qc_fn = detail::nfc_quick_check_fn{chars_span, p.ccc_table, p.compat_decomp_flags};
    if (!cudf::detail::any_of(byte_iter, byte_iter + chars_size, nfc_qc_fn, stream)) {
      return std::make_unique<cudf::column>(input.parent(), stream, mr);
    }
  }

  auto const num_rows = input.size();

  // Decomposition: the number of output codepoints for each input byte (0 for non-lead bytes)
  // fits in a uint8_t. These counts are summed per row for the row boundaries in the packed
  // slots and scanned within each block to locate each byte's slots when filling them.
  auto const decomposer =
    detail::decompose_fn{chars_span, p.decomp_offsets, p.decomp_table, p.ccc_table};
  cudf::detail::grid_1d const grid{chars_size, detail::decompose_block_size};
  auto str_cp_offsets = cuda::device_buffer<int64_t>{
    stream, temp_mr, static_cast<std::size_t>(num_rows + 1), cuda::no_init};
  auto cps = [&] {
    auto d_counts = cuda::device_buffer<uint8_t>{
      stream, temp_mr, static_cast<std::size_t>(chars_size), cuda::no_init};
    auto d_block_offsets = cuda::device_buffer<int64_t>{
      stream, temp_mr, static_cast<std::size_t>(grid.num_blocks), cuda::no_init};
    detail::
      decompose_count_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
        decomposer, d_counts.data(), d_block_offsets.data());
    CUDF_CUDA_TRY(cudaGetLastError());
    detail::exclusive_sum(d_block_offsets.data(), grid.num_blocks, stream);

    // the counts are indexed from first_offset so the row offsets are normalized to match
    auto const d_offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
    auto const d_row_offsets = cuda::transform_iterator(
      d_offsets, cuda::proclaim_return_type<int64_t>([first = first_offset] __device__(int64_t o) {
        return o - first;
      }));
    detail::segmented_sum(
      d_counts.data(), str_cp_offsets.data(), num_rows, d_row_offsets, d_row_offsets + 1, stream);
    CUDF_CUDA_TRY(
      cudaMemsetAsync(str_cp_offsets.data() + num_rows, 0, sizeof(int64_t), stream.get()));
    auto const total_cps = cudf::detail::sizes_to_offsets(str_cp_offsets.data(),
                                                          str_cp_offsets.data() + num_rows + 1,
                                                          str_cp_offsets.data(),
                                                          int64_t{0},
                                                          stream,
                                                          temp_mr);

    // Fill the packed (cp|ccc|width) slots for each input byte
    auto cps = cuda::device_buffer<uint32_t>{
      stream, temp_mr, static_cast<std::size_t>(total_cps), cuda::no_init};
    detail::decompose_fill_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
      decomposer, d_counts.data(), d_block_offsets.data(), cps.data());
    CUDF_CUDA_TRY(cudaGetLastError());
    return cps;
  }();

  auto const d_cps = cuda::std::span<uint32_t>(cps.data(), cps.size());
  auto const d_scp = cuda::std::span<int64_t const>(str_cp_offsets.data(), str_cp_offsets.size());

  // Canonical Reorder + Composition:
  // For NFC/NFKC, fuse reorder and compose in one launch so each thread
  // composes its row immediately after reordering, while the data is still hot.
  // For NFD/NFKD, only the reorder step is needed.
  auto const row_iter = cuda::make_counting_iterator(cudf::size_type{0});
  if (p.form == unicode_normalization_form::NFC || p.form == unicode_normalization_form::NFKC) {
    auto fn = detail::reorder_and_compose_fn{
      detail::reorder_fn{d_cps, d_scp},
      detail::compose_fn{d_cps, d_scp, p.comp_keys, p.comp_values, p.compat_decomp_flags}};
    thrust::for_each_n(policy, row_iter, num_rows, fn);
  } else {
    thrust::for_each_n(policy, row_iter, num_rows, detail::reorder_fn{d_cps, d_scp});
  }

  // Output: the UTF-8 width of each slot is summed per row for the output offsets
  // and scanned to locate each slot's bytes in the output chars
  auto const d_widths = cuda::transform_iterator(cps.data(), detail::utf8_width_fn{});
  auto [offsets_column, total_bytes] = [&] {
    auto sizes = cuda::device_buffer<cudf::size_type>{
      stream, temp_mr, static_cast<std::size_t>(num_rows), cuda::no_init};
    detail::segmented_sum(
      d_widths, sizes.data(), num_rows, str_cp_offsets.data(), str_cp_offsets.data() + 1, stream);
    return cudf::strings::detail::make_offsets_child_column(
      sizes.data(), sizes.data() + sizes.size(), stream, mr);
  }();
  auto chars = rmm::device_uvector<char>(total_bytes, stream, mr);
  detail::scan_with(d_widths,
                    detail::write_utf8_fn{cps.data(), chars.data()},
                    static_cast<int64_t>(cps.size()),
                    stream);

  return cudf::make_strings_column(num_rows,
                                   std::move(offsets_column),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

std::unique_ptr<cudf::column> normalize_unicode(cudf::strings_column_view const& input,
                                                unicode_normalizer const& normalizer,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_unicode(input, normalizer, stream, mr);
}

}  // namespace nvtext
