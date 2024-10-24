/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace cudf {
/**
 * @addtogroup column_reshape
 * @{
 * @file
 * @brief Column APIs for explore list columns
 */

/**
 * @brief Explodes a list column's elements.
 *
 * Any list is exploded, which means the elements of the list in each row are expanded into new rows
 * in the output. The corresponding rows for other columns in the input are duplicated. Example:
 * ```
 * [[5,10,15], 100],
 * [[20,25],   200],
 * [[30],      300],
 * returns
 * [5,         100],
 * [10,        100],
 * [15,        100],
 * [20,        200],
 * [25,        200],
 * [30,        300],
 * ```
 *
 * Nulls and empty lists propagate in different ways depending on what is null or empty.
 *```
 * [[5,null,15], 100],
 * [null,        200],
 * [[],          300],
 * returns
 * [5,           100],
 * [null,        100],
 * [15,          100],
 * ```
 * Note that null lists are not included in the resulting table, but nulls inside
 * lists and empty lists will be represented with a null entry for that column in that row.
 *
 * @param input_table Table to explode.
 * @param explode_column_idx Column index to explode inside the table.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 *
 * @return A new table with explode_col exploded.
 */
std::unique_ptr<table> explode(
  table_view const& input_table,
  size_type explode_column_idx,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Explodes a list column's elements and includes a position column.
 *
 * Any list is exploded, which means the elements of the list in each row are expanded into new rows
 * in the output. The corresponding rows for other columns in the input are duplicated. A position
 * column is added that has the index inside the original list for each row. Example:
 * ```
 * [[5,10,15], 100],
 * [[20,25],   200],
 * [[30],      300],
 * returns
 * [0,   5,     100],
 * [1,   10,    100],
 * [2,   15,    100],
 * [0,   20,    200],
 * [1,   25,    200],
 * [0,   30,    300],
 * ```
 *
 * Nulls and empty lists propagate in different ways depending on what is null or empty.
 *```
 * [[5,null,15], 100],
 * [null,        200],
 * [[],          300],
 * returns
 * [0,     5,    100],
 * [1,  null,    100],
 * [2,    15,    100],
 * ```
 * Note that null lists are not included in the resulting table, but nulls inside
 * lists and empty lists will be represented with a null entry for that column in that row.
 *
 * @param input_table Table to explode.
 * @param explode_column_idx Column index to explode inside the table.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 *
 * @return A new table with exploded value and position. The column order of return table is
 *         [cols before explode_input, explode_position, explode_value, cols after explode_input].
 */
std::unique_ptr<table> explode_position(
  table_view const& input_table,
  size_type explode_column_idx,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Explodes a list column's elements retaining any null entries or empty lists inside.
 *
 * Any list is exploded, which means the elements of the list in each row are expanded into new rows
 * in the output. The corresponding rows for other columns in the input are duplicated. Example:
 * ```
 * [[5,10,15], 100],
 * [[20,25],   200],
 * [[30],      300],
 * returns
 * [5,         100],
 * [10,        100],
 * [15,        100],
 * [20,        200],
 * [25,        200],
 * [30,        300],
 * ```
 *
 * Nulls and empty lists propagate as null entries in the result.
 *```
 * [[5,null,15], 100],
 * [null,        200],
 * [[],          300],
 * returns
 * [5,           100],
 * [null,        100],
 * [15,          100],
 * [null,        200],
 * [null,        300],
 * ```
 *
 * @param input_table Table to explode.
 * @param explode_column_idx Column index to explode inside the table.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 *
 * @return A new table with explode_col exploded.
 */
std::unique_ptr<table> explode_outer(
  table_view const& input_table,
  size_type explode_column_idx,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Explodes a list column's elements retaining any null entries or empty lists and includes a
 *position column.
 *
 * Any list is exploded, which means the elements of the list in each row are expanded into new rows
 * in the output. The corresponding rows for other columns in the input are duplicated. A position
 * column is added that has the index inside the original list for each row. Example:
 * ```
 * [[5,10,15], 100],
 * [[20,25],   200],
 * [[30],      300],
 * returns
 * [0,   5,    100],
 * [1,  10,    100],
 * [2,  15,    100],
 * [0,  20,    200],
 * [1,  25,    200],
 * [0,  30,    300],
 * ```
 *
 * Nulls and empty lists propagate as null entries in the result.
 *```
 * [[5,null,15], 100],
 * [null,        200],
 * [[],          300],
 * returns
 * [0,     5,    100],
 * [1,  null,    100],
 * [2,    15,    100],
 * [0,  null,    200],
 * [0,  null,    300],
 * ```
 *
 * @param input_table Table to explode.
 * @param explode_column_idx Column index to explode inside the table.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 *
 * @return A new table with explode_col exploded.
 */
std::unique_ptr<table> explode_outer_position(
  table_view const& input_table,
  size_type explode_column_idx,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group

}  // namespace cudf
