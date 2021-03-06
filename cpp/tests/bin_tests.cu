/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/bin.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <numeric>
#include <thrust/device_ptr.h>

using namespace cudf::test;

namespace {

// =============================================================================
// ----- tests -----------------------------------------------------------------

TEST(BinColumnTest, TestInvalidLeft)
{
  fixed_width_column_wrapper<double> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fixed_width_column_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  fixed_width_column_wrapper<float> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  EXPECT_THROW(cudf::bin::bin(input, left_edges, cudf::bin::inclusive::YES, right_edges, cudf::bin::inclusive::NO),
          cudf::logic_error);
};

TEST(BinColumnTest, TestInvalidRight)
{
  fixed_width_column_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fixed_width_column_wrapper<double> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  fixed_width_column_wrapper<float> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  EXPECT_THROW(cudf::bin::bin(input, left_edges, cudf::bin::inclusive::YES, right_edges, cudf::bin::inclusive::NO),
          cudf::logic_error);
};

TEST(BinColumnTest, TestInvalidInput)
{
  fixed_width_column_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fixed_width_column_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  fixed_width_column_wrapper<double> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  EXPECT_THROW(cudf::bin::bin(input, left_edges, cudf::bin::inclusive::YES, right_edges, cudf::bin::inclusive::NO),
          cudf::logic_error);
};

TEST(BinColumnTest, TestMismatchedEdges)
{
  fixed_width_column_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fixed_width_column_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9};
  fixed_width_column_wrapper<float> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  EXPECT_THROW(cudf::bin::bin(input, left_edges, cudf::bin::inclusive::YES, right_edges, cudf::bin::inclusive::NO),
          cudf::logic_error);
};

TEST(BinColumnTest, TestSimple)
{
  fixed_width_column_wrapper<float> left_edges{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  fixed_width_column_wrapper<float> right_edges{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  fixed_width_column_wrapper<float> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  auto result = cudf::bin::bin(input, left_edges, cudf::bin::inclusive::YES, right_edges, cudf::bin::inclusive::NO);
  // Check that the total number of elements binned is the number of input elements.
  auto begin = thrust::device_ptr<const unsigned int>(result->view().begin<unsigned int>());
  auto end = thrust::device_ptr<const unsigned int>(result->view().end<unsigned int>());
  ASSERT_EQ(thrust::reduce(begin, end), static_cast<cudf::column_view>(input).size());
  // Check that the first bin contains all the elements (true by construction).
  ASSERT_EQ(*begin, static_cast<cudf::column_view>(input).size());
};

}  // anonymous namespace

CUDF_TEST_PROGRAM_MAIN()