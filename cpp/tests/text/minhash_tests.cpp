/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <nvtext/minhash.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <vector>

struct MinHashTest : public cudf::test::BaseFixture {};

TEST_F(MinHashTest, Basic)
{
  auto validity = cudf::test::iterators::null_at(1);
  auto input =
    cudf::test::strings_column_wrapper({"doc 1",
                                        "",
                                        "this is doc 2",
                                        "",
                                        "doc 3",
                                        "d",
                                        "The quick brown fox jumpéd over the lazy brown dog.",
                                        "line eight",
                                        "line nine",
                                        "line ten"},
                                       validity);

  auto view = cudf::strings_column_view(input);

  auto results = nvtext::minhash(view);

  auto expected = cudf::test::fixed_width_column_wrapper<uint32_t>({1207251914u,
                                                                    0u,
                                                                    21141582u,
                                                                    0u,
                                                                    1207251914u,
                                                                    655955059u,
                                                                    86520422u,
                                                                    304329233u,
                                                                    640477688u,
                                                                    640477688u},
                                                                   validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto results64  = nvtext::minhash64(view);
  auto expected64 = cudf::test::fixed_width_column_wrapper<uint64_t>({774489391575805754ul,
                                                                      0ul,
                                                                      3232308021562742685ul,
                                                                      0ul,
                                                                      13145552576991307582ul,
                                                                      14660046701545912182ul,
                                                                      398062025280761388ul,
                                                                      1273320923074904938ul,
                                                                      3456065052701055601ul,
                                                                      10664519708968191209ul},
                                                                     validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, LengthEqualsWidth)
{
  auto input   = cudf::test::strings_column_wrapper({"abcdé", "fghjk", "lmnop", "qrstu", "vwxyz"});
  auto view    = cudf::strings_column_view(input);
  auto results = nvtext::minhash(view, 0, 5);
  auto expected = cudf::test::fixed_width_column_wrapper<uint32_t>(
    {3825281041u, 2728681928u, 1984332911u, 3965004915u, 192452857u});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(MinHashTest, MultiSeed)
{
  auto input =
    cudf::test::strings_column_wrapper({"doc 1",
                                        "this is doc 2",
                                        "doc 3",
                                        "d",
                                        "The quick brown fox jumpéd over the lazy brown dog.",
                                        "line six",
                                        "line seven",
                                        "line eight",
                                        "line nine",
                                        "line ten"});

  auto view = cudf::strings_column_view(input);

  auto seeds   = cudf::test::fixed_width_column_wrapper<uint32_t>({0, 1, 2});
  auto results = nvtext::minhash(view, cudf::column_view(seeds));

  using LCW = cudf::test::lists_column_wrapper<uint32_t>;
  // clang-format off
  LCW expected({LCW{1207251914u, 1677652962u, 1061355987u},
                LCW{  21141582u,  580916568u, 1258052021u},
                LCW{1207251914u,  943567174u, 1109272887u},
                LCW{ 655955059u,  488346356u, 2394664816u},
                LCW{  86520422u,  236622901u,  102546228u},
                LCW{ 640477688u,  198451716u,  136303992u},
                LCW{ 640477688u,  198451716u,  577802054u},
                LCW{ 304329233u,  198451716u,  714941560u},
                LCW{ 640477688u,  198451716u,  261342259u},
                LCW{ 640477688u,  198451716u,  139988887u}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto seeds64   = cudf::test::fixed_width_column_wrapper<uint64_t>({0, 1, 2});
  auto results64 = nvtext::minhash64(view, cudf::column_view(seeds64));

  using LCW64 = cudf::test::lists_column_wrapper<uint64_t>;
  // clang-format off
  LCW64 expected64({LCW64{  774489391575805754ul, 10435654231793485448ul,  1188598072697676120ul},
                    LCW64{ 3232308021562742685ul,  4445611509348165860ul,  1188598072697676120ul},
                    LCW64{13145552576991307582ul,  6846192680998069919ul,  1188598072697676120ul},
                    LCW64{14660046701545912182ul, 17106501326045553694ul, 17713478494106035784ul},
                    LCW64{  398062025280761388ul,   377720198157450084ul,   984941365662009329ul},
                    LCW64{ 2837259098848821044ul,   650799815433771163ul,  2428991957842356245ul},
                    LCW64{ 2105419906076957667ul,   650799815433771163ul,  2428991957842356245ul},
                    LCW64{ 1273320923074904938ul,   650799815433771163ul,  2428991957842356245ul},
                    LCW64{ 3456065052701055601ul,   650799815433771163ul,  2428991957842356245ul},
                    LCW64{10664519708968191209ul,   650799815433771163ul,  2428991957842356245ul}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, MultiSeedWithNullInputRow)
{
  auto validity = cudf::test::iterators::null_at(1);
  auto input    = cudf::test::strings_column_wrapper({"abcdéfgh", "", "", "stuvwxyz"}, validity);
  auto view     = cudf::strings_column_view(input);

  auto seeds   = cudf::test::fixed_width_column_wrapper<uint32_t>({1, 2});
  auto results = nvtext::minhash(view, cudf::column_view(seeds));

  using LCW = cudf::test::lists_column_wrapper<uint32_t>;
  LCW expected({LCW{484984072u, 1074168784u}, LCW{}, LCW{0u, 0u}, LCW{571652169u, 173528385u}},
               validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto seeds64   = cudf::test::fixed_width_column_wrapper<uint64_t>({11, 22});
  auto results64 = nvtext::minhash64(view, cudf::column_view(seeds64));

  using LCW64 = cudf::test::lists_column_wrapper<uint64_t>;
  LCW64 expected64({LCW64{2597399324547032480ul, 4461410998582111052ul},
                    LCW64{},
                    LCW64{0ul, 0ul},
                    LCW64{2717781266371273264ul, 6977325820868387259ul}},
                   validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, Permuted)
{
  auto input =
    cudf::test::strings_column_wrapper({"doc 1",
                                        "this is doc 2",
                                        "doc 3",
                                        "d",
                                        "The quick brown fox jumpéd over the lazy brown dog.",
                                        "line six",
                                        "line seven",
                                        "line eight",
                                        "line nine",
                                        "line ten"});

  auto view = cudf::strings_column_view(input);

  auto first  = thrust::counting_iterator<uint32_t>(10);
  auto params = cudf::test::fixed_width_column_wrapper<uint32_t>(first, first + 3);
  auto results =
    nvtext::minhash_permuted(view, 0, cudf::column_view(params), cudf::column_view(params), 4);

  using LCW32 = cudf::test::lists_column_wrapper<uint32_t>;
  // clang-format off
  LCW32 expected({
    LCW32{1392101586u,  394869177u,  811528444u},
    LCW32{ 211415830u,  187088503u,  130291444u},
    LCW32{2098117052u,  394869177u,  799753544u},
    LCW32{2264583304u, 2920538364u, 3576493424u},
    LCW32{ 253327882u,   41747273u,  302030804u},
    LCW32{2109809594u, 1017470651u,  326988172u},
    LCW32{1303819864u,  850676747u,  147107852u},
    LCW32{ 736021564u,  720812292u, 1405158760u},
    LCW32{ 902780242u,  134064807u, 1613944636u},
    LCW32{ 547084870u, 1748895564u,  656501844u}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto params64  = cudf::test::fixed_width_column_wrapper<uint64_t, uint32_t>(first, first + 3);
  auto results64 = nvtext::minhash64_permuted(
    view, 0, cudf::column_view(params64), cudf::column_view(params64), 4);

  using LCW64 = cudf::test::lists_column_wrapper<uint64_t>;
  // clang-format off
  LCW64 expected64({
    LCW64{ 827364888116975697ul, 1601854279692781452ul,  70500662054893256ul},
    LCW64{  18312093741021833ul,  133793446674258329ul,  21974512489226198ul},
    LCW64{  22474244732520567ul, 1638811775655358395ul, 949306297364502264ul},
    LCW64{1332357434996402861ul, 2157346081260151330ul, 676491718310205848ul},
    LCW64{  65816830624808020ul,   43323600380520789ul,  63511816333816345ul},
    LCW64{ 629657184954525200ul,   49741036507643002ul,  97466271004074331ul},
    LCW64{ 301611977846331113ul,  101188874709594830ul,  97466271004074331ul},
    LCW64{ 121498891461700668ul,  171065800427907402ul,  97466271004074331ul},
    LCW64{  54617739511834072ul,  231454301607238929ul,  97466271004074331ul},
    LCW64{ 576418665851990314ul,  231454301607238929ul,  97466271004074331ul}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, PermutedWide)
{
  std::string const small(2 << 10, 'x');  // below wide_string_threshold
  std::string const wide(2 << 19, 'y');   // above wide_string_threshold
  auto input = cudf::test::strings_column_wrapper({small, wide});
  auto view  = cudf::strings_column_view(input);

  auto first  = thrust::counting_iterator<uint32_t>(20);
  auto params = cudf::test::fixed_width_column_wrapper<uint32_t>(first, first + 3);
  auto results =
    nvtext::minhash_permuted(view, 0, cudf::column_view(params), cudf::column_view(params), 4);

  using LCW32 = cudf::test::lists_column_wrapper<uint32_t>;
  // clang-format off
  LCW32 expected({
    LCW32{1731998032u,  315359380u, 3193688024u},
    LCW32{1293098788u, 2860992281u,  133918478u}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto params64  = cudf::test::fixed_width_column_wrapper<uint64_t, uint32_t>(first, first + 3);
  auto results64 = nvtext::minhash64_permuted(
    view, 0, cudf::column_view(params64), cudf::column_view(params64), 4);

  using LCW64 = cudf::test::lists_column_wrapper<uint64_t>;
  // clang-format off
   LCW64 expected64({
     LCW64{1818322427062143853ul, 641024893347719371ul, 1769570368846988848ul},
     LCW64{1389920339306667795ul, 421787002125838902ul, 1759496674158703968ul}
   });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, WordsMinHash)
{
  using LCWS    = cudf::test::lists_column_wrapper<cudf::string_view>;
  auto validity = cudf::test::iterators::null_at(1);

  LCWS input(
    {LCWS({"hello", "abcdéfgh"}),
     LCWS{},
     LCWS({"rapids", "moré", "test", "text"}),
     LCWS({"The", "quick", "brown", "fox", "jumpéd", "over", "the", "lazy", "brown", "dog"})},
    validity);

  auto view = cudf::lists_column_view(input);

  auto seeds   = cudf::test::fixed_width_column_wrapper<uint32_t>({1, 2});
  auto results = nvtext::word_minhash(view, cudf::column_view(seeds));
  using LCW32  = cudf::test::lists_column_wrapper<uint32_t>;
  LCW32 expected({LCW32{2069617641u, 1975382903u},
                  LCW32{},
                  LCW32{657297235u, 1010955999u},
                  LCW32{644643885u, 310002789u}},
                 validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto seeds64   = cudf::test::fixed_width_column_wrapper<uint64_t>({11, 22});
  auto results64 = nvtext::word_minhash64(view, cudf::column_view(seeds64));
  using LCW64    = cudf::test::lists_column_wrapper<uint64_t>;
  LCW64 expected64({LCW64{1940333969930105370ul, 272615362982418219ul},
                    LCW64{},
                    LCW64{5331949571924938590ul, 2088583894581919741ul},
                    LCW64{3400468157617183341ul, 2398577492366130055ul}},
                   validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, EmptyTest)
{
  auto input   = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  auto view    = cudf::strings_column_view(input->view());
  auto results = nvtext::minhash(view);
  EXPECT_EQ(results->size(), 0);
  results = nvtext::minhash64(view);
  EXPECT_EQ(results->size(), 0);
}

TEST_F(MinHashTest, ErrorsTest)
{
  auto input = cudf::test::strings_column_wrapper({"this string intentionally left blank"});
  auto view  = cudf::strings_column_view(input);
  EXPECT_THROW(nvtext::minhash(view, 0, 0), std::invalid_argument);
  EXPECT_THROW(nvtext::minhash64(view, 0, 0), std::invalid_argument);
  auto seeds = cudf::test::fixed_width_column_wrapper<uint32_t>();
  EXPECT_THROW(nvtext::minhash(view, cudf::column_view(seeds)), std::invalid_argument);
  auto seeds64 = cudf::test::fixed_width_column_wrapper<uint64_t>();
  EXPECT_THROW(nvtext::minhash64(view, cudf::column_view(seeds64)), std::invalid_argument);

  std::vector<std::string> h_input(50000, "");
  input = cudf::test::strings_column_wrapper(h_input.begin(), h_input.end());
  view  = cudf::strings_column_view(input);

  auto const zeroes = thrust::constant_iterator<uint32_t>(0);
  seeds             = cudf::test::fixed_width_column_wrapper<uint32_t>(zeroes, zeroes + 50000);
  EXPECT_THROW(nvtext::minhash(view, cudf::column_view(seeds)), std::overflow_error);
  seeds64 = cudf::test::fixed_width_column_wrapper<uint64_t>(zeroes, zeroes + 50000);
  EXPECT_THROW(nvtext::minhash64(view, cudf::column_view(seeds64)), std::overflow_error);
}
