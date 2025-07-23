/**
 * AnalyticalTest:
 *
 * Tests for Origami functions
 */

#include <Tensile/analytical/AnalyticalGemm.hpp>
#include <Tensile/analytical/Hardware.hpp>
#include <Tensile/analytical/Utils.hpp>
#include <gtest/gtest.h>

// AnalyticalGemm
TEST(Analytical, ComputeNumMatrixInstructions)
{
    auto mt128x128x64 = TensileLite::analytical::compute_number_matrix_instructions(
        128, 128, 64, 16, 16, 16, false);
    auto mt16x16x64 = TensileLite::analytical::compute_number_matrix_instructions(
        16, 16, 64, 16, 16, 32, false);
    EXPECT_EQ(mt128x128x64, 256); // 8 * 8 * 4
    EXPECT_EQ(mt16x16x64, 2); // 1 * 1 * 2
}

TEST(Analytical, ComputeMTComputeLatency)
{
    auto gfx942arch = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx942
        = TensileLite::analytical::Hardware(gfx942arch, 1, 1, 1, 1.0, 1.0, 1.0, 1, 1.0, 1, 1.0);
    // 4 * 4 * 8 * 32
    auto tn_latency = TensileLite::analytical::compute_mt_compute_latency(
        gfx942, true, false, 128, 128, 64, 32, 32, 8, 16, 16, false);
    auto nt_latency = TensileLite::analytical::compute_mt_compute_latency(
        gfx942, false, true, 128, 128, 64, 32, 32, 8, 16, 16, false);
    auto nn_latency = TensileLite::analytical::compute_mt_compute_latency(
        gfx942, false, false, 128, 128, 64, 32, 32, 8, 16, 16, false);
    auto tt_latency = TensileLite::analytical::compute_mt_compute_latency(
        gfx942, true, true, 128, 128, 64, 32, 32, 8, 16, 16, false);
    EXPECT_EQ(tn_latency, 4096);
    EXPECT_EQ(nt_latency, tn_latency);
    EXPECT_EQ(nn_latency, nn_latency);
    EXPECT_EQ(tt_latency, tt_latency);

    // Check penalty conditions
    // TN K indivisible by 128 bytes
    // 4 * 4 * 4 * 32
    tn_latency = TensileLite::analytical::compute_mt_compute_latency(
        gfx942, true, false, 128, 128, 32, 32, 32, 8, 16, 16, false);
    EXPECT_GT(tn_latency, 2048);

    // NT A is 128 bytes contiguous in M and B is 128 bytes contiguous in N
    // 7 * 7 * 8 * 32
    nt_latency = TensileLite::analytical::compute_mt_compute_latency(
        gfx942, false, true, 224, 224, 64, 32, 32, 8, 16, 16, false);
    EXPECT_GT(nt_latency, 12544);

    // TT A is >= 128 bytes contiguous in K and B is >= 128 bytes contiguous in N
    // 4 * 1 * 4 * 32
    tt_latency = TensileLite::analytical::compute_mt_compute_latency(
        gfx942, true, true, 128, 32, 32, 32, 32, 8, 16, 16, false);
    EXPECT_GT(tt_latency, 512);

    // NN A is >= 128 bytes contiguous in M and B is >= 128 bytes contiguous in K
    // 1 * 4 * 4 * 32
    nn_latency = TensileLite::analytical::compute_mt_compute_latency(
        gfx942, false, false, 32, 128, 32, 32, 32, 8, 16, 16, false);
    EXPECT_GT(nn_latency, 512);
}

TEST(Analytical, ComputeNumberWaves)
{
    auto gfx942arch = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx942
        = TensileLite::analytical::Hardware(gfx942arch, 304, 1, 1, 1.0, 1.0, 1.0, 1, 1.0, 1, 1.0);
    auto num_waves
        = TensileLite::analytical::compute_number_waves(gfx942, 4096, 1024, 3, 256, 256, false);
    // 16 * 4 * 3 == 192 < 304
    EXPECT_EQ(num_waves, 1);
    // 16 * 32 * 3 == 1536 > 5*304
    num_waves
        = TensileLite::analytical::compute_number_waves(gfx942, 4096, 8192, 3, 256, 256, false);
    EXPECT_EQ(num_waves, 6);
}

TEST(Analytical, ComputeLoads)
{
    auto a_loads  = TensileLite::analytical::compute_A_loads(128, 64, false);
    auto b_loads  = TensileLite::analytical::compute_B_loads(128, 64, false);
    auto cu_loads = TensileLite::analytical::compute_CU_loads(128, 128, 64, false);
    EXPECT_EQ(a_loads, 8192);
    EXPECT_EQ(b_loads, 8192);
    EXPECT_EQ(cu_loads, a_loads + b_loads);
}

TEST(Analytical, ComputeActiveCU)
{
    auto gfx942arch = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx942
        = TensileLite::analytical::Hardware(gfx942arch, 304, 1, 1, 1.0, 1.0, 1.0, 1, 1.0, 1, 1.0);
    auto active_cu = TensileLite::analytical::compute_active_CU(gfx942, 4096, 1024, 3, 256, 256);
    EXPECT_EQ(active_cu, 192);

    active_cu = TensileLite::analytical::compute_active_CU(gfx942, 4096, 8192, 3, 256, 256);
    EXPECT_EQ(active_cu, gfx942.N_CU);
}

TEST(Analytical, ComputeMemoryLatency)
{
    auto gfx942arch = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx942     = TensileLite::analytical::Hardware(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.0, 1, 1.0);
    double hmem1             = 0.9; // L2 hit
    auto   mem_latency_large = TensileLite::analytical::compute_memory_latency(
        gfx942, 4096, 4096, 1024, true, false, 1, 256, 256, 64, 1, hmem1, 16, 16, 0, false);
    auto mem_latency_small = TensileLite::analytical::compute_memory_latency(
        gfx942, 4096, 4096, 1024, true, false, 1, 128, 128, 64, 1, hmem1, 16, 16, 0, false);
    EXPECT_LT(mem_latency_small, mem_latency_large);
}

TEST(Analytical, ComputeTileLatency)
{
    auto gfx942arch = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx942     = TensileLite::analytical::Hardware(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.0, 1, 1.0);
    double hmem1              = 0.9; // L2 hit
    auto   tile_latency_large = TensileLite::analytical::compute_tile_latency(gfx942,
                                                                            4096,
                                                                            4096,
                                                                            1024,
                                                                            2,
                                                                            true,
                                                                            false,
                                                                            256,
                                                                            256,
                                                                            64,
                                                                            32,
                                                                            32,
                                                                            8,
                                                                            1,
                                                                            hmem1,
                                                                            16,
                                                                            16,
                                                                            32,
                                                                            0,
                                                                            false);
    auto   tile_latency_small = TensileLite::analytical::compute_tile_latency(gfx942,
                                                                            4096,
                                                                            4096,
                                                                            1024,
                                                                            2,
                                                                            true,
                                                                            false,
                                                                            128,
                                                                            128,
                                                                            64,
                                                                            32,
                                                                            32,
                                                                            8,
                                                                            1,
                                                                            hmem1,
                                                                            16,
                                                                            16,
                                                                            32,
                                                                            0,
                                                                            false);
    EXPECT_GT(tile_latency_large, tile_latency_small);
}

TEST(Analytical, ComputeWaveLatency)
{
    auto gfx942arch = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx942     = TensileLite::analytical::Hardware(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.0, 1, 1.0);
    double hmem1        = 0.9; // L2 hit
    auto   tile_latency = TensileLite::analytical::compute_tile_latency(gfx942,
                                                                      4096,
                                                                      4096,
                                                                      1024,
                                                                      2,
                                                                      true,
                                                                      false,
                                                                      256,
                                                                      256,
                                                                      64,
                                                                      32,
                                                                      32,
                                                                      8,
                                                                      1,
                                                                      hmem1,
                                                                      16,
                                                                      16,
                                                                      32,
                                                                      0,
                                                                      false);
    auto   wave_latency = TensileLite::analytical::compute_wave_latency(gfx942,
                                                                      4096,
                                                                      4096,
                                                                      1024,
                                                                      2,
                                                                      true,
                                                                      false,
                                                                      256,
                                                                      256,
                                                                      64,
                                                                      32,
                                                                      32,
                                                                      8,
                                                                      1,
                                                                      hmem1,
                                                                      16,
                                                                      16,
                                                                      32,
                                                                      0,
                                                                      false);
    EXPECT_DOUBLE_EQ(wave_latency, tile_latency);
}

TEST(Analytical, ComputeTotalLatency)
{
    auto gfx942arch = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx942     = TensileLite::analytical::Hardware(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.0, 1, 1.0);

    double latency_cycles_large = TensileLite::analytical::compute_total_latency(gfx942,
                                                                                 4096,
                                                                                 4096,
                                                                                 1024,
                                                                                 2,
                                                                                 true,
                                                                                 false,
                                                                                 256,
                                                                                 256,
                                                                                 64,
                                                                                 32,
                                                                                 32,
                                                                                 8,
                                                                                 1,
                                                                                 0.0,
                                                                                 16,
                                                                                 16,
                                                                                 32,
                                                                                 1,
                                                                                 0,
                                                                                 false);

    double latency_cycles_small = TensileLite::analytical::compute_total_latency(gfx942,
                                                                                 4096,
                                                                                 4096,
                                                                                 1024,
                                                                                 2,
                                                                                 true,
                                                                                 false,
                                                                                 128,
                                                                                 128,
                                                                                 64,
                                                                                 32,
                                                                                 32,
                                                                                 8,
                                                                                 1,
                                                                                 0.0,
                                                                                 16,
                                                                                 16,
                                                                                 32,
                                                                                 1,
                                                                                 0,
                                                                                 false);
    EXPECT_LT(latency_cycles_large, latency_cycles_small);
}

TEST(Analytical, ComputePerfGflops)
{
    auto gfx942arch  = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx942_slow = TensileLite::analytical::Hardware(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.4, 1, 1.0);
    auto gfx942_fast = TensileLite::analytical::Hardware(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.8, 1, 1.0);
    double flops_slow = TensileLite::analytical::compute_perf_gflops(gfx942_slow,
                                                                     4096,
                                                                     4096,
                                                                     1024,
                                                                     2,
                                                                     true,
                                                                     false,
                                                                     256,
                                                                     256,
                                                                     64,
                                                                     32,
                                                                     32,
                                                                     8,
                                                                     16,
                                                                     16,
                                                                     32,
                                                                     1,
                                                                     0.9,
                                                                     false);
    double flops_fast = TensileLite::analytical::compute_perf_gflops(gfx942_fast,
                                                                     4096,
                                                                     4096,
                                                                     1024,
                                                                     2,
                                                                     true,
                                                                     false,
                                                                     256,
                                                                     256,
                                                                     64,
                                                                     32,
                                                                     32,
                                                                     8,
                                                                     16,
                                                                     16,
                                                                     32,
                                                                     1,
                                                                     0.9,
                                                                     false);
    EXPECT_GT(flops_fast, flops_slow); // faster clock = higher flops
}

TEST(Analytical, EstimateL2Hit)
{
    auto gfx942arch = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx942     = TensileLite::analytical::Hardware(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.0, 1, 1.0);
    double l2_hit;
    for(int i = 1; i < 1025; i++)
    {
        l2_hit = TensileLite::analytical::estimate_l2_hit(
            gfx942, 4096, 4096, 1024, 1, 256, 256, 64, i, 16);
        EXPECT_GT(l2_hit, 0.0);
        EXPECT_LT(l2_hit, 1.0);
    }
}

TEST(Analytical, EstimateMallHit)
{
    auto gfx942arch = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx942     = TensileLite::analytical::Hardware(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 1, 1.0, 1, 1.0);
    double mall_hit;
    for(int i = 1; i < 1025; i++)
    {
        mall_hit = TensileLite::analytical::estimate_mall_hit(
            gfx942, 4096, 4096, 1024, 1, 256, 256, 64, i);
        EXPECT_GT(mall_hit, 0.0);
    }
}

TEST(Analytical, CheckLDSCapacity)
{
    auto gfx942arch = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx942     = TensileLite::analytical::Hardware(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 1, 1.0, 1, 1.0);
    auto fit_lds_memory
        = TensileLite::analytical::check_LDS_capacity(gfx942, 256, 256, 64, 16, false);
    EXPECT_TRUE(fit_lds_memory);
}

// Hardware
TEST(Analytical, HardwareArchEnum)
{
    auto gfx942 = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx950 = TensileLite::analytical::Hardware::archNameToEnum("gfx950");
    auto gfx90a = TensileLite::analytical::Hardware::archNameToEnum("gfx90a");

    EXPECT_EQ(gfx942, TensileLite::analytical::Hardware::Architecture::gfx942);
    EXPECT_EQ(gfx950, TensileLite::analytical::Hardware::Architecture::gfx950);
    EXPECT_EQ(gfx90a, TensileLite::analytical::Hardware::Architecture::Count);
}

// Utils
TEST(Analytical, BestGridSize)
{
    auto gfx942arch = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx942     = TensileLite::analytical::Hardware(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.0, 1, 1.0);

    size_t grid_size = TensileLite::analytical::select_best_grid_size(1024,
                                                                      1024,
                                                                      4096,
                                                                      1,
                                                                      true,
                                                                      false,
                                                                      gfx942,
                                                                      256,
                                                                      256,
                                                                      64,
                                                                      32,
                                                                      32,
                                                                      8,
                                                                      16,
                                                                      16,
                                                                      32,
                                                                      0,
                                                                      0.0,
                                                                      false,
                                                                      1,
                                                                      20);
    EXPECT_GT(grid_size, 16);
}

TEST(Analytical, BestMacroTileSize)
{
    auto gfx942arch = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx942     = TensileLite::analytical::Hardware(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.0, 4, 1.0);
    std::vector<std::tuple<size_t, // MT_M
                           size_t, // MT_N
                           size_t, // MT_K
                           size_t, // MI_M
                           size_t, // MI_N
                           size_t // MI_K
                           >>
         MT_list = {{256, 256, 32, 32, 32, 8}, {128, 128, 64, 32, 32, 8}, {64, 64, 64, 32, 32, 8}};
    auto results = select_best_macro_tile_size(
        4096, 4096, 8192, 1, true, false, gfx942, MT_list, 16, 16, 32, 0, 0.8, false, false, 1);

    EXPECT_EQ(results.size(), MT_list.size());
    for(int i = 0; i < results.size() - 1; i++)
        EXPECT_LT(std::get<0>(results[i]), std::get<0>(results[i + 1]));
}

TEST(Analytical, BestWGM)
{
    auto gfx942arch = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx942     = TensileLite::analytical::Hardware(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.0, 1, 1.0);
    std::vector<size_t> WGM_list            = {1, 2, 4, 6, 8, 12};
    auto                best_wgm_large_tile = select_best_wgm(
        4096, 4096, 8192, 1, gfx942, 256, 256, 32, 32, 32, 8, WGM_list, 16, 0.0, false, false);

    auto best_wgm_small_tile = select_best_wgm(
        4096, 4096, 8192, 1, gfx942, 128, 128, 64, 32, 32, 8, WGM_list, 16, 0.0, false, false);

    auto best_wgm_nonsquare = select_best_wgm(
        2048, 5120, 8192, 1, gfx942, 256, 256, 32, 32, 32, 8, WGM_list, 16, 0.0, false, false);
    EXPECT_EQ(best_wgm_large_tile.second, best_wgm_small_tile.second);
    EXPECT_NE(best_wgm_large_tile.second, best_wgm_nonsquare.second);
}

TEST(Analytical, UtilsTFlopsFromLatency)
{
    double latency_cycles = 2.0;
    size_t M              = 10;
    size_t N              = 10;
    size_t K              = 10;
    double clock_GHz      = 1;

    auto tflops = TensileLite::analytical::compute_TFLOPS_from_latency(
        latency_cycles, M, N, K, clock_GHz, false);
    EXPECT_DOUBLE_EQ(tflops, 1.0);
}