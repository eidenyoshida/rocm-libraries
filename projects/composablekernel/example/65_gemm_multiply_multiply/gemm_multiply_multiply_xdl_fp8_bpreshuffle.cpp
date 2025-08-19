// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_b_preshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using FP8  = ck::f8_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = FP8;
using B0DataType       = FP8;
using AccDataType      = F32;
using CShuffleDataType = F32;
using D0DataType       = F32;
using D1DataType       = F32;
using DsDataType       = ck::Tuple<D0DataType, D1DataType>;
using EDataType        = F16;

using A0Layout = Row;
using B0Layout = Col;
using D0Layout = Row;
using D1Layout = Col;
using DsLayout = ck::Tuple<D0Layout, D1Layout>;
using ELayout  = Row;

struct MultiplyMultiply
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<F16, float, float, float>(F16& e,
                                                                            const float& c,
                                                                            const float& d0,
                                                                            const float& d1) const
    {
        const float x0_f = c * d0 * d1;

        e = ck::type_convert<F16>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<BF16, float, float, float>(BF16& e,
                                                                             const float& c,
                                                                             const float& d0,
                                                                             const float& d1) const
    {
        const float x0_f = c * d0 * d1;

        e = ck::type_convert<BF16>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, int, float, float>(
        ck::half_t& e, const int& c, const float& d0, const float& d1) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<float>(d0) * ck::type_convert<float>(d1);

        e = ck::type_convert<ck::half_t>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::bhalf_t, int, float, float>(
        ck::bhalf_t& e, const int& c, const float& d0, const float& d1) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<float>(d0) * ck::type_convert<float>(d1);

        e = ck::type_convert<ck::bhalf_t>(x0_f);
    }
};

void preShuffleBuffer(const FP8* src, FP8* dst, int N, int K, int NXdl)
{
    int KPack = 16;
    int NLane = NXdl;
    int KLane = 64 / NLane;

    int K0 = K / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[n * K + k];
        }
    }
}
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = MultiplyMultiply;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle
    // clang-format off
    // <Row, Col, DsLayout, ELayout, A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
    // AElementOp,  BElementOp, CDEElementOp, GemmSpec, 256,
    // 256,   256,    128,
    // 16,   16,
    // 16,   16,
    // 16,    4,
    // S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
    // S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
    // 2,    1,   S<1, 32, 1, 8>, S<8, 8, 1>,
    // ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, FP8>;

    <Row, Col, DsLayout, ELayout,
    A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
    AElementOp, BElementOp, CDEElementOp, GemmSpec,

    // 0.00851206
    // BlockSize
       256,
    // MPerBlock | NPerBlock | KPerBlock
       64,         64,         256,
    // AK1 | BK1
       16,   16,
    // MPerXDL | NPerXDL
       16,       16,
    // MXdlPerWave | NXdlPerWave
       4,            1,
    // ABlockTransferThreadClusterLengths_AK0_M_AK1 | ABlockTransferThreadClusterArrangeOrder | ABlockTransferSrcAccessOrder
       S<16, 16, 1>,                                  S<1, 0, 2>,                               S<1, 0, 2>,
    // ABlockTransferSrcVectorDim | ABlockTransferSrcScalarPerVector | ABlockTransferDstScalarPerVector_AK1 | ABlockLdsExtraM
       2,                           16,                                16,                                    0, 
    // BBlockTransferThreadClusterLengths_BK0_N_BK1 | BBlockTransferThreadClusterArrangeOrder | BBlockTransferSrcAccessOrder,
       S<16, 16, 1>,                                  S<1, 0, 2>,                               S<1, 0, 2>,
    // BBlockTransferSrcVectorDim | BBlockTransferSrcScalarPerVector | BBlockTransferDstScalarPerVector_BK1 | BBlockLdsExtraN
       2,                           16,                                16,                                    0,
    // CShuffleMXdlPerWavePerShuffle | CShuffleNXdlPerWavePerShuffle
       2,                              1,
    // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock | CDEShuffleBlockTransferScalarPerVectors
       S<1, 32, 1, 8>,                                                         S<8, 8, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3,

    // // from device_gemm_multiply_multiply_wp_xdl_f8_f8_f16_mk_mfma_mn.hpp
    // // 0.0131355
    // 256,    64,    128,   128,  16,  16,  32,   32,    2,    1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8, 8, 1>,  ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v2,
    // // 0.010373
    // 256,    32,    128,   128,  16,  16,  32,   32,    1,    1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8, 8, 1>,  ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1,
    FP8>;
// clang-format on

// expected launch with 1 block & 1 wave
template <typename DeviceOpInstance>
__global__ void test_entry([[maybe_unused]] typename DeviceOpInstance::Argument args)
{
    using namespace ck;
    using GridwiseGemm = typename DeviceOpInstance::GridwiseGemm;

    printf("shr mem usage: %d\n", GridwiseGemm::GetSharedMemoryNumberOfByte());

    // MakeAGridDescriptor_AK0_M_AK1
    [[maybe_unused]] static constexpr auto I0 = Number<0>{};
    [[maybe_unused]] static constexpr auto I1 = Number<1>{};
    [[maybe_unused]] static constexpr auto I2 = Number<2>{};
    [[maybe_unused]] static constexpr auto I3 = Number<3>{};
    [[maybe_unused]] static constexpr auto I4 = Number<4>{};
    [[maybe_unused]] static constexpr auto I5 = Number<5>{};
    [[maybe_unused]] static constexpr auto I6 = Number<6>{};
    [[maybe_unused]] static constexpr auto I7 = Number<7>{};

    [[maybe_unused]] auto M        = args.M;
    [[maybe_unused]] auto N        = args.N;
    [[maybe_unused]] auto K        = args.K;
    [[maybe_unused]] auto MPadded  = args.MPadded;
    [[maybe_unused]] auto KPadded  = args.KPadded;
    [[maybe_unused]] auto StrideA  = args.StrideA;
    [[maybe_unused]] auto AK0      = args.AK0;
    [[maybe_unused]] auto AK1Value = 16;

    auto a_grid_desc_mraw_kraw =
        make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));

    const auto a_grid_desc_ak0_m_ak1 =
        transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                    make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                                               make_pass_through_transform(M)),
                                    make_tuple(Sequence<1>{}, Sequence<0>{}),
                                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    // a_grid_desc_mraw_kraw.Print();
    // printf("\n");
    a_grid_desc_ak0_m_ak1.Print();
    printf("\n");

    printf("id: %d\n", a_grid_desc_ak0_m_ak1.CalculateOffset(make_multi_index(0, 0, 1)));
    printf("id: %d\n", a_grid_desc_ak0_m_ak1.CalculateOffset(make_multi_index(0, 1, 0)));
    printf("id: %d\n", a_grid_desc_ak0_m_ak1.CalculateOffset(make_multi_index(1, 0, 0)));

    // printf("a_grid_desc_mraw_kraw top: %d %d %d\n",
    //        a_grid_desc_mraw_kraw.get_length(I0),
    //        a_grid_desc_mraw_kraw.get_length(I1),
    //        a_grid_desc_mraw_kraw.get_length(I2));
}

int main(int argc, char* argv[])
{
    [[maybe_unused]] bool do_verification = true;
    [[maybe_unused]] int init_method      = 1;
    [[maybe_unused]] bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideD = 0;
    ck::index_t StrideE = N;

    ck::index_t KBatch = 1;

    [[maybe_unused]] ck::index_t Warmup = 50;
    [[maybe_unused]] ck::index_t Repeat = 50;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 12)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideD = std::stoi(argv[9]);
        StrideE = std::stoi(argv[10]);

        KBatch = std::stoi(argv[11]);
    }
    else if(argc == 14)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideD = std::stoi(argv[9]);
        StrideE = std::stoi(argv[10]);

        KBatch = std::stoi(argv[11]);

        Warmup = std::stoi(argv[12]);
        Repeat = std::stoi(argv[13]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf(
            "arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD, StrideE, KBatch\n");
        printf("arg10 to 11: Warmup, Repeat\n");
        exit(0);
    }

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<A0DataType> a0_m_k(f_host_tensor_descriptor(M, K, StrideA, A0Layout{}));
    Tensor<B0DataType> b0_k_n(f_host_tensor_descriptor(K, N, StrideB, B0Layout{}));
    Tensor<B0DataType> b0_preshuffled(
        f_host_tensor_descriptor(K, N, StrideB, B0Layout{})); // use laout only for size
    Tensor<D0DataType> d0_m_n(f_host_tensor_descriptor(M, N, StrideD, D0Layout{}));
    Tensor<D1DataType> d1_m_n(f_host_tensor_descriptor(M, N, StrideD, D1Layout{}));
    Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    std::cout << "a0_m_k: " << a0_m_k.mDesc << std::endl;
    std::cout << "b0_k_n: " << b0_k_n.mDesc << std::endl;
    std::cout << "d1_m_n: " << d1_m_n.mDesc << std::endl;
    std::cout << "d0_m_n: " << d0_m_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_host_result.mDesc << std::endl;
    std::cout << "kbatch " << KBatch << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{0, 2});
        d0_m_n.GenerateTensorValue(GeneratorTensor_2<D0DataType>{-2, 2});
        d1_m_n.GenerateTensorValue(GeneratorTensor_2<D1DataType>{-2, 2});
        break;
    case 2:
        a0_m_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{});
        b0_k_n.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        d0_m_n.GenerateTensorValue(GeneratorTensor_1<D0DataType>{});
        d1_m_n.GenerateTensorValue(GeneratorTensor_1<D1DataType>{});
        break;
    default:
        a0_m_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        b0_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
        d0_m_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{0.0, 1.0});
        d1_m_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{0.0, 1.0});
    }
    DeviceMem a0_device_buf(sizeof(A0DataType) * a0_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_k_n.mDesc.GetElementSpaceSize());
    DeviceMem d0_device_buf(sizeof(D0DataType) * d0_m_n.mDesc.GetElementSpaceSize());
    DeviceMem d1_device_buf(sizeof(D1DataType) * d1_m_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    a0_device_buf.ToDevice(a0_m_k.mData.data());
    d0_device_buf.ToDevice(d0_m_n.mData.data());
    d1_device_buf.ToDevice(d1_m_n.mData.data());
    e_device_buf.ToDevice(e_m_n_device_result.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DsDataType::Size();

    constexpr auto I0 = ck::Number<0>{};

    // do GEMM
    auto device_op = DeviceOpInstance{};

    int NPerXdl = device_op.GetPreShuffleParameters();

    preShuffleBuffer(b0_k_n.mData.data(), b0_preshuffled.mData.data(), N, K, NPerXdl);

    b0_device_buf.ToDevice(b0_preshuffled.mData.data());

    auto invoker = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(a0_device_buf.GetDeviceBuffer(),
                               b0_device_buf.GetDeviceBuffer(),
                               std::array<const void*, NumDTensor>{d0_device_buf.GetDeviceBuffer(),
                                                                   d1_device_buf.GetDeviceBuffer()},
                               e_device_buf.GetDeviceBuffer(),
                               M,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               std::array<ck::index_t, NumDTensor>{I0, I0},
                               StrideE,
                               KBatch,
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    // test_entry<DeviceOpInstance><<<dim3(1), dim3(1)>>>(argument);

    size_t total_size =
        (M * K * sizeof(A0DataType) + N * K * sizeof(B0DataType) + M * sizeof(D0DataType) +
         N * sizeof(D1DataType) + M * N * sizeof(EDataType));
    int rotate_buf_num =
        ck::math::min(size_t(Repeat), ck::math::integer_divide_ceil(512 * 1024 * 1024, total_size));

    float ave_time = invoker.Run(
        argument, StreamConfig{nullptr, time_kernel, 0, Warmup, Repeat, true, rotate_buf_num});
    // float ave_time =
    //     invoker.Run(argument, StreamConfig{nullptr, time_kernel, 0, 1, 0, true, rotate_buf_num});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N + sizeof(EDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    // clang-format off
    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
              << " GB/s" << std::endl;
    // clang-format on

    // if(do_verification)
    // {
    //     invoker.Run(argument, StreamConfig{nullptr, false});

    //     e_device_buf.FromDevice(e_m_n_device_result.mData.data());

    //     Tensor<CShuffleDataType> c_m_n({M, N});

    //     using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<A0DataType,
    //                                                                             B0DataType,
    //                                                                             CShuffleDataType,
    //                                                                             AccDataType,
    //                                                                             PassThrough,
    //                                                                             PassThrough,
    //                                                                             PassThrough>;
    //     auto ref_gemm               = ReferenceGemmInstance{};
    //     auto ref_invoker            = ref_gemm.MakeInvoker();

    //     auto ref_argument = ref_gemm.MakeArgument(
    //         a0_m_k, b0_k_n, c_m_n, PassThrough{}, PassThrough{}, PassThrough{});

    //     ref_invoker.Run(ref_argument);

    //     for(int m = 0; m < M; ++m)
    //     {
    //         for(int n = 0; n < N; ++n)
    //         {
    //             cde_element_op(e_m_n_host_result(m, n), c_m_n(m, n), d0_m_n(m, n), d1_m_n(m, n));
    //         }
    //     }

    //     e_device_buf.FromDevice(e_m_n_device_result.mData.data());

    //     return ck::utils::check_err(
    //                e_m_n_device_result, e_m_n_host_result, "Error: Incorrect results!", 1e-3,
    //                5e-2) ? 0 : 1;
    // }

    return 0;
}
