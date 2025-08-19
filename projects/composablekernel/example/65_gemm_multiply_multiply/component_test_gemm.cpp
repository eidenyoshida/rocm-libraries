#include <cstdlib>
#include <iostream>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/sequence_helper.hpp"
#include "ck/utility/thread_group.hpp"

template <typename... Ts>
struct [[deprecated]] pntS
{
};

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"

using namespace ck;

using ADataType      = f8_t;
constexpr index_t M_ = 64;
constexpr index_t N_ = 3072;
constexpr index_t K_ = 1536;

__global__ void test_entry(ADataType* p_a_grid)
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    constexpr auto M  = Number<M_>{};
    constexpr auto N  = Number<N_>{};
    constexpr auto K  = Number<K_>{};
    constexpr auto K1 = Number<16>{};
    constexpr auto K0 = K / K1;

    constexpr auto MPerBlock = Number<64>{};
    constexpr auto NPerBlock = Number<64>{};
    constexpr auto KPerBlock = Number<256>{};

    const index_t BlockSize = 256;

    // a_grid_desc_ak0_m_ak
    const auto a_grid_desc_m_k = make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(K, I1));
    const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
        a_grid_desc_m_k,
        make_tuple(make_unmerge_transform(make_tuple(K0, K1)), make_pass_through_transform(M)),
        make_tuple(Sequence<1>{}, Sequence<0>{}),
        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    // a_block_desc_ak0_m_ak1
    constexpr auto a_lds_block_desc_raw =
        make_naive_tensor_descriptor(make_tuple(K0, MPerBlock, K1), make_tuple(K1, KPerBlock, I1));
    constexpr auto a_block_desc_ak0_m_ak1 = transform_tensor_descriptor(
        a_lds_block_desc_raw,
        make_tuple(make_xor_with_modulo_transform(make_tuple(MPerBlock, K0)),
                   make_pass_through_transform(K1)),
        make_tuple(Sequence<1, 0>{}, Sequence<2>{}),
        make_tuple(Sequence<1, 0>{}, Sequence<2>{}));

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;
    using PassThrough     = ck::tensor_operation::element_wise::PassThrough;

    using ThreadGroupTensorSliceTransfer = ThreadGroupTensorSliceTransfer_v4r1<
        ThisThreadBlock,
        PassThrough, // AElementwiseOperation
        PassThrough,
        InMemoryDataOperationEnum::Set,
        Sequence<K0, MPerBlock, K1>, // BlockSliceLengths
        Sequence<16, 16, 1>,         // ABlockTransferThreadClusterLengths_AK0_M_AK1
        Sequence<1, 0, 2>,           // ABlockTransferThreadClusterArrangeOrder
        ck::f8_t,                    // ADataType
        ck::f8_t,                    // LDSTypeA
        decltype(a_grid_desc_ak0_m_ak1),
        decltype(a_block_desc_ak0_m_ak1),
        Sequence<1, 0, 2>, // ABlockTransferSrcAccessOrder
        Sequence<0, 1, 2>,
        2, // ABlockTransferSrcVectorDim
        2,
        16, // ABlockTransferSrcScalarPerVector
        16, // ABlockTransferDstScalarPerVector_AK1
        1,
        1,
        false, // AThreadTransferSrcResetCoordinateAfterRun
        true,
        1>;

    const auto block_2_ctile_map =
        BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock, NPerBlock>{M, N, 4};
    const auto block_work_idx =
        block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));
    const index_t block_m_id               = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);
    const index_t m_block_data_idx_on_grid = __builtin_amdgcn_readfirstlane(block_m_id * MPerBlock);

    // set blockwise copy
    auto a_blockwise_copy =
        ThreadGroupTensorSliceTransfer(a_grid_desc_ak0_m_ak1,
                                       make_multi_index(0, m_block_data_idx_on_grid, 0),
                                       PassThrough{},
                                       a_block_desc_ak0_m_ak1,
                                       make_multi_index(0, 0, 0),
                                       PassThrough{});

    // do some real copy
    constexpr auto a_block_copy_step = make_multi_index(KPerBlock / K1, 0, 0);

    const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
        p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());

    __shared__ char p_shared[a_block_desc_ak0_m_ak1.GetElementSpaceSize()];
    void* p_shared_void = p_shared;
    auto a_block_buf    = make_dynamic_buffer<AddressSpaceEnum::Lds>(
        static_cast<ck::f8_t*>(p_shared_void), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

    a_blockwise_copy.RunRead(a_grid_desc_ak0_m_ak1, a_grid_buf);
    // a_grid_desc_ak0_m_ak1.Print();
    // printf("\n-lengths: ");
    // a_grid_desc_ak0_m_ak1.GetLengths().Print();
    // printf("\n");
    // printf("-id: %d\n", a_grid_desc_ak0_m_ak1.CalculateOffset(make_multi_index(0, 0, 0)));
    // printf("-id: %d\n", a_grid_desc_ak0_m_ak1.CalculateOffset(make_multi_index(0, 0, 1)));
    // printf("-id: %d\n", a_grid_desc_ak0_m_ak1.CalculateOffset(make_multi_index(0, 1, 0)));
    // printf("-id: %d\n", a_grid_desc_ak0_m_ak1.CalculateOffset(make_multi_index(1, 0, 0)));

    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_ak0_m_ak1, a_block_copy_step);
    __builtin_amdgcn_sched_barrier(0);
    a_blockwise_copy.RunWrite(a_block_desc_ak0_m_ak1, a_block_buf);
}

int main()
{
    Tensor<ADataType> a0_m_k = HostTensorDescriptor({M_, K_}, {K_, 1});
    a0_m_k.SetZero();

    DeviceMem a0_device_buf(sizeof(ADataType) * a0_m_k.mDesc.GetElementSpaceSize());
    a0_device_buf.ToDevice(a0_m_k.mData.data());

    test_entry<<<dim3(1), dim3(1)>>>(static_cast<ADataType*>(a0_device_buf.GetDeviceBuffer()));
}
