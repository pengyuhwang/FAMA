#include <Kun/Context.hpp>
#include <Kun/Module.hpp>
#include <Kun/Ops.hpp>
#include <Kun/Rank.hpp>
#include <Kun/Scale.hpp>
#include <Kun/Ops/Quantile.hpp>


using namespace kun;
using namespace kun::ops;






static void stage_test__8077911ee3530fde_133c7900b21163d92_ad9c4e82fedf78d4(Context* __ctx, size_t __stock_idx, size_t __total_time, size_t __start, size_t __length) {
    InputTS<float, 4> buf_close{__ctx->buffers[0].ptr, __stock_idx, __ctx->stock_count, __total_time, __start};
    OutputSTs<float, 4> buf_8077911ee3530fde{__ctx->buffers[2].ptr, __stock_idx, __ctx->stock_count, __length, 0};
    OutputSTs<float, 4> buf_133c7900b21163d92{__ctx->buffers[4].ptr, __stock_idx, __ctx->stock_count, __length, 0};
    OutputSTs<float, 4> buf_ad9c4e82fedf78d4{__ctx->buffers[6].ptr, __stock_idx, __ctx->stock_count, __length, 0};
    OutputWindow<float, 4, 10> temp_19{};
    FastWindowedSum<float, 4, 30> sum_5;
    for(size_t i = 0;i < __length;i++) {
        auto v0 = buf_close.step(i);
        ReduceMin<float, 4> v3{};
        for(int iter = 4;iter >= 0;iter--) {
            auto v2 = buf_close.getWindow(i, iter);
            v3.step(v2, iter);
        }
        buf_8077911ee3530fde.store(i, v3);
        auto v4 = v3;
        auto v5 = sum_5.step(buf_close, v0, i);
        auto v6 = Div(v5, 30.f);
        ReduceAdd<float, 4> v11{};
        for(int iter = 29;iter >= 0;iter--) {
            auto v8 = buf_close.getWindow(i, iter);
            auto v9 = Sub(v8, v6);
            auto v10 = Mul(v9, v9);
            v11.step(v10, iter);
        }
        auto v12 = Div(v11, 29.f);
        auto v13 = Sqrt(v12);
        auto v14 = GreaterThan(v13, v6);
        auto v15 = windowedRef<float, 4, 1>(buf_close, i);
        auto v16 = Div(v0, v15);
        auto v17 = constVec<4>(1.0f);
        auto v18 = Sub(v16, v17);
        temp_19.store(i, v18);
        auto v19 = v18;
        ReduceDecayLinear<float, 4, 10> v22{};
        for(int iter = 9;iter >= 0;iter--) {
            auto v21 = temp_19.getWindow(i, iter);
            v22.step(v21, iter);
        }
        buf_133c7900b21163d92.store(i, v22);
        auto v23 = v22;
        buf_ad9c4e82fedf78d4.store(i, v14);
        auto v24 = v14;
    }
}

static void stage_test__out(Context* __ctx, size_t __stock_idx, size_t __total_time, size_t __start, size_t __length) {
    InputSTs<float, 4> buf_ad9c4e82fedf78d4{__ctx->buffers[6].ptr, __stock_idx, __ctx->stock_count, __length, 0};
    InputSTs<float, 4> buf_0913d7250c35756a{__ctx->buffers[5].ptr, __stock_idx, __ctx->stock_count, __length, 0};
    InputSTs<float, 4> buf_dc2e653e313df2d2{__ctx->buffers[3].ptr, __stock_idx, __ctx->stock_count, __length, 0};
    OutputTS<float, 4> buf_out{__ctx->buffers[1].ptr, __stock_idx, __ctx->stock_count, __length, 0};
    for(size_t i = 0;i < __length;i++) {
        auto v0 = buf_ad9c4e82fedf78d4.step(i);
        auto v1 = buf_0913d7250c35756a.step(i);
        auto v2 = buf_dc2e653e313df2d2.step(i);
        auto v3 = Select(v0, v1, v2);
        buf_out.store(i, v3);
        auto v4 = v3;
    }
}

static BufferInfo __buffers_test[]{
    {0, "close", 1, BufferKind::INPUT, 0, 31},
    {1, "out", 0, BufferKind::OUTPUT, 30, 1},
    {2, "8077911ee3530fde", 1, BufferKind::TEMP, 0, 1},
    {3, "dc2e653e313df2d2", 1, BufferKind::TEMP, 0, 1},
    {4, "133c7900b21163d92", 1, BufferKind::TEMP, 0, 1},
    {5, "0913d7250c35756a", 1, BufferKind::TEMP, 0, 1},
    {6, "ad9c4e82fedf78d4", 1, BufferKind::TEMP, 0, 1}
};

static BufferInfo *stage_dc2e653e313df2d2_in_buf[] = {&__buffers_test[2]};
static BufferInfo *stage_dc2e653e313df2d2_out_buf[] = {&__buffers_test[3]};
static BufferInfo *stage_0913d7250c35756a_in_buf[] = {&__buffers_test[4]};
static BufferInfo *stage_0913d7250c35756a_out_buf[] = {&__buffers_test[5]};
static BufferInfo *stage_8077911ee3530fde_133c7900b21163d92_ad9c4e82fedf78d4_in_buf[] = {&__buffers_test[0]};
static BufferInfo *stage_8077911ee3530fde_133c7900b21163d92_ad9c4e82fedf78d4_out_buf[] = {&__buffers_test[2], &__buffers_test[4], &__buffers_test[6]};
static BufferInfo *stage_out_in_buf[] = {&__buffers_test[6], &__buffers_test[5], &__buffers_test[3]};
static BufferInfo *stage_out_out_buf[] = {&__buffers_test[1]};

namespace {
extern Stage *stage_test__dc2e653e313df2d2_dep[1];
extern Stage *stage_test__0913d7250c35756a_dep[1];
extern Stage *stage_test__8077911ee3530fde_133c7900b21163d92_ad9c4e82fedf78d4_dep[3];
Stage **stage_test__out_dep = nullptr;
}


static auto stage_test__dc2e653e313df2d2 = RankStocks<MapperSTs<float, 4>, MapperSTs<float, 4>>;
static auto stage_test__0913d7250c35756a = RankStocks<MapperSTs<float, 4>, MapperSTs<float, 4>>;

static Stage __stages_test[] = {
    {/*f*/ stage_test__dc2e653e313df2d2, /*dependers*/ stage_test__dc2e653e313df2d2_dep, /*num_dependers*/ 1,
     /*in_buffers*/ stage_dc2e653e313df2d2_in_buf, /*num_in_buffers*/ 1,
     /*out_buffers*/ stage_dc2e653e313df2d2_out_buf, /*num_out_buffers*/ 1, /*pending_out*/ 1,
     /*num_tasks*/ TaskExecKind::SLICE_BY_TIME, /*id*/ 0},
    {/*f*/ stage_test__0913d7250c35756a, /*dependers*/ stage_test__0913d7250c35756a_dep, /*num_dependers*/ 1,
     /*in_buffers*/ stage_0913d7250c35756a_in_buf, /*num_in_buffers*/ 1,
     /*out_buffers*/ stage_0913d7250c35756a_out_buf, /*num_out_buffers*/ 1, /*pending_out*/ 1,
     /*num_tasks*/ TaskExecKind::SLICE_BY_TIME, /*id*/ 1},
    {/*f*/ stage_test__8077911ee3530fde_133c7900b21163d92_ad9c4e82fedf78d4, /*dependers*/ stage_test__8077911ee3530fde_133c7900b21163d92_ad9c4e82fedf78d4_dep, /*num_dependers*/ 3,
     /*in_buffers*/ stage_8077911ee3530fde_133c7900b21163d92_ad9c4e82fedf78d4_in_buf, /*num_in_buffers*/ 1,
     /*out_buffers*/ stage_8077911ee3530fde_133c7900b21163d92_ad9c4e82fedf78d4_out_buf, /*num_out_buffers*/ 3, /*pending_out*/ 0,
     /*num_tasks*/ TaskExecKind::SLICE_BY_STOCK, /*id*/ 2},
    {/*f*/ stage_test__out, /*dependers*/ stage_test__out_dep, /*num_dependers*/ 0,
     /*in_buffers*/ stage_out_in_buf, /*num_in_buffers*/ 3,
     /*out_buffers*/ stage_out_out_buf, /*num_out_buffers*/ 1, /*pending_out*/ 3,
     /*num_tasks*/ TaskExecKind::SLICE_BY_STOCK, /*id*/ 3}
};

namespace {
Stage *stage_test__dc2e653e313df2d2_dep[] = {&__stages_test[3]};
Stage *stage_test__0913d7250c35756a_dep[] = {&__stages_test[3]};
Stage *stage_test__8077911ee3530fde_133c7900b21163d92_ad9c4e82fedf78d4_dep[] = {&__stages_test[0], &__stages_test[1], &__stages_test[3]};
}


KUN_EXPORT Module test{
    0x64100003,
    4,
    __stages_test,
    7,
    __buffers_test,
    MemoryLayout::TS,
    MemoryLayout::TS,
    4,
    Datatype::Float,
    1
};