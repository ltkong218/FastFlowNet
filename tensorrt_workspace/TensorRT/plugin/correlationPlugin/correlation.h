//
// Created by jerry on 2022/2/26.
//

#ifndef TENSORRT_CORRELATION_H
#define TENSORRT_CORRELATION_H

namespace torch{
namespace detail{

enum class CorrelationDataType{
    GFLOAT,
    GHALF
};
}
}
int correlation_cuda(int batchSize, const void* queryPtr, const void* referencePtr, void* const outputPtr,
    int inp_C,int inp_H, int inp_W, int out_H,int out_W,
    int inp_sN, int inp_sC, int inp_sH, int inp_sW,
    int out_sN, int out_sC, int out_sH, int out_sW,
    int kH,int kW,int patchH,int patchW,int padH, int padW, int dilation,
    torch::detail::CorrelationDataType dataType, cudaStream_t stream);

#endif // TENSORRT_CORRELATION_H
