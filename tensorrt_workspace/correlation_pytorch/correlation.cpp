//
// Created by jerry on 2022/2/26.
//

//borrowed from https://github.com/leilegelei1/Pytorch-Correlation-extension/blob/master/Correlation_Module/correlation.cpp

#include <torch/extension.h>
#include <torch/script.h>
using namespace torch;

#include <vector>
#include <iostream>
using namespace std;

#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < H && y >= 0 && y < W)

template <typename scalar_t>
static void correlate_patch(
        TensorAccessor<scalar_t,3> input1,
        TensorAccessor<scalar_t,3> input2,
        scalar_t *dst,
        int64_t kH, int64_t kW,
        int64_t dilationH, int64_t dilationW,
        int64_t u, int64_t v,
        int64_t shiftU, int64_t shiftV){
    const int64_t C = input1.size(0);
    const int64_t iH = input1.size(1);
    const int64_t iW = input1.size(2);
    for (int64_t c=0; c<C; ++c){
        for (int64_t i=0; i<kH; ++i){
            int64_t i1 = u + i * dilationH;
            int64_t i2 = i1 + shiftU;
            if WITHIN_BOUNDS(i1, i2, iH, iH){
                for (int64_t j=0; j<kW; ++j){
                    int64_t j1 = v + j * dilationW;
                    int64_t j2 = j1 + shiftV;
                    if WITHIN_BOUNDS(j1, j2, iW, iW){
                        scalar_t v1 = input1[c][i1][j1];
                        scalar_t v2 = input2[c][i2][j2];
                        *dst += v1 * v2;
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
static void correlate_patch_grad(
        TensorAccessor<scalar_t,3> input1,
        TensorAccessor<scalar_t,3> gradInput1,
        TensorAccessor<scalar_t,3> input2,
        TensorAccessor<scalar_t,3> gradInput2,
        scalar_t gradOutput,
        int64_t kH, int64_t kW,
        int64_t dilationH, int64_t dilationW,
        int64_t u, int64_t v,
        int64_t shiftU, int64_t shiftV){

    const int64_t C = input1.size(0);
    const int64_t iH = input1.size(1);
    const int64_t iW = input1.size(2);

    for (int64_t c=0; c<C; ++c){
        for (int64_t i=0; i<kH; ++i){
            int64_t i1 = u + i * dilationH;
            int64_t i2 = i1 + shiftU;
            if WITHIN_BOUNDS(i1, i2, iH, iH){
                for (int64_t j=0; j<kW; ++j){
                    int64_t j1 = v + j * dilationW;
                    int64_t j2 = j1 + shiftV;
                    if WITHIN_BOUNDS(j1, j2, iW, iW){
                        scalar_t v1 = input1[c][i1][j1];
                        scalar_t v2 = input2[c][i2][j2];
                        gradInput2[c][i2][j2] += gradOutput * v1;
                        gradInput1[c][i1][j1] += gradOutput * v2;
                    }
                }
            }
        }
    }
}


//simplify the forward function to output tensor with size of (b,c,h,w) and use dilation_patchH as dilation
torch::Tensor correlation_cpp_forward(
        torch::Tensor input1,
        torch::Tensor input2,
        int64_t kH, int64_t kW,
        int64_t patchH, int64_t patchW,
        int64_t padH, int64_t padW,
        int64_t dilationH, int64_t dilationW,
        int64_t dilation_patchH, int64_t dilation_patchW,
        int64_t dH, int64_t dW) {
    const auto batch_size = input1.size(0);
    const auto iH = input1.size(2);
    const auto iW = input1.size(3);
    const int64_t patchRadH = (patchH - 1) / 2;
    const int64_t patchRadW = (patchW - 1) / 2;
    const int64_t dilatedKH = (kH - 1) * dilationH + 1;
    const int64_t dilatedKW = (kW - 1) * dilationW + 1;
    const auto oH = (iH + 2 * padH - dilatedKH) / dH + 1;
    const auto oW = (iW + 2 * padW - dilatedKW) / dW + 1;

    int64_t dilation = dilation_patchH;
    int64_t outC = (patchH * patchW -1) / dilation + 1;
    auto output = at::zeros({batch_size, outC, oH, oW}, input1.options());


    int64_t n, ph, pw, h, w;
#pragma omp parallel for private(n, ph, pw, h, w) collapse(2)
    for (n = 0; n < batch_size; ++n) {
        for(ph = 0; ph < patchH; ++ph){
            for(pw = 0; pw < patchW; ++pw){
                int64_t channelCounter = ph * patchH + pw;
                if(channelCounter%dilation!=0)
                    continue;
                int64_t channelIndex = channelCounter / dilation;
                AT_DISPATCH_FLOATING_TYPES(input1.scalar_type(), "correlation_forward_cpp", ([&] {
                    auto input1_acc = input1.accessor<scalar_t, 4>();
                    auto input2_acc = input2.accessor<scalar_t, 4>();
                    auto output_acc = output.accessor<scalar_t, 4>();
                    for (h = 0; h < oH; ++h) {
                        for (w = 0; w < oW; ++w) {
                            correlate_patch(input1_acc[n],
                                            input2_acc[n],
                                            &output_acc[n][channelIndex][h][w],
                                            kH, kW,
                                            dilationH, dilationW,
                                            -padH + h * dH,
                                            -padW + w * dW,
                                            (ph - patchRadH)  * dilation_patchH,
                                            (pw - patchRadW)  * dilation_patchW);
                        }
                    }
                }));
            }
        }
    }
    return output;
}

std::vector<torch::Tensor> correlation_cpp_backward(
        torch::Tensor input1,
        torch::Tensor input2,
        torch::Tensor gradOutput,
        int64_t kH, int64_t kW,
        int64_t patchH, int64_t patchW,
        int64_t padH, int64_t padW,
        int64_t dilationH, int64_t dilationW,
        int64_t dilation_patchH, int64_t dilation_patchW,
        int64_t dH, int64_t dW) {
    cout << 1 << endl;
    const int64_t batch_size = input1.size(0);
    const int64_t patchRadH = (patchH - 1) / 2;
    const int64_t patchRadW = (patchW - 1) / 2;
    const int64_t oH = gradOutput.size(3);
    const int64_t oW = gradOutput.size(4);

    auto gradInput1 = torch::zeros_like(input1);

    auto gradInput2 = torch::zeros_like(input2);

    int64_t n, ph, pw, h, w;
#pragma omp parallel for private(n, ph, pw, h, w)
    for (n = 0; n < batch_size; ++n) {
        AT_DISPATCH_FLOATING_TYPES(input1.scalar_type(), "correlation_backward_cpp", ([&] {
            auto input1_acc = input1.accessor<scalar_t, 4>();
            auto gradInput1_acc = gradInput1.accessor<scalar_t, 4>();
            auto input2_acc = input2.accessor<scalar_t, 4>();
            auto gradInput2_acc = gradInput2.accessor<scalar_t, 4>();
            auto gradOutput_acc = gradOutput.accessor<scalar_t, 5>();

            for(ph = 0; ph < patchH; ++ph){
                for(pw = 0; pw < patchW; ++pw){
                    for (h = 0; h < oH; ++h) {
                        for (w = 0; w < oW; ++w) {
                            correlate_patch_grad(input1_acc[n], gradInput1_acc[n],
                                                 input2_acc[n], gradInput2_acc[n],
                                                 gradOutput_acc[n][ph][pw][h][w],
                                                 kH, kW,
                                                 dilationH, dilationW,
                                                 -padH + h * dH,
                                                 -padW + w * dW,
                                                 (ph - patchRadH)  * dilation_patchH,
                                                 (pw - patchRadW)  * dilation_patchW);
                        }
                    }
                }
            }
        }));
    }

    return {gradInput1, gradInput2};
}

static auto registry = torch::RegisterOperators("mynamespace::correlation",&correlation_cpp_forward);