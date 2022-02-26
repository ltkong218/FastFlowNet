/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#ifndef GRID_SAMPLER_H
#define GRID_SAMPLER_H

namespace torch {

namespace detail {

  enum class GridSamplerInterpolation {Bilinear, Nearest};
  enum class GridSamplerPadding {Zeros, Border, Reflection};
  enum class GridSamplerDataType {GFLOAT, GHALF};

}  // namespace detail

} // namespace torch


// function naming is algined with Torch
int grid_sampler_2d_cuda(int batchSize, const void* inputPtr, const void* gridPtr,
                            void* const outputPtr,
                            int C,
                            int inp_H,
                            int inp_W,
                            int out_H, // same as grid_H
                            int out_W, // same as grid_W
                            int inp_sN,
                            int inp_sC,
                            int inp_sH,
                            int inp_sW,
                            int grid_sN,
                            int grid_sH,
                            int grid_sW,
                            int grid_sCoor,
                            int out_sN,
                            int out_sC,
                            int out_sH,
                            int out_sW,
                            torch::detail::GridSamplerInterpolation interpolation_mode,
                            torch::detail::GridSamplerPadding padding_mode,
                            bool align_corners, torch::detail::GridSamplerDataType dataType, cudaStream_t stream);

#endif