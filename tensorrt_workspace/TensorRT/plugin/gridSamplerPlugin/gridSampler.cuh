
#ifndef GRID_SAMPLER_CUH
#define GRID_SAMPLER_CUH

#include <cuda_runtime.h>

#include <cuda_fp16.h>

#include "gridSampler.h"

static __forceinline__ __device__
__half operator*(const __half& a, const int& b)
{
    return a * __int2half_rn(b);
}

static __forceinline__ __device__
__half operator/(const __half& a, const int& b)
{
    return a / __int2half_rn(b);
}

static __forceinline__ __device__
__half operator+(const __half& a, const float& b)
{
    return a + __float2half(b);
}

static __forceinline__ __device__
__half operator-(const __half& a, const int& b)
{
    return a - __int2half_rn(b);
}

static __forceinline__ __device__
__half operator+=(const __half& a, const __half& b)
{
    return a + b;
}

static __forceinline__ __device__
__half min(const __half& a, const half& b)
{
    return __float2half(min(__half2float(a), __half2float(b)));
}

static __forceinline__ __device__
__half max(const __half& a, const half& b)
{
    return __float2half(max(__half2float(a), __half2float(b)));
}

static __forceinline__ __device__
__half fabs(const __half& a)
{
    //TODO return __habs(a); what happened.
    return __float2half(fabs(__half2float(a)));
}

static __forceinline__ __device__
__half floorf(const __half& a)
{
    return hfloor(a);
}

static __forceinline__ __device__
__half roundf(const __half& a)
{
    return hrint(a);
}

static __forceinline__ __device__
__half fmod(const __half& a, const __half& b)
{
  return __float2half(fmodf(__half2float(a), __half2float(b)));
}

// Unnormalizes a coordinate from the -1 to +1 scale to its pixel index value,
// where we view each pixel as an area between (idx - 0.5) and (idx + 0.5).
// if align_corners: -1 and +1 get sent to the centers of the corner pixels
//     -1 --> 0
//     +1 --> (size - 1)
//     scale_factor = (size - 1) / 2
// if not align_corners: -1 and +1 get sent to the image edges
//     -1 --> -0.5
//     +1 --> (size - 1) + 0.5 == size - 0.5
//     scale_factor = size / 2
template <typename scalar_t>
static __forceinline__ __device__
scalar_t grid_sampler_unnormalize(scalar_t coord, int size, bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1.f) * size - 1) / 2;
  }
}


// Clips coordinates to between 0 and clip_limit - 1
template <typename scalar_t>
static __forceinline__ __device__
scalar_t clip_coordinates(scalar_t in, int clip_limit) {
  return ::min(static_cast<scalar_t>(clip_limit - 1), ::max(in, static_cast<scalar_t>(0)));
}

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template <typename scalar_t>
static __forceinline__ __device__
scalar_t reflect_coordinates(scalar_t in, int twice_low, int twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = ::fabs(in - min);
  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  scalar_t extra = ::fmod(in, span);
  int flips = static_cast<int>(::floorf(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}


// Computes the pixel source index value for a grid coordinate
template <typename scalar_t>
static __forceinline__ __device__
scalar_t grid_sampler_compute_source_index(
    scalar_t coord,
    int size,
    torch::detail::GridSamplerPadding padding_mode,
    bool align_corners) {
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  if (padding_mode == torch::detail::GridSamplerPadding::Border) {
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == torch::detail::GridSamplerPadding::Reflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2*(size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2*size - 1);
      // when align_corners=False, reflection does not auto clip coords
      coord = clip_coordinates(coord, size);
    }
  }
  return coord;
}


static __forceinline__ __device__
bool within_bounds_2d(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}


#endif