// cnn.cpp — U-Net Convolutional Neural Network for Astar Island terrain prediction
// Takes simulator predictions + initial grid as input, learns context-dependent corrections.
// Architecture: 2-level U-Net (40→20→10→20→40) with skip connections, instance norm,
//               residual learning, and 8× data augmentation.
// Usage: cnn <grids.bin> <ground_truth.bin> <round> <seed> [prediction.bin]
//        [--exclude <round>] [--epochs N] [--lr F] [--sim-dir path]
//
// Compile: c++ -std=c++17 -O3 -framework Accelerate -o cnn cnn.cpp
// Linux:   g++ -std=c++17 -O3 -pthread -lopenblas -o cnn cnn.cpp

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <algorithm>
#include <numeric>
#include <set>
#include <cassert>
#include <thread>
#include <mutex>
#include <atomic>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

static const int NUM_CLASSES = 6;
static const int GRID_W = 40;
static const int GRID_H = 40;
static const int IN_CHANNELS = 13;  // 6 sim + 6 one-hot + 1 is_land

// --- Terrain mapping (from features.hpp) ---

inline int terrain_to_class(int code)
{
    switch(code)
    {
        case 1: return 1;
        case 2: return 2;
        case 3: return 3;
        case 4: return 4;
        case 5: return 5;
        default: return 0; // 0, 10, 11 -> class 0
    }
}

inline bool is_land_code(int code)
{
    return code != 10;
}

// --- Tensor: a 3D (C x H x W) contiguous buffer ---

struct Tensor
{
    int C, H, W;
    std::vector<float> data;

    Tensor() : C(0), H(0), W(0) {}
    Tensor(int c, int h, int w) : C(c), H(h), W(w), data(c * h * w, 0.0f) {}

    float& at(int c, int y, int x) { return data[(c * H + y) * W + x]; }
    float  at(int c, int y, int x) const { return data[(c * H + y) * W + x]; }

    void zero() { std::fill(data.begin(), data.end(), 0.0f); }
    int size() const { return (int)data.size(); }
};

// ============================================================
// Im2col / Col2im for GEMM-based convolution
// ============================================================

// Extracts overlapping patches into column matrix [in_ch*kH*kW, H*W] (row-major).
// Same-padding: out-of-bounds positions get 0.0f.
inline void im2col(const float* data_im, int C, int H, int W,
                   int kH, int kW, int padH, int padW,
                   float* data_col)
{
    const int HW = H * W;
    for (int ic = 0; ic < C; ic++) {
        for (int ky = 0; ky < kH; ky++) {
            for (int kx = 0; kx < kW; kx++) {
                const int row = (ic * kH + ky) * kW + kx;
                float* col_row = data_col + row * HW;
                const int iy_base = ky - padH;
                const int ix_base = kx - padW;
                const float* im_ch = data_im + ic * H * W;
                for (int y = 0; y < H; y++) {
                    const int iy = y + iy_base;
                    if (iy < 0 || iy >= H) {
                        // Entire row is zero-padded
                        memset(col_row + y * W, 0, W * sizeof(float));
                    } else {
                        const float* im_row = im_ch + iy * W;
                        float* out_row = col_row + y * W;
                        for (int x = 0; x < W; x++) {
                            const int ix = x + ix_base;
                            out_row[x] = (ix >= 0 && ix < W) ? im_row[ix] : 0.0f;
                        }
                    }
                }
            }
        }
    }
}

// Inverse of im2col: accumulates column data back into CHW tensor.
// Output buffer must be zero-initialized before calling.
inline void col2im(const float* data_col, int C, int H, int W,
                   int kH, int kW, int padH, int padW,
                   float* data_im)
{
    const int HW = H * W;
    for (int ic = 0; ic < C; ic++) {
        for (int ky = 0; ky < kH; ky++) {
            for (int kx = 0; kx < kW; kx++) {
                const int row = (ic * kH + ky) * kW + kx;
                const float* col_row = data_col + row * HW;
                const int iy_base = ky - padH;
                const int ix_base = kx - padW;
                float* im_ch = data_im + ic * H * W;
                for (int y = 0; y < H; y++) {
                    const int iy = y + iy_base;
                    if (iy < 0 || iy >= H) continue;
                    float* im_row = im_ch + iy * W;
                    const float* col_ptr = col_row + y * W;
                    for (int x = 0; x < W; x++) {
                        const int ix = x + ix_base;
                        if (ix >= 0 && ix < W)
                            im_row[ix] += col_ptr[x];
                    }
                }
            }
        }
    }
}

// ============================================================
// U-Net layers: Conv2D, InstanceNorm, ReLU, MaxPool, Upsample
// ============================================================

// --- Conv2D layer with 3x3 or arbitrary kernel, same padding ---

struct ConvLayer
{
    int in_ch, out_ch, kH, kW;
    std::vector<float> weights, bias;
    std::vector<float> dweights, dbias;
    std::vector<float> mw, vw, mb, vb; // Adam state

    Tensor input_cache;

    ConvLayer() : in_ch(0), out_ch(0), kH(0), kW(0) {}

    ConvLayer(int ic, int oc, int kh, int kw)
        : in_ch(ic), out_ch(oc), kH(kh), kW(kw),
          weights(oc * ic * kh * kw, 0.0f), bias(oc, 0.0f),
          dweights(oc * ic * kh * kw, 0.0f), dbias(oc, 0.0f),
          mw(oc * ic * kh * kw, 0.0f), vw(oc * ic * kh * kw, 0.0f),
          mb(oc, 0.0f), vb(oc, 0.0f)
    {}

    float& w(int oc, int ic, int ky, int kx)
    { return weights[((oc * in_ch + ic) * kH + ky) * kW + kx]; }

    float w(int oc, int ic, int ky, int kx) const
    { return weights[((oc * in_ch + ic) * kH + ky) * kW + kx]; }

    float& dw(int oc, int ic, int ky, int kx)
    { return dweights[((oc * in_ch + ic) * kH + ky) * kW + kx]; }

    void init_he(std::mt19937& rng)
    {
        float scale = sqrtf(2.0f / (in_ch * kH * kW));
        std::normal_distribution<float> d(0.0f, scale);
        for(auto& v : weights) v = d(rng);
        std::fill(bias.begin(), bias.end(), 0.0f);
    }

    void init_xavier(std::mt19937& rng)
    {
        float scale = sqrtf(1.0f / (in_ch * kH * kW));
        std::normal_distribution<float> d(0.0f, scale);
        for(auto& v : weights) v = d(rng);
        std::fill(bias.begin(), bias.end(), 0.0f);
    }

    void zero_grad()
    {
        std::fill(dweights.begin(), dweights.end(), 0.0f);
        std::fill(dbias.begin(), dbias.end(), 0.0f);
    }

    Tensor forward(const Tensor& input, float* workspace, bool cache = true)
    {
        if(cache) input_cache = input;
        int H = input.H, W = input.W;
        int padH = kH / 2, padW = kW / 2;
        int N = H * W;          // spatial output size
        int K = in_ch * kH * kW; // column height

        // Im2col: input → workspace [K × N]
        im2col(input.data.data(), in_ch, H, W, kH, kW, padH, padW, workspace);

        // GEMM: output = weights[out_ch × K] × workspace[K × N]
        Tensor out(out_ch, H, W);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    out_ch, N, K,
                    1.0f,
                    weights.data(), K,
                    workspace, N,
                    0.0f,
                    out.data.data(), N);

        // Add bias
        for (int oc = 0; oc < out_ch; oc++) {
            float b = bias[oc];
            float* row = out.data.data() + oc * N;
            for (int i = 0; i < N; i++) row[i] += b;
        }
        return out;
    }

    Tensor backward(const Tensor& d_output, float* workspace)
    {
        int H = d_output.H, W = d_output.W;
        int padH = kH / 2, padW = kW / 2;
        int N = H * W;
        int K = in_ch * kH * kW;

        // 1. dbias: sum each output channel's gradient over spatial dims
        for (int oc = 0; oc < out_ch; oc++) {
            const float* dout_row = d_output.data.data() + oc * N;
            float sum = 0;
            for (int i = 0; i < N; i++) sum += dout_row[i];
            dbias[oc] += sum;
        }

        // 2. Recompute im2col(input_cache) → workspace [K × N]
        im2col(input_cache.data.data(), in_ch, H, W, kH, kW, padH, padW, workspace);

        // 3. dweights += d_output[out_ch × N] × workspace^T[N × K]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    out_ch, K, N,
                    1.0f,
                    d_output.data.data(), N,
                    workspace, N,
                    1.0f,  // beta=1.0: accumulate into existing dweights
                    dweights.data(), K);

        // 4. d_col = weights^T[K × out_ch] × d_output[out_ch × N] → workspace [K × N]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    K, N, out_ch,
                    1.0f,
                    weights.data(), K,
                    d_output.data.data(), N,
                    0.0f,
                    workspace, N);

        // 5. col2im: workspace → d_input
        Tensor d_input(in_ch, H, W);
        col2im(workspace, in_ch, H, W, kH, kW, padH, padW, d_input.data.data());
        return d_input;
    }
};

// --- Instance Normalization layer (normalize across H×W per channel) ---
// With learnable gamma (scale) and beta (shift), plus running stats for inference

struct InstanceNormLayer
{
    int channels;
    std::vector<float> gamma, beta;           // learnable params [C]
    std::vector<float> dgamma, dbeta;         // gradients
    std::vector<float> mg, vg, mbe, vbe;      // Adam state for gamma, beta
    std::vector<float> running_mean, running_var; // running stats for inference
    float momentum;
    bool has_running_stats;

    // Cached values for backprop
    Tensor input_norm;     // normalized input (before scale/shift)
    std::vector<float> cached_mean, cached_invstd; // per-channel

    InstanceNormLayer() : channels(0), momentum(0.9f), has_running_stats(false) {}

    InstanceNormLayer(int c, float mom = 0.9f)
        : channels(c), momentum(mom), has_running_stats(false),
          gamma(c, 1.0f), beta(c, 0.0f),
          dgamma(c, 0.0f), dbeta(c, 0.0f),
          mg(c, 0.0f), vg(c, 0.0f), mbe(c, 0.0f), vbe(c, 0.0f),
          running_mean(c, 0.0f), running_var(c, 1.0f)
    {}

    void zero_grad()
    {
        std::fill(dgamma.begin(), dgamma.end(), 0.0f);
        std::fill(dbeta.begin(), dbeta.end(), 0.0f);
    }

    // Forward: training mode (compute stats from input, update running stats)
    Tensor forward_train(const Tensor& input)
    {
        int H = input.H, W = input.W;
        int HW = H * W;
        float eps = 1e-5f;

        Tensor out(channels, H, W);
        input_norm = Tensor(channels, H, W);
        cached_mean.resize(channels);
        cached_invstd.resize(channels);

        for(int c = 0; c < channels; c++)
        {
            // Compute mean across spatial dims
            float mean = 0;
            for(int y = 0; y < H; y++)
                for(int x = 0; x < W; x++)
                    mean += input.at(c, y, x);
            mean /= HW;

            // Compute variance
            float var = 0;
            for(int y = 0; y < H; y++)
                for(int x = 0; x < W; x++)
                {
                    float d = input.at(c, y, x) - mean;
                    var += d * d;
                }
            var /= HW;

            float invstd = 1.0f / sqrtf(var + eps);
            cached_mean[c] = mean;
            cached_invstd[c] = invstd;

            // Update running stats
            running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * mean;
            running_var[c]  = momentum * running_var[c]  + (1.0f - momentum) * var;
            has_running_stats = true;

            // Normalize, scale, shift
            for(int y = 0; y < H; y++)
                for(int x = 0; x < W; x++)
                {
                    float xn = (input.at(c, y, x) - mean) * invstd;
                    input_norm.at(c, y, x) = xn;
                    out.at(c, y, x) = gamma[c] * xn + beta[c];
                }
        }
        return out;
    }

    // Forward: inference mode (use running stats)
    Tensor forward_infer(const Tensor& input)
    {
        int H = input.H, W = input.W;
        float eps = 1e-5f;
        Tensor out(channels, H, W);

        for(int c = 0; c < channels; c++)
        {
            float mean = has_running_stats ? running_mean[c] : 0.0f;
            float var  = has_running_stats ? running_var[c]  : 1.0f;
            float invstd = 1.0f / sqrtf(var + eps);

            for(int y = 0; y < H; y++)
                for(int x = 0; x < W; x++)
                {
                    float xn = (input.at(c, y, x) - mean) * invstd;
                    out.at(c, y, x) = gamma[c] * xn + beta[c];
                }
        }
        return out;
    }

    // Backward: compute d_input, accumulate dgamma, dbeta
    Tensor backward(const Tensor& d_output)
    {
        int H = d_output.H, W = d_output.W;
        int HW = H * W;
        Tensor d_input(channels, H, W);

        for(int c = 0; c < channels; c++)
        {
            float invstd = cached_invstd[c];

            // Accumulate dgamma and dbeta
            float dg = 0, db = 0;
            for(int y = 0; y < H; y++)
                for(int x = 0; x < W; x++)
                {
                    float dout = d_output.at(c, y, x);
                    dg += dout * input_norm.at(c, y, x);
                    db += dout;
                }
            dgamma[c] += dg;
            dbeta[c]  += db;

            // d_input via instance norm backward
            // d_xhat = d_out * gamma
            // d_input = invstd * (d_xhat - mean(d_xhat) - xhat * mean(d_xhat * xhat))
            float sum_dxhat = 0, sum_dxhat_xhat = 0;
            for(int y = 0; y < H; y++)
                for(int x = 0; x < W; x++)
                {
                    float dxhat = d_output.at(c, y, x) * gamma[c];
                    sum_dxhat += dxhat;
                    sum_dxhat_xhat += dxhat * input_norm.at(c, y, x);
                }

            float inv_HW = 1.0f / HW;
            float mean_dxhat = sum_dxhat * inv_HW;
            float mean_dxhat_xhat = sum_dxhat_xhat * inv_HW;

            for(int y = 0; y < H; y++)
                for(int x = 0; x < W; x++)
                {
                    float dxhat = d_output.at(c, y, x) * gamma[c];
                    d_input.at(c, y, x) = invstd * (dxhat - mean_dxhat
                        - input_norm.at(c, y, x) * mean_dxhat_xhat);
                }
        }
        return d_input;
    }
};

// --- MaxPool 2x2 ---

struct MaxPoolResult
{
    Tensor output;
    std::vector<int> indices; // index into input spatial (y*W+x) for each output cell
};

MaxPoolResult maxpool2x2_forward(const Tensor& input)
{
    int C = input.C, H = input.H, W = input.W;
    int oH = H / 2, oW = W / 2;
    MaxPoolResult res;
    res.output = Tensor(C, oH, oW);
    res.indices.resize(C * oH * oW);

    for(int c = 0; c < C; c++)
    {
        for(int oy = 0; oy < oH; oy++)
        {
            for(int ox = 0; ox < oW; ox++)
            {
                int iy = oy * 2, ix = ox * 2;
                float maxv = input.at(c, iy, ix);
                int maxidx = iy * W + ix;

                float v1 = input.at(c, iy, ix + 1);
                if(v1 > maxv) { maxv = v1; maxidx = iy * W + ix + 1; }

                float v2 = input.at(c, iy + 1, ix);
                if(v2 > maxv) { maxv = v2; maxidx = (iy + 1) * W + ix; }

                float v3 = input.at(c, iy + 1, ix + 1);
                if(v3 > maxv) { maxv = v3; maxidx = (iy + 1) * W + ix + 1; }

                res.output.at(c, oy, ox) = maxv;
                res.indices[(c * oH + oy) * oW + ox] = maxidx;
            }
        }
    }
    return res;
}

Tensor maxpool2x2_backward(const Tensor& d_output, const std::vector<int>& indices,
                            int in_H, int in_W)
{
    int C = d_output.C, oH = d_output.H, oW = d_output.W;
    Tensor d_input(C, in_H, in_W);

    for(int c = 0; c < C; c++)
    {
        for(int oy = 0; oy < oH; oy++)
        {
            for(int ox = 0; ox < oW; ox++)
            {
                int idx = indices[(c * oH + oy) * oW + ox];
                int iy = idx / in_W, ix = idx % in_W;
                d_input.at(c, iy, ix) += d_output.at(c, oy, ox);
            }
        }
    }
    return d_input;
}

// --- Upsample 2x2 (nearest neighbor) ---

Tensor upsample2x2_forward(const Tensor& input)
{
    int C = input.C, H = input.H, W = input.W;
    int oH = H * 2, oW = W * 2;
    Tensor out(C, oH, oW);

    for(int c = 0; c < C; c++)
        for(int y = 0; y < H; y++)
            for(int x = 0; x < W; x++)
            {
                float v = input.at(c, y, x);
                out.at(c, y*2,   x*2  ) = v;
                out.at(c, y*2,   x*2+1) = v;
                out.at(c, y*2+1, x*2  ) = v;
                out.at(c, y*2+1, x*2+1) = v;
            }
    return out;
}

Tensor upsample2x2_backward(const Tensor& d_output)
{
    int C = d_output.C, oH = d_output.H, oW = d_output.W;
    int H = oH / 2, W = oW / 2;
    Tensor d_input(C, H, W);

    for(int c = 0; c < C; c++)
        for(int y = 0; y < H; y++)
            for(int x = 0; x < W; x++)
                d_input.at(c, y, x) = d_output.at(c, y*2, x*2)
                                     + d_output.at(c, y*2, x*2+1)
                                     + d_output.at(c, y*2+1, x*2)
                                     + d_output.at(c, y*2+1, x*2+1);
    return d_input;
}

// --- Concatenate two tensors along channel dim ---

Tensor concat_channels(const Tensor& a, const Tensor& b)
{
    int H = a.H, W = a.W;
    Tensor out(a.C + b.C, H, W);

    // CHW layout: a's data is contiguous, then b's
    memcpy(out.data.data(), a.data.data(), a.data.size() * sizeof(float));
    memcpy(out.data.data() + a.data.size(), b.data.data(), b.data.size() * sizeof(float));
    return out;
}

// Split gradient along channel dim: first aC channels, rest
void split_gradient(const Tensor& d_concat, int aC, Tensor& d_a, Tensor& d_b)
{
    int H = d_concat.H, W = d_concat.W;
    int bC = d_concat.C - aC;
    d_a = Tensor(aC, H, W);
    d_b = Tensor(bC, H, W);

    // CHW layout: first aC channels contiguous, then bC
    memcpy(d_a.data.data(), d_concat.data.data(), aC * H * W * sizeof(float));
    memcpy(d_b.data.data(), d_concat.data.data() + aC * H * W, bC * H * W * sizeof(float));
}

// --- ReLU helpers ---

void relu_forward(const Tensor& in, Tensor& out)
{
    out = Tensor(in.C, in.H, in.W);
    for(int i = 0; i < in.size(); i++)
        out.data[i] = in.data[i] > 0 ? in.data[i] : 0;
}

Tensor relu_backward(const Tensor& d_output, const Tensor& pre_act)
{
    Tensor d_input(d_output.C, d_output.H, d_output.W);
    for(int i = 0; i < d_output.size(); i++)
        d_input.data[i] = pre_act.data[i] > 0 ? d_output.data[i] : 0;
    return d_input;
}

// Softmax per-cell across C channels
void softmax_forward(const Tensor& z, Tensor& out)
{
    out = Tensor(z.C, z.H, z.W);
    for(int y = 0; y < z.H; y++)
    {
        for(int x = 0; x < z.W; x++)
        {
            float maxv = -1e30f;
            for(int c = 0; c < z.C; c++)
                maxv = std::max(maxv, z.at(c, y, x));
            float esum = 0;
            for(int c = 0; c < z.C; c++)
            {
                float e = expf(z.at(c, y, x) - maxv);
                out.at(c, y, x) = e;
                esum += e;
            }
            for(int c = 0; c < z.C; c++)
                out.at(c, y, x) /= esum;
        }
    }
}

// ============================================================
// U-Net Network
// ============================================================

struct UNet
{
    int ch0, ch1, ch_bot; // channel sizes for each level

    // Encoder Level 0: 40x40
    ConvLayer enc0_conv1; // IN_CHANNELS -> ch0
    InstanceNormLayer enc0_bn1;
    ConvLayer enc0_conv2; // ch0 -> ch0
    InstanceNormLayer enc0_bn2;

    // Encoder Level 1: 20x20
    ConvLayer enc1_conv1; // ch0 -> ch1
    InstanceNormLayer enc1_bn1;
    ConvLayer enc1_conv2; // ch1 -> ch1
    InstanceNormLayer enc1_bn2;

    // Bottleneck: 10x10
    ConvLayer bot_conv1; // ch1 -> ch_bot
    InstanceNormLayer bot_bn1;
    ConvLayer bot_conv2; // ch_bot -> ch1
    InstanceNormLayer bot_bn2;

    // Decoder Level 1: 20x20
    ConvLayer dec1_conv1; // ch1*2 -> ch1 (ch1 upsample + ch1 skip)
    InstanceNormLayer dec1_bn1;
    ConvLayer dec1_conv2; // ch1 -> ch0
    InstanceNormLayer dec1_bn2;

    // Decoder Level 0: 40x40
    ConvLayer dec0_conv1; // ch0*2 -> ch0 (ch0 upsample + ch0 skip)
    InstanceNormLayer dec0_bn1;
    ConvLayer dec0_conv2; // ch0 -> 6 (output logits)
    // No BN or ReLU after final conv

    // --- Cached forward activations for backprop ---
    // Encoder 0
    Tensor enc0_z1, enc0_a1, enc0_n1; // conv -> instnorm -> relu
    Tensor enc0_z2, enc0_a2, enc0_n2;
    Tensor enc0_out; // after relu (skip connection)

    // MaxPool 0
    MaxPoolResult pool0;

    // Encoder 1
    Tensor enc1_z1, enc1_a1, enc1_n1;
    Tensor enc1_z2, enc1_a2, enc1_n2;
    Tensor enc1_out; // skip connection

    // MaxPool 1
    MaxPoolResult pool1;

    // Bottleneck
    Tensor bot_z1, bot_a1, bot_n1;
    Tensor bot_z2, bot_a2, bot_n2;
    Tensor bot_out;

    // Decoder 1
    Tensor up1, cat1;
    Tensor dec1_z1, dec1_a1, dec1_n1;
    Tensor dec1_z2, dec1_a2, dec1_n2;
    Tensor dec1_out;

    // Decoder 0
    Tensor up0, cat0;
    Tensor dec0_z1, dec0_a1, dec0_n1;
    Tensor dec0_z2; // final logits (before residual add)

    // Output
    Tensor logits_out; // after residual add
    Tensor softmax_out;

    int adam_t = 0;

    // Shared workspace for im2col/col2im (sized to max across all layers)
    std::vector<float> conv_workspace;

    UNet(int c0 = 32, int c1 = 64, int cb = 128)
        : ch0(c0), ch1(c1), ch_bot(cb),
          enc0_conv1(IN_CHANNELS, c0, 3, 3), enc0_bn1(c0),
          enc0_conv2(c0, c0, 3, 3),          enc0_bn2(c0),
          enc1_conv1(c0, c1, 3, 3),          enc1_bn1(c1),
          enc1_conv2(c1, c1, 3, 3),          enc1_bn2(c1),
          bot_conv1(c1, cb, 3, 3),           bot_bn1(cb),
          bot_conv2(cb, c1, 3, 3),           bot_bn2(c1),
          dec1_conv1(c1*2, c1, 3, 3),        dec1_bn1(c1),
          dec1_conv2(c1, c0, 3, 3),          dec1_bn2(c0),
          dec0_conv1(c0*2, c0, 3, 3),        dec0_bn1(c0),
          dec0_conv2(c0, NUM_CLASSES, 3, 3)
    {}

    void allocate_workspace(int H = GRID_H, int W = GRID_W)
    {
        // Find max col_buf size: in_ch * kH * kW * spatial_HW
        auto col_size = [](const ConvLayer& l, int h, int w) {
            return l.in_ch * l.kH * l.kW * h * w;
        };
        int max_col = 0;
        max_col = std::max(max_col, col_size(enc0_conv1, H, W));
        max_col = std::max(max_col, col_size(enc0_conv2, H, W));
        max_col = std::max(max_col, col_size(enc1_conv1, H/2, W/2));
        max_col = std::max(max_col, col_size(enc1_conv2, H/2, W/2));
        max_col = std::max(max_col, col_size(bot_conv1, H/4, W/4));
        max_col = std::max(max_col, col_size(bot_conv2, H/4, W/4));
        max_col = std::max(max_col, col_size(dec1_conv1, H/2, W/2));
        max_col = std::max(max_col, col_size(dec1_conv2, H/2, W/2));
        max_col = std::max(max_col, col_size(dec0_conv1, H, W));
        max_col = std::max(max_col, col_size(dec0_conv2, H, W));
        conv_workspace.resize(max_col);
    }

    void init(std::mt19937& rng)
    {
        enc0_conv1.init_he(rng); enc0_conv2.init_he(rng);
        enc1_conv1.init_he(rng); enc1_conv2.init_he(rng);
        bot_conv1.init_he(rng);  bot_conv2.init_he(rng);
        dec1_conv1.init_he(rng); dec1_conv2.init_he(rng);
        dec0_conv1.init_he(rng);
        // Final layer: Xavier init for better softmax behavior
        dec0_conv2.init_xavier(rng);
        adam_t = 0;
        allocate_workspace();
    }

    // --- Forward pass ---
    // training: use instance norm training mode (compute stats from input)
    // sim_channels: first 6 channels of input (for residual connection)
    Tensor forward(const Tensor& input, bool training)
    {
        int H = input.H, W = input.W;
        float* ws = conv_workspace.data();

        // === Encoder Level 0 (40x40) ===
        enc0_z1 = enc0_conv1.forward(input, ws, training);
        enc0_a1 = training ? enc0_bn1.forward_train(enc0_z1) : enc0_bn1.forward_infer(enc0_z1);
        relu_forward(enc0_a1, enc0_n1);

        enc0_z2 = enc0_conv2.forward(enc0_n1, ws, training);
        enc0_a2 = training ? enc0_bn2.forward_train(enc0_z2) : enc0_bn2.forward_infer(enc0_z2);
        relu_forward(enc0_a2, enc0_out); // skip0 = enc0_out (32 ch, 40x40)

        // MaxPool -> 20x20
        pool0 = maxpool2x2_forward(enc0_out);

        // === Encoder Level 1 (20x20) ===
        enc1_z1 = enc1_conv1.forward(pool0.output, ws, training);
        enc1_a1 = training ? enc1_bn1.forward_train(enc1_z1) : enc1_bn1.forward_infer(enc1_z1);
        relu_forward(enc1_a1, enc1_n1);

        enc1_z2 = enc1_conv2.forward(enc1_n1, ws, training);
        enc1_a2 = training ? enc1_bn2.forward_train(enc1_z2) : enc1_bn2.forward_infer(enc1_z2);
        relu_forward(enc1_a2, enc1_out); // skip1 = enc1_out (64 ch, 20x20)

        // MaxPool -> 10x10
        pool1 = maxpool2x2_forward(enc1_out);

        // === Bottleneck (10x10) ===
        bot_z1 = bot_conv1.forward(pool1.output, ws, training);
        bot_a1 = training ? bot_bn1.forward_train(bot_z1) : bot_bn1.forward_infer(bot_z1);
        relu_forward(bot_a1, bot_n1);

        bot_z2 = bot_conv2.forward(bot_n1, ws, training);
        bot_a2 = training ? bot_bn2.forward_train(bot_z2) : bot_bn2.forward_infer(bot_z2);
        relu_forward(bot_a2, bot_out); // 64 ch, 10x10

        // === Decoder Level 1 ===
        up1 = upsample2x2_forward(bot_out);       // 64 ch, 20x20
        cat1 = concat_channels(up1, enc1_out);     // 128 ch, 20x20

        dec1_z1 = dec1_conv1.forward(cat1, ws, training);
        dec1_a1 = training ? dec1_bn1.forward_train(dec1_z1) : dec1_bn1.forward_infer(dec1_z1);
        relu_forward(dec1_a1, dec1_n1);

        dec1_z2 = dec1_conv2.forward(dec1_n1, ws, training);
        dec1_a2 = training ? dec1_bn2.forward_train(dec1_z2) : dec1_bn2.forward_infer(dec1_z2);
        relu_forward(dec1_a2, dec1_out); // 32 ch, 20x20

        // === Decoder Level 0 ===
        up0 = upsample2x2_forward(dec1_out);      // 32 ch, 40x40
        cat0 = concat_channels(up0, enc0_out);     // 64 ch, 40x40

        dec0_z1 = dec0_conv1.forward(cat0, ws, training);
        dec0_a1 = training ? dec0_bn1.forward_train(dec0_z1) : dec0_bn1.forward_infer(dec0_z1);
        relu_forward(dec0_a1, dec0_n1);

        dec0_z2 = dec0_conv2.forward(dec0_n1, ws, training); // 6 ch, 40x40

        // === Residual add: add sim prediction channels (first 6 of input) ===
        logits_out = Tensor(NUM_CLASSES, H, W);
        for(int c = 0; c < NUM_CLASSES; c++)
            for(int y = 0; y < H; y++)
                for(int x = 0; x < W; x++)
                    logits_out.at(c, y, x) = dec0_z2.at(c, y, x) + input.at(c, y, x);

        // === Softmax ===
        softmax_forward(logits_out, softmax_out);

        return softmax_out;
    }

    // --- Backward pass ---
    // Returns KL divergence loss, accumulates all gradients
    float backward(const Tensor& input, const Tensor& target, const std::vector<float>& mask)
    {
        int H = softmax_out.H, W = softmax_out.W;
        float loss = 0;

        // Gradient of KL loss w.r.t. logits (softmax + KL combined)
        // d(KL)/d(logit_c) = out_c - target_c  for dynamic cells
        Tensor d_logits(NUM_CLASSES, H, W);

        for(int y = 0; y < H; y++)
        {
            for(int x = 0; x < W; x++)
            {
                if(mask[y * W + x] < 0.5f) continue;
                for(int c = 0; c < NUM_CLASSES; c++)
                {
                    float p = target.at(c, y, x);
                    float q = softmax_out.at(c, y, x);
                    if(p > 0)
                        loss += p * logf(p / fmaxf(q, 1e-10f));
                    d_logits.at(c, y, x) = softmax_out.at(c, y, x) - target.at(c, y, x);
                }
            }
        }

        // Residual add backward: gradient flows identically to both branches
        // d_dec0_z2 = d_logits (the UNet output branch)
        // d_input_residual = d_logits (the skip branch — but we don't need to propagate to input)
        Tensor& d_dec0_z2 = d_logits; // reuse

        // === Decoder Level 0 backward ===
        float* ws = conv_workspace.data();
        Tensor d_dec0_n1 = dec0_conv2.backward(d_dec0_z2, ws);
        Tensor d_dec0_a1 = relu_backward(d_dec0_n1, dec0_a1);
        Tensor d_dec0_z1 = dec0_bn1.backward(d_dec0_a1);
        Tensor d_cat0    = dec0_conv1.backward(d_dec0_z1, ws);

        // Split cat0 gradient: up0 (ch0) + enc0_out skip (ch0)
        Tensor d_up0, d_enc0_skip;
        split_gradient(d_cat0, ch0, d_up0, d_enc0_skip);

        Tensor d_dec1_out = upsample2x2_backward(d_up0);

        // === Decoder Level 1 backward ===
        Tensor d_dec1_a2 = relu_backward(d_dec1_out, dec1_a2);
        Tensor d_dec1_z2 = dec1_bn2.backward(d_dec1_a2);
        Tensor d_dec1_n1 = dec1_conv2.backward(d_dec1_z2, ws);
        Tensor d_dec1_a1 = relu_backward(d_dec1_n1, dec1_a1);
        Tensor d_dec1_z1 = dec1_bn1.backward(d_dec1_a1);
        Tensor d_cat1    = dec1_conv1.backward(d_dec1_z1, ws);

        // Split cat1 gradient: up1 (ch1) + enc1_out skip (ch1)
        Tensor d_up1, d_enc1_skip;
        split_gradient(d_cat1, ch1, d_up1, d_enc1_skip);

        Tensor d_bot_out = upsample2x2_backward(d_up1);

        // === Bottleneck backward ===
        Tensor d_bot_a2 = relu_backward(d_bot_out, bot_a2);
        Tensor d_bot_z2 = bot_bn2.backward(d_bot_a2);
        Tensor d_bot_n1 = bot_conv2.backward(d_bot_z2, ws);
        Tensor d_bot_a1 = relu_backward(d_bot_n1, bot_a1);
        Tensor d_bot_z1 = bot_bn1.backward(d_bot_a1);
        Tensor d_pool1_out = bot_conv1.backward(d_bot_z1, ws);

        // === MaxPool 1 backward ===
        Tensor d_enc1_out_pool = maxpool2x2_backward(d_pool1_out, pool1.indices,
                                                      enc1_out.H, enc1_out.W);
        // Add skip gradient
        for(int i = 0; i < d_enc1_out_pool.size(); i++)
            d_enc1_out_pool.data[i] += d_enc1_skip.data[i];

        // === Encoder Level 1 backward ===
        Tensor d_enc1_a2 = relu_backward(d_enc1_out_pool, enc1_a2);
        Tensor d_enc1_z2 = enc1_bn2.backward(d_enc1_a2);
        Tensor d_enc1_n1 = enc1_conv2.backward(d_enc1_z2, ws);
        Tensor d_enc1_a1 = relu_backward(d_enc1_n1, enc1_a1);
        Tensor d_enc1_z1 = enc1_bn1.backward(d_enc1_a1);
        Tensor d_pool0_out = enc1_conv1.backward(d_enc1_z1, ws);

        // === MaxPool 0 backward ===
        Tensor d_enc0_out_pool = maxpool2x2_backward(d_pool0_out, pool0.indices,
                                                      enc0_out.H, enc0_out.W);
        // Add skip gradient
        for(int i = 0; i < d_enc0_out_pool.size(); i++)
            d_enc0_out_pool.data[i] += d_enc0_skip.data[i];

        // === Encoder Level 0 backward ===
        Tensor d_enc0_a2 = relu_backward(d_enc0_out_pool, enc0_a2);
        Tensor d_enc0_z2 = enc0_bn2.backward(d_enc0_a2);
        Tensor d_enc0_n1 = enc0_conv2.backward(d_enc0_z2, ws);
        Tensor d_enc0_a1 = relu_backward(d_enc0_n1, enc0_a1);
        Tensor d_enc0_z1 = enc0_bn1.backward(d_enc0_a1);
        // We don't need d_input for the first layer
        enc0_conv1.backward(d_enc0_z1, ws);

        return loss;
    }

    void zero_grad()
    {
        enc0_conv1.zero_grad(); enc0_bn1.zero_grad();
        enc0_conv2.zero_grad(); enc0_bn2.zero_grad();
        enc1_conv1.zero_grad(); enc1_bn1.zero_grad();
        enc1_conv2.zero_grad(); enc1_bn2.zero_grad();
        bot_conv1.zero_grad();  bot_bn1.zero_grad();
        bot_conv2.zero_grad();  bot_bn2.zero_grad();
        dec1_conv1.zero_grad(); dec1_bn1.zero_grad();
        dec1_conv2.zero_grad(); dec1_bn2.zero_grad();
        dec0_conv1.zero_grad(); dec0_bn1.zero_grad();
        dec0_conv2.zero_grad();
    }

    // --- Adam update helpers ---

    void adam_update_conv(ConvLayer& layer, float lr, float scale,
                          float beta1, float beta2, float eps,
                          float bc1, float bc2)
    {
        int nw = (int)layer.weights.size();
        for(int i = 0; i < nw; i++)
        {
            float gi = layer.dweights[i] * scale;
            layer.mw[i] = beta1 * layer.mw[i] + (1 - beta1) * gi;
            layer.vw[i] = beta2 * layer.vw[i] + (1 - beta2) * gi * gi;
            float mh = layer.mw[i] / bc1;
            float vh = layer.vw[i] / bc2;
            layer.weights[i] -= lr * mh / (sqrtf(vh) + eps);
        }
        int nb = (int)layer.bias.size();
        for(int i = 0; i < nb; i++)
        {
            float gi = layer.dbias[i] * scale;
            layer.mb[i] = beta1 * layer.mb[i] + (1 - beta1) * gi;
            layer.vb[i] = beta2 * layer.vb[i] + (1 - beta2) * gi * gi;
            float mh = layer.mb[i] / bc1;
            float vh = layer.vb[i] / bc2;
            layer.bias[i] -= lr * mh / (sqrtf(vh) + eps);
        }
    }

    void adam_update_instnorm(InstanceNormLayer& layer, float lr, float scale,
                               float beta1, float beta2, float eps,
                               float bc1, float bc2)
    {
        int nc = layer.channels;
        for(int i = 0; i < nc; i++)
        {
            // gamma
            float gg = layer.dgamma[i] * scale;
            layer.mg[i] = beta1 * layer.mg[i] + (1 - beta1) * gg;
            layer.vg[i] = beta2 * layer.vg[i] + (1 - beta2) * gg * gg;
            float mh = layer.mg[i] / bc1;
            float vh = layer.vg[i] / bc2;
            layer.gamma[i] -= lr * mh / (sqrtf(vh) + eps);

            // beta
            float gb = layer.dbeta[i] * scale;
            layer.mbe[i] = beta1 * layer.mbe[i] + (1 - beta1) * gb;
            layer.vbe[i] = beta2 * layer.vbe[i] + (1 - beta2) * gb * gb;
            mh = layer.mbe[i] / bc1;
            vh = layer.vbe[i] / bc2;
            layer.beta[i] -= lr * mh / (sqrtf(vh) + eps);
        }
    }

    void update(float lr, int batch_sz,
                float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
    {
        adam_t++;
        float scale = 1.0f / batch_sz;
        float bc1 = 1.0f - powf(beta1, adam_t);
        float bc2 = 1.0f - powf(beta2, adam_t);

        adam_update_conv(enc0_conv1, lr, scale, beta1, beta2, eps, bc1, bc2);
        adam_update_instnorm(enc0_bn1, lr, scale, beta1, beta2, eps, bc1, bc2);
        adam_update_conv(enc0_conv2, lr, scale, beta1, beta2, eps, bc1, bc2);
        adam_update_instnorm(enc0_bn2, lr, scale, beta1, beta2, eps, bc1, bc2);

        adam_update_conv(enc1_conv1, lr, scale, beta1, beta2, eps, bc1, bc2);
        adam_update_instnorm(enc1_bn1, lr, scale, beta1, beta2, eps, bc1, bc2);
        adam_update_conv(enc1_conv2, lr, scale, beta1, beta2, eps, bc1, bc2);
        adam_update_instnorm(enc1_bn2, lr, scale, beta1, beta2, eps, bc1, bc2);

        adam_update_conv(bot_conv1, lr, scale, beta1, beta2, eps, bc1, bc2);
        adam_update_instnorm(bot_bn1, lr, scale, beta1, beta2, eps, bc1, bc2);
        adam_update_conv(bot_conv2, lr, scale, beta1, beta2, eps, bc1, bc2);
        adam_update_instnorm(bot_bn2, lr, scale, beta1, beta2, eps, bc1, bc2);

        adam_update_conv(dec1_conv1, lr, scale, beta1, beta2, eps, bc1, bc2);
        adam_update_instnorm(dec1_bn1, lr, scale, beta1, beta2, eps, bc1, bc2);
        adam_update_conv(dec1_conv2, lr, scale, beta1, beta2, eps, bc1, bc2);
        adam_update_instnorm(dec1_bn2, lr, scale, beta1, beta2, eps, bc1, bc2);

        adam_update_conv(dec0_conv1, lr, scale, beta1, beta2, eps, bc1, bc2);
        adam_update_instnorm(dec0_bn1, lr, scale, beta1, beta2, eps, bc1, bc2);
        adam_update_conv(dec0_conv2, lr, scale, beta1, beta2, eps, bc1, bc2);
    }
};

// ============================================================
// Data Augmentation: 8 variants (4 rotations × 2 flips)
// ============================================================

// Rotate a tensor 90° clockwise: (y,x) -> (x, H-1-y)
Tensor rotate90(const Tensor& t)
{
    int C = t.C, H = t.H, W = t.W;
    // After 90° CW rotation of HxW grid, output is WxH
    // But our grids are square (40x40) so output is still HxW
    Tensor out(C, W, H);
    for(int c = 0; c < C; c++)
        for(int y = 0; y < H; y++)
            for(int x = 0; x < W; x++)
                out.at(c, x, H - 1 - y) = t.at(c, y, x);
    return out;
}

// Horizontal flip: (y,x) -> (y, W-1-x)
Tensor hflip(const Tensor& t)
{
    int C = t.C, H = t.H, W = t.W;
    Tensor out(C, H, W);
    for(int c = 0; c < C; c++)
        for(int y = 0; y < H; y++)
            for(int x = 0; x < W; x++)
                out.at(c, y, W - 1 - x) = t.at(c, y, x);
    return out;
}

// Apply the same transform to a mask (H*W flat vector, same grid dims)
std::vector<float> rotate90_mask(const std::vector<float>& mask, int H, int W)
{
    std::vector<float> out(W * H);
    for(int y = 0; y < H; y++)
        for(int x = 0; x < W; x++)
            out[x * H + (H - 1 - y)] = mask[y * W + x];
    return out;
}

std::vector<float> hflip_mask(const std::vector<float>& mask, int H, int W)
{
    std::vector<float> out(H * W);
    for(int y = 0; y < H; y++)
        for(int x = 0; x < W; x++)
            out[y * W + (W - 1 - x)] = mask[y * W + x];
    return out;
}

// ============================================================
// I/O: ASTP binary format (identical to original)
// ============================================================

Tensor read_prediction(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if(!f) return {};

    char magic[4]; f.read(magic, 4);
    if(memcmp(magic, "ASTP", 4) != 0) { printf("Bad magic in %s\n", path.c_str()); return {}; }
    uint16_t ver; f.read((char*)&ver, 2);
    int32_t round, seed, w, h;
    f.read((char*)&round, 4);
    f.read((char*)&seed, 4);
    f.read((char*)&w, 4);
    f.read((char*)&h, 4);

    Tensor t(NUM_CLASSES, h, w);
    for(int y = 0; y < h; y++)
        for(int x = 0; x < w; x++)
            for(int c = 0; c < NUM_CLASSES; c++)
                f.read((char*)&t.at(c, y, x), 4);
    return t;
}

void write_prediction(const std::string& path, int round, int seed, int W, int H,
                      const Tensor& pred)
{
    std::ofstream f(path, std::ios::binary);
    f.write("ASTP", 4);
    uint16_t ver = 1; f.write((char*)&ver, 2);
    f.write((char*)&round, 4);
    f.write((char*)&seed, 4);
    f.write((char*)&W, 4);
    f.write((char*)&H, 4);
    for(int y = 0; y < H; y++)
        for(int x = 0; x < W; x++)
            for(int c = 0; c < NUM_CLASSES; c++)
            {
                float v = pred.at(c, y, x);
                f.write((char*)&v, 4);
            }
    f.close();
}

void write_prediction_vec(const std::string& path, int round, int seed, int W, int H,
                          const std::vector<std::vector<std::vector<float>>>& pred)
{
    std::ofstream f(path, std::ios::binary);
    f.write("ASTP", 4);
    uint16_t ver = 1; f.write((char*)&ver, 2);
    f.write((char*)&round, 4);
    f.write((char*)&seed, 4);
    f.write((char*)&W, 4);
    f.write((char*)&H, 4);
    for(int y = 0; y < H; y++)
        for(int x = 0; x < W; x++)
            for(int c = 0; c < NUM_CLASSES; c++)
                f.write((char*)&pred[y][x][c], 4);
    f.close();
}

// Load grid from grids.bin
bool load_grid(const std::string& path, int want_round, int want_seed,
               std::vector<std::vector<int>>& grid, int& W, int& H)
{
    std::ifstream f(path, std::ios::binary);
    if(!f) { printf("Error: cannot open %s\n", path.c_str()); return false; }

    char magic[4]; f.read(magic, 4);
    uint16_t ver; f.read((char*)&ver, 2);
    uint32_t count; f.read((char*)&count, 4);

    for(uint32_t i = 0; i < count; i++)
    {
        int32_t round, seed, w, h;
        f.read((char*)&round, 4);
        f.read((char*)&seed, 4);
        f.read((char*)&w, 4);
        f.read((char*)&h, 4);

        if(round == want_round && seed == want_seed)
        {
            W = w; H = h;
            grid.resize(H);
            for(int y = 0; y < H; y++)
            {
                grid[y].resize(W);
                for(int x = 0; x < W; x++)
                    f.read((char*)&grid[y][x], 4);
            }
            return true;
        }
        else
        {
            f.seekg((int64_t)w * h * 4, std::ios::cur);
        }
    }
    printf("Grid not found: round=%d seed=%d\n", want_round, want_seed);
    return false;
}

// Load ground truth
bool load_ground_truth(const std::string& path, int want_round, int want_seed,
                       Tensor& gt, int& W, int& H)
{
    std::ifstream f(path, std::ios::binary);
    if(!f) return false;

    char magic[4]; f.read(magic, 4);
    uint16_t ver; f.read((char*)&ver, 2);
    uint32_t count; f.read((char*)&count, 4);

    for(uint32_t i = 0; i < count; i++)
    {
        int32_t round, seed, w, h;
        f.read((char*)&round, 4);
        f.read((char*)&seed, 4);
        f.read((char*)&w, 4);
        f.read((char*)&h, 4);

        if(round == want_round && seed == want_seed)
        {
            W = w; H = h;
            gt = Tensor(NUM_CLASSES, H, W);
            for(int y = 0; y < H; y++)
                for(int x = 0; x < W; x++)
                    for(int c = 0; c < NUM_CLASSES; c++)
                        f.read((char*)&gt.at(c, y, x), 4);
            return true;
        }
        else
        {
            f.seekg((int64_t)w * h * NUM_CLASSES * 4, std::ios::cur);
        }
    }
    return false;
}

// Enumerate all (round, seed) pairs in ground_truth.bin
std::vector<std::pair<int,int>> enumerate_ground_truth(const std::string& path)
{
    std::vector<std::pair<int,int>> entries;
    std::ifstream f(path, std::ios::binary);
    if(!f) return entries;

    char magic[4]; f.read(magic, 4);
    uint16_t ver; f.read((char*)&ver, 2);
    uint32_t count; f.read((char*)&count, 4);

    for(uint32_t i = 0; i < count; i++)
    {
        int32_t round, seed, w, h;
        f.read((char*)&round, 4);
        f.read((char*)&seed, 4);
        f.read((char*)&w, 4);
        f.read((char*)&h, 4);
        entries.push_back({round, seed});
        f.seekg((int64_t)w * h * NUM_CLASSES * 4, std::ios::cur);
    }
    return entries;
}

// Build CNN input tensor (13 x H x W)
Tensor build_input(const Tensor& sim_pred, const std::vector<std::vector<int>>& grid,
                   int H, int W)
{
    Tensor input(IN_CHANNELS, H, W);

    for(int y = 0; y < H; y++)
    {
        for(int x = 0; x < W; x++)
        {
            // Channels 0-5: sim prediction probabilities
            for(int c = 0; c < NUM_CLASSES; c++)
                input.at(c, y, x) = sim_pred.at(c, y, x);

            // Channels 6-11: initial terrain one-hot
            int tc = terrain_to_class(grid[y][x]);
            for(int c = 0; c < NUM_CLASSES; c++)
                input.at(6 + c, y, x) = (c == tc) ? 1.0f : 0.0f;

            // Channel 12: is_land
            input.at(12, y, x) = is_land_code(grid[y][x]) ? 1.0f : 0.0f;
        }
    }
    return input;
}

// Build dynamic cell mask
std::vector<float> build_mask(const Tensor& gt, int H, int W)
{
    std::vector<float> mask(H * W, 0.0f);
    for(int y = 0; y < H; y++)
    {
        for(int x = 0; x < W; x++)
        {
            double ent = 0;
            for(int c = 0; c < NUM_CLASSES; c++)
            {
                float p = gt.at(c, y, x);
                if(p > 0) ent -= p * log(p);
            }
            if(ent > 1e-6) mask[y * W + x] = 1.0f;
        }
    }
    return mask;
}

// Floor probabilities and renormalize
void apply_floor_tensor(Tensor& pred, float floor_val = 0.005f)
{
    int H = pred.H, W = pred.W;
    for(int y = 0; y < H; y++)
    {
        for(int x = 0; x < W; x++)
        {
            for(int c = 0; c < NUM_CLASSES; c++)
                if(pred.at(c, y, x) < floor_val) pred.at(c, y, x) = floor_val;
            float total = 0;
            for(int c = 0; c < NUM_CLASSES; c++) total += pred.at(c, y, x);
            for(int c = 0; c < NUM_CLASSES; c++) pred.at(c, y, x) /= total;
        }
    }
}

// --- Training sample ---

struct Sample
{
    int round, seed;
    Tensor input;  // 13 x H x W
    Tensor target; // 6 x H x W
    std::vector<float> mask; // H * W
};

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[])
{
    if(argc < 5)
    {
        printf("Usage: cnn <grids.bin> <ground_truth.bin> <round> <seed> [prediction.bin]\n"
               "  [--exclude <round>] [--epochs N] [--lr F]\n"
               "  [--sim-dir path] [--threads N] [--channels N]\n");
        return 1;
    }

    std::string grids_path = argv[1];
    std::string gt_path    = argv[2];
    int target_round = std::stoi(argv[3]);
    int target_seed  = std::stoi(argv[4]);

    std::string out_path;
    int exclude_round = -1;
    int epochs = 50;
    float lr = 0.001f;
    std::string sim_dir = "data/";
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    int base_channels = 16; // default slim for fast Mac training

    for(int i = 5; i < argc; i++)
    {
        std::string arg = argv[i];
        if(arg == "--exclude" && i+1 < argc) exclude_round = std::stoi(argv[++i]);
        else if(arg == "--epochs" && i+1 < argc) epochs = std::stoi(argv[++i]);
        else if(arg == "--lr" && i+1 < argc) lr = std::stof(argv[++i]);
        else if(arg == "--sim-dir" && i+1 < argc) sim_dir = argv[++i];
        else if(arg == "--threads" && i+1 < argc) n_threads = std::stoi(argv[++i]);
        else if(arg == "--channels" && i+1 < argc) base_channels = std::stoi(argv[++i]);
        else if(out_path.empty()) out_path = arg;
    }

    // Ensure sim_dir ends with /
    if(!sim_dir.empty() && sim_dir.back() != '/') sim_dir += '/';

    // --- Enumerate all ground truth entries ---
    auto gt_entries = enumerate_ground_truth(gt_path);
    printf("Ground truth entries: %d\n", (int)gt_entries.size());

    // --- Build training data ---
    std::vector<Sample> base_samples;
    int excluded = 0, loaded = 0, skipped_no_sim = 0;

    for(auto& [r, s] : gt_entries)
    {
        if(exclude_round >= 0 && r == exclude_round)
        {
            excluded++;
            continue;
        }

        // Load sim prediction
        char sim_name[256];
        snprintf(sim_name, sizeof(sim_name), "%spred_sim_r%d_s%d.bin",
                 sim_dir.c_str(), r, s);
        Tensor sim_pred = read_prediction(sim_name);
        if(sim_pred.size() == 0)
        {
            skipped_no_sim++;
            continue;
        }

        // Load initial grid
        std::vector<std::vector<int>> grid;
        int W, H;
        if(!load_grid(grids_path, r, s, grid, W, H))
        {
            skipped_no_sim++;
            continue;
        }

        // Load ground truth
        Tensor gt;
        int gtW, gtH;
        if(!load_ground_truth(gt_path, r, s, gt, gtW, gtH))
        {
            skipped_no_sim++;
            continue;
        }

        Sample sample;
        sample.round = r;
        sample.seed = s;
        sample.input = build_input(sim_pred, grid, H, W);
        sample.target = gt;
        sample.mask = build_mask(gt, H, W);

        base_samples.push_back(std::move(sample));
        loaded++;
    }

    printf("Base training samples: %d loaded", loaded);
    if(excluded > 0) printf(", %d excluded (round %d)", excluded, exclude_round);
    if(skipped_no_sim > 0) printf(", %d skipped (no sim prediction)", skipped_no_sim);
    printf("\n");

    if(base_samples.empty())
    {
        printf("Error: no training data available\n");
        return 1;
    }

    // --- Data augmentation: 8× (4 rotations × 2 flips) ---
    printf("Applying 8x data augmentation (4 rotations x 2 flips)...\n");
    std::vector<Sample> train_samples;
    train_samples.reserve(base_samples.size() * 8);

    for(auto& base : base_samples)
    {
        int H = base.input.H, W = base.input.W;

        // Generate 4 rotations × 2 (original + hflip) = 8 variants
        Tensor inp_r0 = base.input;
        Tensor tgt_r0 = base.target;
        std::vector<float> msk_r0 = base.mask;

        for(int rot = 0; rot < 4; rot++)
        {
            Tensor inp = inp_r0;
            Tensor tgt = tgt_r0;
            std::vector<float> msk = msk_r0;

            // Original (rotated)
            {
                Sample s;
                s.round = base.round; s.seed = base.seed;
                s.input = inp; s.target = tgt; s.mask = msk;
                train_samples.push_back(std::move(s));
            }

            // Horizontal flip
            {
                Sample s;
                s.round = base.round; s.seed = base.seed;
                s.input = hflip(inp);
                s.target = hflip(tgt);
                s.mask = hflip_mask(msk, inp.H, inp.W);
                train_samples.push_back(std::move(s));
            }

            // Rotate for next iteration
            if(rot < 3)
            {
                inp_r0 = rotate90(inp_r0);
                tgt_r0 = rotate90(tgt_r0);
                msk_r0 = rotate90_mask(msk_r0, inp_r0.H, inp_r0.W);
                // After rotation of a square grid, H and W stay the same
            }
        }
    }

    printf("Augmented training samples: %d\n", (int)train_samples.size());

    // --- Initialize U-Net ---
    int c0 = base_channels, c1 = base_channels * 2, cb = base_channels * 4;
    UNet net(c0, c1, cb);
    std::mt19937 rng(42);
    net.init(rng);

    // Count parameters
    auto count_conv_params = [](const ConvLayer& l) { return (int)(l.weights.size() + l.bias.size()); };
    auto count_bn_params = [](const InstanceNormLayer& l) { return l.channels * 2; };
    int total_params = 0;
    total_params += count_conv_params(net.enc0_conv1) + count_bn_params(net.enc0_bn1);
    total_params += count_conv_params(net.enc0_conv2) + count_bn_params(net.enc0_bn2);
    total_params += count_conv_params(net.enc1_conv1) + count_bn_params(net.enc1_bn1);
    total_params += count_conv_params(net.enc1_conv2) + count_bn_params(net.enc1_bn2);
    total_params += count_conv_params(net.bot_conv1)  + count_bn_params(net.bot_bn1);
    total_params += count_conv_params(net.bot_conv2)  + count_bn_params(net.bot_bn2);
    total_params += count_conv_params(net.dec1_conv1) + count_bn_params(net.dec1_bn1);
    total_params += count_conv_params(net.dec1_conv2) + count_bn_params(net.dec1_bn2);
    total_params += count_conv_params(net.dec0_conv1) + count_bn_params(net.dec0_bn1);
    total_params += count_conv_params(net.dec0_conv2);

    printf("U-Net Architecture (channels: %d/%d/%d):\n", c0, c1, cb);
    printf("  Encoder:    Conv(13->%d)->IN->ReLU->Conv(%d->%d)->IN->ReLU -> Pool\n", c0, c0, c0);
    printf("              Conv(%d->%d)->IN->ReLU->Conv(%d->%d)->IN->ReLU -> Pool\n", c0, c1, c1, c1);
    printf("  Bottleneck: Conv(%d->%d)->IN->ReLU->Conv(%d->%d)->IN->ReLU\n", c1, cb, cb, c1);
    printf("  Decoder:    Up->Cat(%d)->Conv(%d->%d)->IN->ReLU->Conv(%d->%d)->IN->ReLU\n", c1*2, c1*2, c1, c1, c0);
    printf("              Up->Cat(%d)->Conv(%d->%d)->IN->ReLU->Conv(%d->6) + Residual\n", c0*2, c0*2, c0, c0);
    printf("  Total parameters: %d\n", total_params);
    printf("  Threads: %d\n", n_threads);
    printf("Epochs: %d, Adam LR: %.4f, Augmented samples: %d\n",
           epochs, lr, (int)train_samples.size());

    // --- Create per-thread UNet clones for parallel training ---
    std::vector<UNet> thread_nets;
    if(n_threads > 1) {
        thread_nets.resize(n_threads, UNet(c0, c1, cb));
        for (auto& tnet : thread_nets) tnet.allocate_workspace();
        // They don't need their own weights - we'll copy before each batch
    }

    // --- Train with mini-batch parallelism ---
    int N = (int)train_samples.size();
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    int batch_size = std::max(1, std::min(n_threads, N));

    for(int ep = 0; ep < epochs; ep++)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        double epoch_loss = 0;
        int total_dynamic = 0;

        for(int batch_start = 0; batch_start < N; batch_start += batch_size)
        {
            int batch_end = std::min(batch_start + batch_size, N);
            int this_batch = batch_end - batch_start;

            if(n_threads <= 1 || this_batch == 1)
            {
                // Sequential path (Mac single-threaded or single sample)
                for(int i = batch_start; i < batch_end; i++)
                {
                    auto& s = train_samples[indices[i]];
                    net.zero_grad();
                    net.forward(s.input, true);
                    float loss = net.backward(s.input, s.target, s.mask);
                    int dyn = 0;
                    for(float m : s.mask) if(m > 0.5f) dyn++;
                    total_dynamic += dyn;
                    net.update(lr, std::max(dyn, 1));
                    epoch_loss += loss;
                }
            }
            else
            {
                // Parallel path: each thread processes one sample into its own net
                std::vector<double> t_loss(this_batch, 0);
                std::vector<int> t_dyn(this_batch, 0);
                std::vector<std::thread> threads;

                for(int t = 0; t < this_batch; t++)
                {
                    threads.emplace_back([&, t]() {
                        auto& tnet = thread_nets[t];
                        // Copy current weights to thread net
                        // (we copy all layer weights - shared read is fine for forward)
                        auto copy_conv_w = [](ConvLayer& dst, const ConvLayer& src) {
                            dst.weights = src.weights;
                            dst.bias = src.bias;
                        };
                        auto copy_bn_w = [](InstanceNormLayer& dst, const InstanceNormLayer& src) {
                            dst.gamma = src.gamma;
                            dst.beta = src.beta;
                            dst.running_mean = src.running_mean;
                            dst.running_var = src.running_var;
                            dst.has_running_stats = src.has_running_stats;
                        };
                        copy_conv_w(tnet.enc0_conv1, net.enc0_conv1); copy_bn_w(tnet.enc0_bn1, net.enc0_bn1);
                        copy_conv_w(tnet.enc0_conv2, net.enc0_conv2); copy_bn_w(tnet.enc0_bn2, net.enc0_bn2);
                        copy_conv_w(tnet.enc1_conv1, net.enc1_conv1); copy_bn_w(tnet.enc1_bn1, net.enc1_bn1);
                        copy_conv_w(tnet.enc1_conv2, net.enc1_conv2); copy_bn_w(tnet.enc1_bn2, net.enc1_bn2);
                        copy_conv_w(tnet.bot_conv1, net.bot_conv1);   copy_bn_w(tnet.bot_bn1, net.bot_bn1);
                        copy_conv_w(tnet.bot_conv2, net.bot_conv2);   copy_bn_w(tnet.bot_bn2, net.bot_bn2);
                        copy_conv_w(tnet.dec1_conv1, net.dec1_conv1); copy_bn_w(tnet.dec1_bn1, net.dec1_bn1);
                        copy_conv_w(tnet.dec1_conv2, net.dec1_conv2); copy_bn_w(tnet.dec1_bn2, net.dec1_bn2);
                        copy_conv_w(tnet.dec0_conv1, net.dec0_conv1); copy_bn_w(tnet.dec0_bn1, net.dec0_bn1);
                        copy_conv_w(tnet.dec0_conv2, net.dec0_conv2);

                        auto& s = train_samples[indices[batch_start + t]];
                        tnet.zero_grad();
                        tnet.forward(s.input, true);
                        t_loss[t] = tnet.backward(s.input, s.target, s.mask);
                        int dyn = 0;
                        for(float m : s.mask) if(m > 0.5f) dyn++;
                        t_dyn[t] = dyn;
                    });
                }
                for(auto& th : threads) th.join();

                // Accumulate gradients from all threads into main net
                net.zero_grad();
                auto accum_conv = [](ConvLayer& dst, const ConvLayer& src) {
                    for(size_t i = 0; i < dst.dweights.size(); i++) dst.dweights[i] += src.dweights[i];
                    for(size_t i = 0; i < dst.dbias.size(); i++) dst.dbias[i] += src.dbias[i];
                };
                auto accum_bn = [](InstanceNormLayer& dst, const InstanceNormLayer& src) {
                    for(int i = 0; i < dst.channels; i++) { dst.dgamma[i] += src.dgamma[i]; dst.dbeta[i] += src.dbeta[i]; }
                };
                int batch_dynamic = 0;
                for(int t = 0; t < this_batch; t++)
                {
                    auto& tnet = thread_nets[t];
                    accum_conv(net.enc0_conv1, tnet.enc0_conv1); accum_bn(net.enc0_bn1, tnet.enc0_bn1);
                    accum_conv(net.enc0_conv2, tnet.enc0_conv2); accum_bn(net.enc0_bn2, tnet.enc0_bn2);
                    accum_conv(net.enc1_conv1, tnet.enc1_conv1); accum_bn(net.enc1_bn1, tnet.enc1_bn1);
                    accum_conv(net.enc1_conv2, tnet.enc1_conv2); accum_bn(net.enc1_bn2, tnet.enc1_bn2);
                    accum_conv(net.bot_conv1, tnet.bot_conv1);   accum_bn(net.bot_bn1, tnet.bot_bn1);
                    accum_conv(net.bot_conv2, tnet.bot_conv2);   accum_bn(net.bot_bn2, tnet.bot_bn2);
                    accum_conv(net.dec1_conv1, tnet.dec1_conv1); accum_bn(net.dec1_bn1, tnet.dec1_bn1);
                    accum_conv(net.dec1_conv2, tnet.dec1_conv2); accum_bn(net.dec1_bn2, tnet.dec1_bn2);
                    accum_conv(net.dec0_conv1, tnet.dec0_conv1); accum_bn(net.dec0_bn1, tnet.dec0_bn1);
                    accum_conv(net.dec0_conv2, tnet.dec0_conv2);
                    epoch_loss += t_loss[t];
                    batch_dynamic += t_dyn[t];
                    total_dynamic += t_dyn[t];
                }

                // Update with this batch's gradients only
                net.update(lr, std::max(batch_dynamic, 1));
            }
        }

        printf("  Epoch %3d/%d  loss=%.6f  (avg per dynamic cell: %.6f)\n",
               ep + 1, epochs, epoch_loss,
               total_dynamic > 0 ? epoch_loss / total_dynamic : 0.0);
    }

    // --- Predict target round+seed ---
    printf("\nPredicting round %d, seed %d\n", target_round, target_seed);

    // Load sim prediction for target
    char sim_name[256];
    snprintf(sim_name, sizeof(sim_name), "%spred_sim_r%d_s%d.bin",
             sim_dir.c_str(), target_round, target_seed);
    Tensor sim_pred = read_prediction(sim_name);
    if(sim_pred.size() == 0)
    {
        printf("Error: cannot load sim prediction %s\n", sim_name);
        return 1;
    }

    // Load initial grid
    std::vector<std::vector<int>> grid;
    int W, H;
    if(!load_grid(grids_path, target_round, target_seed, grid, W, H))
        return 1;
    printf("Grid [r%d.s%d]: %dx%d\n", target_round, target_seed, W, H);

    // Build input and run forward (inference mode)
    Tensor input = build_input(sim_pred, grid, H, W);
    // Use training mode for inference too — instance norm computes per-sample
    // spatial stats, so running stats aren't needed (and may not be updated
    // correctly when training with multiple threads)
    Tensor prediction = net.forward(input, /*training=*/true);

    // Apply probability floor
    apply_floor_tensor(prediction, 0.005f);

    // --- Validate against ground truth ---
    Tensor gt;
    int gtW, gtH;
    if(load_ground_truth(gt_path, target_round, target_seed, gt, gtW, gtH))
    {
        double total_entropy = 0, total_weighted_kl = 0;
        int dynamic_cells = 0;

        for(int y = 0; y < H && y < gtH; y++)
        {
            for(int x = 0; x < W && x < gtW; x++)
            {
                double ent = 0;
                for(int c = 0; c < NUM_CLASSES; c++)
                {
                    float p = gt.at(c, y, x);
                    if(p > 0) ent -= p * log(p);
                }
                if(ent < 1e-6) continue;
                dynamic_cells++;
                total_entropy += ent;

                double kl = 0;
                for(int c = 0; c < NUM_CLASSES; c++)
                {
                    float p = gt.at(c, y, x);
                    float q = prediction.at(c, y, x);
                    if(p > 0) kl += p * log(p / std::max(q, 1e-10f));
                }
                total_weighted_kl += ent * kl;
            }
        }

        double weighted_kl = total_entropy > 0 ? total_weighted_kl / total_entropy : 0;
        double score = std::max(0.0, std::min(100.0, 100.0 * exp(-3.0 * weighted_kl)));
        printf("\nValidation vs ground truth:\n");
        printf("  Dynamic cells: %d\n", dynamic_cells);
        printf("  Weighted KL:   %.6f\n", weighted_kl);
        printf("  Score:         %.2f / 100\n", score);
    }

    // --- Write output ---
    if(!out_path.empty())
    {
        // Write in standard ASTP format (y, x, c order)
        std::vector<std::vector<std::vector<float>>> pred_vec(H,
            std::vector<std::vector<float>>(W, std::vector<float>(NUM_CLASSES)));
        for(int y = 0; y < H; y++)
            for(int x = 0; x < W; x++)
                for(int c = 0; c < NUM_CLASSES; c++)
                    pred_vec[y][x][c] = prediction.at(c, y, x);

        write_prediction_vec(out_path, target_round, target_seed, W, H, pred_vec);
        printf("Written to %s\n", out_path.c_str());
    }

    return 0;
}
