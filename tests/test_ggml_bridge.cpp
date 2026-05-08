#include "models/ggml_bridge.h"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    vlacpp::Tensor weight;
    weight.shape = {2, 3};
    weight.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    vlacpp::Tensor bias;
    bias.shape = {2};
    bias.data = {0.5f, -1.0f};
    const std::vector<float> input = {10.0f, 20.0f, 30.0f};
    std::vector<float> output;

    vlacpp::ggml_linear(weight, bias, input, output, 1);
    if (output.size() != 2) {
        std::cerr << "unexpected output size\n";
        return 1;
    }
    if (std::fabs(output[0] - 140.5f) > 1.0e-5f || std::fabs(output[1] - 319.0f) > 1.0e-5f) {
        std::cerr << "unexpected ggml linear result: " << output[0] << ", " << output[1] << "\n";
        return 1;
    }
    return 0;
}
