import torch
import torch.ao.quantization
import torch.nn.functional as F
import sys
import numpy as np
import ast
from torch.ao.nn.quantized import functional as qF


sys.path.append('../ML')  # noqa
from CNN import *  # noqa
from Quantized import *  # noqa
from save_to_txt import *  # noqa
# 读取参数文件的辅助函数


def read_txt(file_path):
    with open("save_activation/layer_weight/" + file_path, "r") as f:
        content = f.read()
        data = ast.literal_eval(content)
        tensor_data = torch.tensor(data, dtype=torch.float32)
        f.close()
    return tensor_data


def linearqunat(input, input_scale, input_zp, weight, weight_scale, weight_zp, output_scale, output_zp, bias, za: torch.Tensor) -> torch.Tensor:
    scale_param = input_scale * weight_scale/output_scale
    bias_q = bias/(weight_scale*input_scale)
    shift = 2 ** (-32)
    m = input_scale * weight_scale/output_scale
    m_0 = m/shift
    print(-za + bias_q)
    mult = torch.matmul(input, weight.t())
    out = torch.round(shift * m_0 * (mult-za + bias_q)) + output_zp
    return out


def Conv2dquant(quant_input: torch.Tensor, input_scale: torch.Tensor, input_zp, weight_scale: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor, za: torch.Tensor) -> torch.Tensor:
    # input_x_weight = quant_input * weight
    shift = 2 ** (-32)
    bias_q = torch.round(bias/(weight_scale
                               * input_scale))
    m = input_scale * weight_scale/scale
    m_0 = m/shift

    output = torch.round(shift*m_0*(quant_input-za + bias_q))+zp
    return output


if __name__ == '__main__':
    with open("save_activation/input.txt", "r") as f:
        content = f.read()
        data = ast.literal_eval(content)
        tensor_data = torch.tensor(data)
        f.close()
    input = tensor_data
    with open("save_activation/conv2_pw.txt", "r") as f:
        content = f.read()
        data = ast.literal_eval(content)
        tensor_data = torch.tensor(data)
        f.close()
    model_conv2_pw_out = tensor_data
    with open("save_activation/dequant.txt", "r") as f:
        content = f.read()
        data = ast.literal_eval(content)
        tensor_data = torch.tensor(data)
        f.close()
    dequant = tensor_data
    # 读取 scale 和 zero_point
    conv1_dw_scale = read_txt("conv1_dw.scale.txt")
    conv1_dw_zero_point = read_txt("conv1_dw.zero_point.txt")
    conv1_dw_weight = read_txt("conv1_dw.weight.txt")
    conv1_dw_bias = read_txt("conv1_dw.bias.txt")
    conv1_dw_weight_scale = read_txt("conv1_dw.weight_scale.txt")
    conv1_dw_weight_zero_point = read_txt("conv1_dw.weight_zero_points.txt")
    conv1_dw_za = read_txt("conv1_dw.weight_za.txt")

    conv1_pw_scale = read_txt("conv1_pw.scale.txt")
    conv1_pw_zero_point = read_txt("conv1_pw.zero_point.txt")
    conv1_pw_weight = read_txt("conv1_pw.weight.txt")
    conv1_pw_bias = read_txt("conv1_pw.bias.txt")
    conv1_pw_weight_scale = read_txt("conv1_pw.weight_scale.txt")
    conv1_pw_weight_zero_point = read_txt("conv1_pw.weight_zero_points.txt")
    conv1_pw_za = read_txt("conv1_pw.weight_za.txt")

    conv2_dw_scale = read_txt("conv2_dw.scale.txt")
    conv2_dw_zero_point = read_txt("conv2_dw.zero_point.txt")
    conv2_dw_weight = read_txt("conv2_dw.weight.txt")
    conv2_dw_bias = read_txt("conv2_dw.bias.txt")
    conv2_dw_weight_scale = read_txt("conv2_dw.weight_scale.txt")
    conv2_dw_weight_zero_point = read_txt("conv2_dw.weight_zero_points.txt")
    conv2_dw_za = read_txt("conv2_dw.weight_za.txt")

    conv2_pw_scale = read_txt("conv2_pw.scale.txt")
    conv2_pw_zero_point = read_txt("conv2_pw.zero_point.txt")
    conv2_pw_weight = read_txt("conv2_pw.weight.txt")
    conv2_pw_bias = read_txt("conv2_pw.bias.txt")
    conv2_pw_weight_scale = read_txt("conv2_pw.weight_scale.txt")
    conv2_pw_weight_zero_point = read_txt("conv2_pw.weight_zero_points.txt")
    conv2_pw_za = read_txt("conv2_pw.weight_za.txt")

    fc1_scale = read_txt("fc1.scale.txt")
    fc1_zero_point = read_txt("fc1.zero_point.txt")
    fc1_weight = read_txt("fc1._packed_params._packed_params_weight.txt")
    fc1_bias = read_txt("fc1._packed_params._packed_params_bias.txt")
    fc1_weight_zp = read_txt(
        "fc1._packed_params._packed_params_zero_points.txt")
    fc1_weights_scale = read_txt(
        "fc1._packed_params._packed_params_scale.txt")
    fc1_za = read_txt("fc1._packed_params._packed_params_za.txt")

    fc2_scale = read_txt("fc2.scale.txt")
    fc2_zero_point = read_txt("fc2.zero_point.txt")
    fc2_weight = read_txt("fc2._packed_params._packed_params_weight.txt")
    fc2_bias = read_txt("fc2._packed_params._packed_params_bias.txt")
    fc2_weight_zp = read_txt(
        "fc2._packed_params._packed_params_zero_points.txt")
    fc2_weights_scale = read_txt("fc2._packed_params._packed_params_scale.txt")
    fc2_za = read_txt("fc2._packed_params._packed_params_za.txt")

    quant_scale = read_txt("quant.scale.txt")
    quant_zero_point = read_txt("quant.zero_point.txt")
    next = torch.clamp(
        torch.round(input/quant_scale + quant_zero_point), min=0, max=255)
    save_array_to_txt(next, "save_activation_out/quant_output.txt")

    next = F.conv2d(next, conv1_dw_weight,
                    bias=None, stride=1)

    next = Conv2dquant(next, input_scale=quant_scale, input_zp=quant_zero_point,
                       weight_scale=conv1_dw_weight_scale, bias=conv1_dw_bias, scale=conv1_dw_scale, zp=conv1_dw_zero_point, za=conv1_dw_za)
    next = F.relu(next)
    save_array_to_txt(next, "save_activation_out/conv1_dw_output.txt")
    next = F.max_pool2d(next, kernel_size=(2, 2), stride=(2, 2))
    save_array_to_txt(next, "save_activation_out/Max_pool2d.txt")
    next = F.conv2d(next, conv1_pw_weight,
                    bias=None, stride=1, padding=0)
    next = Conv2dquant(next, input_scale=conv1_dw_scale, input_zp=conv1_dw_zero_point, weight_scale=conv1_pw_weight_scale,
                       bias=conv1_pw_bias, scale=conv1_pw_scale, zp=conv1_pw_zero_point, za=conv1_pw_za)
    next = F.relu(next)
    save_array_to_txt(next, "save_activation_out/conv1_pw_output.txt")
    next = F.conv2d(next, conv2_dw_weight, groups=10,
                    bias=None, stride=1, padding=0)
    next = Conv2dquant(next, input_scale=conv1_pw_scale, input_zp=conv1_pw_zero_point, weight_scale=conv2_dw_weight_scale,
                       bias=conv2_dw_bias, scale=conv2_dw_scale, zp=conv2_dw_zero_point, za=conv2_dw_za)
    next = F.relu(next)
    save_array_to_txt(next, "save_activation_out/conv2_dw_output.txt")
    next = F.max_pool2d(next, kernel_size=(2, 2), stride=(2, 2))
    next = F.conv2d(next, conv2_pw_weight,
                    bias=None, stride=1, padding=0)
    next = Conv2dquant(next, input_scale=conv2_dw_scale, input_zp=conv2_dw_zero_point, weight_scale=conv2_pw_weight_scale,
                       bias=conv2_pw_bias, scale=conv2_pw_scale, zp=conv2_pw_zero_point, za=conv2_pw_za)
    next = F.relu(next)
    save_array_to_txt(next, "save_activation_out/conv2_pw_output.txt")
    next = next.contiguous().view(-1, 320)
    next = linearqunat(next, input_scale=conv2_pw_scale, input_zp=conv2_pw_zero_point, weight=fc1_weight, weight_scale=fc1_weights_scale,
                       weight_zp=fc1_weight_zp, output_scale=fc1_scale, output_zp=fc1_zero_point, bias=fc1_bias, za=fc1_za)
    next = F.relu(next)
    save_array_to_txt(next, "save_activation_out/FC1.txt")
    next = linearqunat(next, input_scale=fc1_scale, input_zp=fc1_zero_point, weight=fc2_weight, weight_scale=fc2_weights_scale,
                       weight_zp=fc2_weight_zp, output_scale=fc2_scale, za=fc2_za, output_zp=fc2_zero_point, bias=fc2_bias)
    # dequant fc2 output
    next = (next - fc2_zero_point) * fc2_scale
    output = F.log_softmax(next, dim=1)
    with open("save_activation_out/model_int8_output.txt", "r") as f:
        content = f.read()
        data = ast.literal_eval(content)
        tensor_data = torch.tensor(data)
        f.close()
    model_int8_output = tensor_data
    with open("save_activation_out/model_output.txt", "r") as f:
        content = f.read()
        data = ast.literal_eval(content)
        tensor_data = torch.tensor(data)
        f.close()
    model_output = tensor_data
    print(
        f"The final error of cal is {F.mse_loss(output, model_output)} (vs. Float32 model )")
    print(
        f"The final error of cal is {F.mse_loss(output, model_int8_output)} (vs. int8 model)")
