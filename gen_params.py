import argparse
import torch
import numpy as np
import os
import re
key_words = ['bias', 'scale', 'zero_point']


def save_array_to_txt(array, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if array is None:
        print(f"警告: 儲存 {filename} 時遇到空陣列")
        with open(filename, 'w') as f:
            array = np.zeros(1)
            array_str = np.array2string(
                array,
                separator=', ',    # 用逗號和空格分隔
                precision=8,       # 8位小數
                suppress_small=True,  # 抑制科學記號
                threshold=np.inf,   # 顯示所有元素
                max_line_width=np.inf  # 不換行
            )
            f.write(array_str)
            f.close()
        return
    if torch.is_tensor(array):
        if (array.dtype == torch.qint8) or (array.dtype == torch.quint8):
            # print(array)
            array = array.int_repr().numpy()
        else:
            array = array.detach().cpu().numpy()

        with open(filename, 'w') as f:
            array_str = np.array2string(
                array,
                separator=', ',    # 用逗號和空格分隔
                precision=20,       # 8位小數
                suppress_small=True,  # 抑制科學記號
                threshold=np.inf,   # 顯示所有元素
                max_line_width=np.inf  # 不換行
            )
            f.write(array_str)
            f.close()
    else:
        array = np.array(array)
        with open(filename, 'w') as f:
            array_str = np.array2string(
                array,
                separator=', ',    # 用逗號和空格分隔
                precision=20,       # 8位小數
                suppress_small=True,  # 抑制科學記號
                threshold=np.inf,   # 顯示所有元素
                max_line_width=np.inf  # 不換行
            )
            f.write(array_str)
            f.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate parameters for Pytorch Quantlizaton model')
    parser.add_argument('file', help="Input path of the Pytorch model")
    parser.add_argument('-o', "--output", metavar='output path', default='./weight/',
                        help="Specify an output path. If not specified, the output will be placed at the same level as this python file.", required=False)
    parser.add_argument("number", type=int,
                        help="A normalized constant (e.g., 16 or 32)", default=32)
    arg = parser.parse_args()
    return arg


input_scale = 0
bias_array = []
bias = 0
output_scale = 0
weight_scale = 0
pre_scale = 0
pattern = r'fc\d+'
new_zp = 0
if __name__ == '__main__':
    args = parse_args()
    if (args.number <= 0):
        print("Constant should bigger than 0")
        raise ValueError
    shift = 2 ** (-args.number)
    print(shift)
    model = torch.load(args.file, weights_only=True)
    keys = model.keys()
    layer = next(iter(model.items()))[0].split('.')[0]
    model_items = list(model.items())
    for i, (name, value) in enumerate(model_items):
        print(name)
        if name.split('.')[0] != layer:

            if re.search(pattern, layer) or re.search(r'conv\d+', layer):
                print(layer)
                bias_q = bias / (weight_scale*input_scale)
                za_bias = -za + bias_q
                save_array_to_txt(za_bias, args.output +
                                  f"params/{layer}_za_bias.txt")
                save_array_to_txt(round(input_scale*weight_scale /
                                        output_scale/shift), args.output+f"params/{layer}_M0.txt")

            layer = name.split('.')[0]
            input_scale = output_scale
        if isinstance(value, torch.dtype):
            save_array_to_txt(value, args.output+f"{name}.txt")
        elif 'packed_params' in name:
            ###################################################################################################
            #           In Pytorch. The fully-connect layer's parameters is packed in packed-format           #
            #           And it contain the weight and bias.                                                   #
            #           If you want to do something to the FC layer weight just do it here                    #
            ###################################################################################################
            weight, bias = value
            save_array_to_txt(
                weight, args.output+f"{name}_weight.txt")
            weight_scale = weight.q_scale()
            save_array_to_txt(weight.q_scale(
            ), args.output+f"{name}_scale.txt")
            save_array_to_txt(weight.q_zero_point(
            ), args.output+f"{name}_zero_points.txt")
            save_array_to_txt(
                bias, args.output+f"{name}_bias.txt")
            save_array_to_txt(
                torch.sum(weight.int_repr(), dim=1)*pre_zp, f"save_activation/layer_weight/{name}_za.txt")
            za = torch.sum(weight.int_repr(), dim=1)*pre_zp
            print(pre_zp)
        elif any(keyword in name for keyword in key_words):
            ###########################################################################
            #           You can use the array('key_words') to select the layer.       #
            #           Define the data format you want for output.                   #
            ###########################################################################
            save_array_to_txt(
                value,  args.output+f"{name}.txt")
            if 'scale' in name:
                output_scale = value.item()
            if 'bias' in name:
                if value is None:
                    bias = 0
                else:
                    bias = value
            if 'zero_point' in name:
                pre_zp = new_zp
                new_zp = value
        elif value.is_quantized:
            ###################################################################################################################################
            #           In Pytorch.If a layer is quantized, The array will contain parmeters (like. scale and zeropoint).                     #
            #           So we need to extract these from the quant-array by .q_scale and q_zero_point.                                        #
            #           And If you want to know how to use these parameter. Please read README. : )                                           #
            ###################################################################################################################################
            weight_scale = value.q_scale()
            save_array_to_txt(value.q_scale(
            ), args.output+f"{name}_scale.txt")
            save_array_to_txt(value.q_zero_point(
            ), args.output+f"{name}_zero_points.txt")
            save_array_to_txt(value, args.output+f"{name}.txt")
            save_array_to_txt(
                value.int_repr().numpy().flatten().sum() * pre_zp, f"save_activation/layer_weight/{name}_za.txt")
            za = value.int_repr().numpy().flatten().sum() * pre_zp
        else:
            save_array_to_txt(value, args.output+f"{name}.txt")
        if i + 1 == len(model_items):
            bias_q = bias / (weight_scale*input_scale)
            za_bias = -za + bias_q
            save_array_to_txt(za_bias, args.output +
                              f"params/{layer}_za_bias.txt")
            save_array_to_txt(round(input_scale*weight_scale /
                                    output_scale/shift), args.output+f"params/{layer}_M0.txt")
    save_array_to_txt(bias_array, args.output+f"conv_bias.txt")
