#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import math
import argparse
import warnings
import pkg_resources
import collections.abc
from collections import abc
from itertools import repeat
from functools import partial
from packaging.version import parse
from typing import Any, Tuple, Callable, Optional, Type, Union

import cv2
import numpy as np
from network.Vision_Transformers import creat_transformers

import torch
import torch.nn as nn
import pytorch_grad_cam as cam
from torchvision.models import resnet50
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

METHOD_MAP = {
    "gradcam": cam.GradCAM,
    "gradcam++": cam.GradCAMPlusPlus,
    "scorecam": cam.ScoreCAM,
}

"""
使用文档
> img              代表要可视化的图片路径
> checkpoint       表示权重文件路径
> model-name       表示模型名称
> num-classes      表示模型分类类别
> target-layers    表示用哪一层的梯度进行热力图绘制
> preview-model    表示打印模型的各个层次结构
> method           表示使用的热力图方法
> target-category  表示绘制目标类别的热力图
> eigen-smooth     表示平滑热力图
> aug-smooth       表示平滑热力图
> save-path        表示热力图存放路径
> device           表示使用设备，cpu或者cuda
> vit-like         表示创建的模型是否是vit的变体
> num-extra-tokens 表示去除vit变体中额外的token    "swint":0
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('--img', type=str, default=r'dataset\D\242.png', help='Image file')####这里要改：bai zhi min.png换成你自己的
    parser.add_argument('--checkpoint', type=str, default=r'swint_T1C.pth', help='Checkpoint file')
    parser.add_argument('--model-name', type=str, default='swint', help='Model')
    parser.add_argument('--num-classes', type=int, default=3, help='The num of classes')####这里要改
    parser.add_argument('--target-layers', default=[], nargs='+', type=str, help='The target layers to get CAM, if not set, the tool will '
                                                                                                'specify the norm layer in the last block. Backbones '
                                                                                                'implemented by users are recommended to manually specify'
                                                                                                ' target layers in commmad statement.')
    parser.add_argument('--preview-model', default=False, action='store_true', help='To preview all the model layers')
    parser.add_argument('--method', default='gradcam', help='Type of method to use, supports 'f'{", ".join(list(METHOD_MAP.keys()))}.')
    parser.add_argument('--target-category', default=[], nargs='+', type=int, help='The target category to get CAM, default to use result get from given model.')
    parser.add_argument('--eigen-smooth', default=False, action='store_true', help='Reduce noise by taking the first principle componenet of ``cam_weights*activations``')
    parser.add_argument('--aug-smooth', default=False, action='store_true', help='Wether to use test time augmentation, default not to use')
    parser.add_argument('--save-path', default='./242_cam.jpg', help='The path to save visualize cam image, default not to save.')####这里要改，但是“_cam.jpg”必须保留
    parser.add_argument('--device', default='cpu', help='Device to use cpu')
    parser.add_argument('--vit-like', default=True, action='store_true', help='Whether the network is a ViT-like network.')
    parser.add_argument('--num-extra-tokens', default=0, type=int, help='The number of extra tokens in ViT-like backbones. Defaults to use num_extra_tokens of the backbone.')
    args = parser.parse_args()
    if args.method.lower() not in METHOD_MAP.keys():
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(list(METHOD_MAP.keys()))}.')

    return args

def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Defaults to 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert 'parrots' not in version_str
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)

def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

def reshape_transform(tensor, model, args):
    """Build reshape_transform for `cam.activations_and_grads`, which is
    necessary for ViT-like networks."""
    # ViT_based_Transformers have an additional clstoken in features
    if tensor.ndim == 4:
        # For (B, C, H, W)
        return tensor
    elif tensor.ndim == 3:
        if not args.vit_like:
            raise ValueError(f"The tensor shape is {tensor.shape}, if it's a "
                             'vit-like backbone, please specify `--vit-like`.')
        # For (B, L, C)
        if args.num_extra_tokens is not None:
            num_extra_tokens = args.num_extra_tokens
        else:
            num_extra_tokens = args.num_extra_tokens or getattr(model, 'num_extra_tokens', 1)

        tensor = tensor[:, num_extra_tokens:, :]
        # get heat_map_height and heat_map_width, preset input is a square
        heat_map_area = tensor.size()[1]
        height, width = _ntuple(2)(int(math.sqrt(heat_map_area)))
        assert height * height == heat_map_area, \
            (f"The input feature's length ({heat_map_area+num_extra_tokens}) "
             f'minus num-extra-tokens ({num_extra_tokens}) is {heat_map_area},'
             ' which is not a perfect square number. Please check if you used '
             'a wrong num-extra-tokens.')
        # (B, L, C) -> (B, H, W, C)
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        # (B, H, W, C) -> (B, C, H, W)
        result = result.permute(0, 3, 1, 2)
        return result
    else:
        raise ValueError(f'Unsupported tensor shape {tensor.shape}.')

def init_cam(method, model, target_layers, use_cuda, reshape_transform):
    """Construct the CAM object once, In order to be compatible with
    mmpretrain, here we modify the ActivationsAndGradients object."""
    GradCAM_Class = METHOD_MAP[method.lower()]
    cam = GradCAM_Class(
        model=model, target_layers=target_layers)
    # Release the original hooks in ActivationsAndGradients to use
    # ActivationsAndGradients.
    cam.activations_and_grads.release()
    cam.activations_and_grads = ActivationsAndGradients(
        cam.model, cam.target_layers, reshape_transform)

    return cam

def is_seq_of(seq: Any,
              expected_type: Union[Type, tuple],
              seq_type: Type = None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True
def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)
def is_norm(layer: nn.Module,
            exclude: Optional[Union[type, Tuple[type]]] = None) -> bool:
    """Check if a layer is a normalization layer.

    Args:
        layer (nn.Module): The layer to be checked.
        exclude (type, tuple[type], optional): Types to be excluded.

    Returns:
        bool: Whether the layer is a norm layer.
    """
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude, )
        if not is_tuple_of(exclude, type):
            raise TypeError(
                f'"exclude" must be either None or type or a tuple of types, '
                f'but got {type(exclude)}: {exclude}')

    if exclude and isinstance(layer, exclude):
        return False

    all_norm_bases = (_BatchNorm, _InstanceNorm, nn.GroupNorm, nn.LayerNorm)
    return isinstance(layer, all_norm_bases)

def get_layer(layer_str, model):
    """get model layer from given str."""
    for name, layer in model.named_modules():
        if name == layer_str:
            return layer
    raise AttributeError(
        f'Cannot get the layer "{layer_str}". Please choose from: \n' +
        '\n'.join(name for name, _ in model.named_modules()))

def show_cam_grad(grayscale_cam, src_img, title, out_path=None):
    """fuse src_img and grayscale_cam and show or save."""
    grayscale_cam = grayscale_cam[0, :]
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    src_img = np.float32(src_img) / 255
    visualization_img = show_cam_on_image(
        src_img, grayscale_cam, use_rgb=False)
    if out_path:
        cv2.imwrite(out_path, visualization_img)
        cv2.imwrite(out_path.replace('cam','only_cam'), heatmap)
    else:
        raise ValueError("args.out_path is None.")

def get_default_target_layers(model, args):
    """get default target layers from given model, here choose nrom type layer
    as default target layer."""
    norm_layers = [
        (name, layer)
        for name, layer in model.named_modules()
        if is_norm(layer)
    ]
    if args.vit_like:
        # For ViT models, the final classification is done on the class token.
        # And the patch tokens and class tokens won't interact each other after
        # the final attention layer. Therefore, we need to choose the norm
        # layer before the last attention layer.
        num_extra_tokens = args.num_extra_tokens or getattr(
            model, 'num_extra_tokens', 1)

        # models like swin have no attr 'out_type', set out_type to avg_featmap
        out_type = getattr(model, 'out_type', 'avg_featmap')
        if out_type == 'cls_token' or num_extra_tokens > 0:
            # Assume the backbone feature is class token.
            name, layer = norm_layers[-3]
            print('Automatically choose the last norm layer before the '
                  f'final attention block "{name}" as the target layer.')
            return [layer]

    # For CNN models, use the last norm layer as the target-layer
    name, layer = norm_layers[-1]
    print('Automatically choose the last norm layer '
          f'"{name}" as the target layer.')
    return [layer]

def main():
    args = parse_args()
    print(args)

    # build the model
    model = creat_transformers(model_name=args.model_name, num_classes=args.num_classes, pretrained=False)
    # model = resnet50(num_classes=args.num_classes, pretrained=False)
    if args.preview_model:
        print(model)
        print('\n Please remove `--preview-model` to get the CAM.')
        return
    if args.checkpoint is not None:
        try:
            model.load_state_dict(torch.load(args.checkpoint)['model'])
        except:
            model.load_state_dict(torch.load(args.checkpoint))
        print("Load checkpoint Done!")

    # apply transform and perpare data
    rgb_img = cv2.imread(args.img)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # build target layers
    if args.target_layers:
        target_layers = [
            get_layer(layer, model) for layer in args.target_layers
        ]
    else:
        target_layers = get_default_target_layers(model, args)

    # init a cam grad calculator
    # use_cuda = ('cuda' in args.device)
    use_cuda = False
    cam = init_cam(args.method, model, target_layers, use_cuda,
                    partial(reshape_transform, model=model, args=args))

    targets = None
    if args.target_category:
        grad_cam_v = pkg_resources.get_distribution('grad_cam').version
        if digit_version(grad_cam_v) >= digit_version('1.3.7'):
            from pytorch_grad_cam.utils.model_targets import \
                ClassifierOutputTarget
            targets = [ClassifierOutputTarget(c) for c in args.target_category]
        else:
            targets = args.target_category

    # calculate cam grads and show|save the visualization image
    grayscale_cam = cam(
        input_tensor,
        targets,
        eigen_smooth=args.eigen_smooth,
        aug_smooth=args.aug_smooth)
    show_cam_grad(
        grayscale_cam, rgb_img, title=args.method, out_path=args.save_path)

if __name__ == "__main__":
    main()
