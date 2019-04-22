import argparse
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import resnet_v2 as resnet_ext

pt_models = { 'resnet18': models.resnet18,
              'resnet50': models.resnet50,
              'resnet34': models.resnet34,
              'inception': models.inception_v3,
              'alexnet': models.alexnet,
              'vgg16': models.vgg16,
              'squeezenet0': models.squeezenet1_0,
              'squeezenet1': models.squeezenet1_1,
              'densenet121': models.densenet121,
              'densenet161': models.densenet161,
              'densenet169': models.densenet169,
              'densenet201': models.densenet201
            }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation script for Face Recognition in PyTorch')
    parser.add_argument('--device', type=int, default=-1, help='Device to store the model.')
    parser.add_argument('--model', choices=list(pt_models.keys()), type=str, default='resnet18')
    args = parser.parse_args()

    net = pt_models[args.model]()
    # net = resnet_ext.build_resnet('resnet50', 'classic')
    flops, params = get_model_complexity_info(net, (224, 224), as_strings=True, print_per_layer_stat=True)
    print('Flops: ' + flops)
    print('Params: ' + params)
