import torch

from matrenderer import SVBRDF_MAPS

def create_single_batch_maps(maps: {str: torch.Tensor}) -> {str: torch.Tensor}:
    batched_maps = {}
    for k in maps.keys():
        data = maps[k]
        if data is None:
            batched_maps[k] = None
        else:
            if len(data.shape) == 3:
                batched_maps[k] = data.unsqueeze(0)

    return batched_maps

def create_rendered_maps(color, ambient, light, diffuse, specular: torch.Tensor) -> {str: torch.Tensor}:
    res = {}
    res['color'] = color.to('cpu')
    res['ambient'] = ambient.to('cpu')
    res['light'] = light.to('cpu')
    res['diffuse'] = diffuse.to('cpu')
    res['specular'] = specular.to('cpu')

    return res

def create_learnable_maps(height: int, width: int) -> ([torch.Tensor], {str: torch.Tensor}):
    maps = {}
    for map_key in SVBRDF_MAPS:
        n_channel = 1
        if map_key == "basecolor" or map_key == "normal":
            n_channel = 3
        param = torch.rand((n_channel, height, width), requires_grad=True)
        maps[map_key] = param
    return maps

def sigmoid_maps(maps: {str: torch.Tensor}) -> {str: torch.Tensor}:
    sigmoid_maps = {}
    for k in maps.keys():
        sigmoid_maps[k] = torch.sigmoid(maps[k])
    return sigmoid_maps