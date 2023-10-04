import torch

def create_single_batch_maps(maps: {str: torch.Tensor}) -> {str: torch.Tensor}:
    for k in maps.keys():
        data = maps[k]
        if data is not None:
            if len(data.shape) == 3:
                maps[k] = data.unsqueeze(0)

    return maps

def create_rendered_maps(color, ambient, radiance, diffuse, specular: torch.Tensor) -> {str: torch.Tensor}:
    res = {}
    res['color'] = color.to('cpu')
    res['ambient'] = ambient.to('cpu')
    res['radiance'] = radiance.to('cpu')
    res['diffuse'] = diffuse.to('cpu')
    res['specular'] = specular.to('cpu')

    return res