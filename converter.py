import torch
import numpy as np
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str)
    parser.add_argument('--dst', type=str, default='./model.npy')
    args = parser.parse_args()

    state_dict = torch.load(args.src, map_location='cpu')['state_dict']

    padding = torch.zeros(13, 16)
    rgb_out = state_dict['model.rgb_net.output_layer.weight']
    print(rgb_out.shape)
    rgb_out = torch.cat([rgb_out, padding], dim=0)
    
    new_dict = {
        # 'camera_angle_x': meta['camera_angle_x'],
        'poses': state_dict['poses'].numpy(),
        'directions': state_dict['directions'].numpy(),
        'model.density_bitfield': state_dict['model.density_bitfield'].numpy(),
        'model.hash_encoder.params': state_dict['model.hash_encoder.params'].numpy(),
        'model.xyz_encoder.params': 
            torch.cat(
                [state_dict['model.xyz_encoder.hidden_layers.0.weight'].reshape(-1),
                state_dict['model.xyz_encoder.output_layer.weight'].reshape(-1)]
            ).numpy(),
        'model.rgb_net.params': 
            torch.cat(
                [state_dict['model.rgb_net.hidden_layers.0.weight'].reshape(-1),
                rgb_out.reshape(-1)]
            ).numpy(),
        # 'model.xyz_encoder.params': state_dict['model.xyz_encoder.params'].numpy(),
        # 'model.xyz_sigmas.params': state_dict['model.xyz_sigmas.params'].numpy(),
        # 'model.rgb_net.params': state_dict['model.rgb_net.params'].numpy(),
    }
    np.save(args.dst, new_dict)
