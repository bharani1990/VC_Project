import os
import pandas as pd
import argparse
import torch
from compress import compress_lt, compress_dct
from lpips_calculator import calculate_lpips
from model import LappedTransform

def main(model, dataset_path, save_path):
    results = {}
    for filename in os.listdir(dataset_path):
        if filename.endswith(('.jpg', '.tif', '.png')):
            original_image_path = os.path.join(dataset_path, filename)
            lt_compressed_path = compress_lt(model, original_image_path, save_path)
            dct_compressed_path = compress_dct(original_image_path, save_path)

            lpips_lt = calculate_lpips(original_image_path, lt_compressed_path)
            lpips_dct = calculate_lpips(original_image_path, dct_compressed_path)

            original_size = os.path.getsize(original_image_path)
            lt_compressed_size = os.path.getsize(lt_compressed_path)
            dct_compressed_size = os.path.getsize(dct_compressed_path)

            compression_ratio_lt = lt_compressed_size / original_size
            compression_ratio_dct = dct_compressed_size / original_size
            
            better = 'Lapped Transform' if lpips_lt < lpips_dct else 'DCT'
            
            results[filename] = {
                'lt_lpips': lpips_lt,
                'dct_lpips': lpips_dct,
                'lt_compression_ratio': compression_ratio_lt,
                'dct_compression_ratio': compression_ratio_dct,
                'winner': better,
            }
    return results

def create_directories(path):
    os.makedirs(path, exist_ok=True)
    os.makedirs(path, exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Compression using Lapped Transform and DCT.')
    parser.add_argument('--type', choices=['gray', 'colored'], required=True, help='Specify the type of images to process.')
    args = parser.parse_args()
    model = LappedTransform(kernel_size=16)
    if args.type == 'gray':
        dataset_path = 'data/gray'
        save_path = 'results/images/gray'
        model.load_state_dict(torch.load('models/lapped_transform_gray.pth'))
    else:
        dataset_path = 'data/colored'
        save_path = 'results/images/colored'
        model.load_state_dict(torch.load('models/lapped_transform_colored.pth'))
    create_directories(save_path)
    results = main(model, dataset_path, save_path)
    results_df = pd.DataFrame(results).T
    path_to_save = 'results/csvs'
    results_df.to_csv(os.path.join(path_to_save,f'compression_results_{args.type}.csv'))
    print(results_df)
