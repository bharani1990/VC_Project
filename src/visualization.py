import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

def plot_compression_ratios(results_df, colored=False):
    plt.figure(figsize=(10, 6))
    
    # Plotting the compression ratios
    plt.scatter(results_df['dct_compression_ratio'], results_df['dct_compression_ratio'], c='blue', label='DCT')
    plt.scatter(results_df['lt_compression_ratio'], results_df['lt_compression_ratio'], c='red', label='Lapped Transform')
    
    plt.xlabel('Lapped Transform Compression Ratio')
    plt.ylabel('DCT Compression Ratio')
    
    if colored:
        plt.title('Compression Ratio Comparison - Colored Images')
        save_path = 'results/plots/colored/compression_ratios_colored.png'
    else:
        plt.title('Compression Ratio Comparison - Gray Images')
        save_path = 'results/plots/gray/compression_ratios_gray.png'
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.legend()
    plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Compression Ratios.')
    parser.add_argument('--type', choices=['gray', 'colored'], required=True, help='Specify the type of images to plot.')
    args = parser.parse_args()

    if args.type == 'gray':
        results_df = pd.read_csv('results/csvs/compression_results_gray.csv')
        plot_compression_ratios(results_df, colored=False)
    else:
        results_df = pd.read_csv('results/csvs/compression_results_colored.csv')
        plot_compression_ratios(results_df, colored=True)
