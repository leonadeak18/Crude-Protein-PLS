"""
    This script processes Near-Infrared (NIR) spectra data from a CSV file, performs various preprocessing steps,
    and generates plots of the original, preprocessed, and derivative spectra.

    Functions:
    - remove_na: Removes rows with NA or null values from the dataframe.
    - combine_wavelength: Combines wavelength data with the absorbance values if they are in separate CSV files.
    - plotting: Plots the spectra data.
    - preprocess_msc: Applies Multiplicative Scatter Correction (MSC) preprocessing to the spectra.
    - preprocess_snv: Applies Standard Normal Variate (SNV) preprocessing to the spectra.
    - derivate: Calculates the derivative of the spectra using Savitzky-Golay smoothing.

    Usage:
    The script can be run from the command line with various options for preprocessing and plotting.

    Arguments:
    - -plot: Path to the input CSV file containing the spectra data.
    - -w, --wavelength: (Optional) Path to the CSV file containing wavelength data.
    - -o, --out_file: (Optional) Path to save the output plot as a PNG file.
    - -msc: Apply MSC preprocessing (mutually exclusive with SNV).
    - -snv: Apply SNV preprocessing (mutually exclusive with MSC).
    - -der: Apply derivative calculation. Choices are 'der1' for 1st derivative and 'der2' for 2nd derivative.

    Example usage:
    python nir_plot_der.py -plot spectra.csv -w wavelength.csv -o output.png -msc -der der1 der2

    The script generates and saves or displays plots of the original, preprocessed, and derivative spectra.
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from chemotools.derivative import SavitzkyGolay

#cleaning NA data
def remove_na(df):
    if df.isnull().values.any():
        print("NA or null values found. Removing rows with NA or null values")
        df_cleaned = df.dropna()
    else:
        df_cleaned = df
    return df_cleaned

#Combine wavelength with the absorbance if csv doesn't contain the wavelength
def combine_wavelength(df, wavelength):
    wave_data = wavelength.values.flatten()
    df_data = df.values
    
    # Create a DataFrame from df_data
    combined_df = pd.DataFrame(df_data, columns=wave_data)

    return combined_df

#Plotting
def plotting(df):
    x = df.columns.values
    y = df.iloc[0:].values
    y_transpose = y.T
    plt.plot(x, y_transpose)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption") 
        

# Multiplicative scatter correction (MSC) preprocessing
def preprocess_msc(df):
    # Convert DataFrame to numpy array
    y = df.iloc[0:].values
    
    # Zero mean each spectrum (subtract the mean of each row)
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    
    # Initialize variables
    temp = []
    y_msc = np.zeros(y.shape)
    
    # Process each spectrum
    for i in range(y_centered.shape[0]):
        for j in range(0, y_centered.shape[0], 10):
            temp.append(np.mean(y_centered[j:j + 10], axis=0))
        
        # Compute the mean of the temp array for polyfit
        mean_temp = np.mean(temp, axis=0)
        fit = np.polyfit(mean_temp, y_centered[i, :], 1)
        y_msc[i, :] = (y_centered[i, :] - fit[1]) / fit[0]
    
    # Convert numpy array back to DataFrame
    df_msc = pd.DataFrame(y_msc, columns=df.columns, index=df.index)
    
    return df_msc

#Standard Normal Variate (SNV) preprocessing
def preprocess_snv(df):
    # Convert DataFrame to numpy array
    y = df.iloc[0:].values
    
    #initialize variable
    y_snv = np.zeros_like(df)
    
    #running the correction
    for i in range(y.shape[0]):
        y_snv[i,:] = (y[i,:] - np.mean(y[i,:])) / np.std(y[i,:])
        
    # Convert numpy array back to dataframe
    df_snv = pd.DataFrame(y_snv, columns=df.columns, index=df.index)
    return df_snv

#derivatization
def derivate(df, window=15, p_order=2, d_order=1):
    sg = SavitzkyGolay(window_size=window, polynomial_order=p_order, derivate_order=d_order)
    # Convert DataFrame to numpy array
    y = df.iloc[0:].values
    der = sg.fit_transform(y)
    df_der = pd.DataFrame(der, columns=df.columns, index=df.index)
    return df_der

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plotting, Preprocessing and Derivate the NIR Spectra')
    
    #Plotting
    parser.add_argument('-plot', help="Plotting the NIR spectra and output as a jpg", type=str, required = True)
    parser.add_argument('-w','--wavelength', help="Reading a csv file that contains wavelength of the spectra", type=str)
    parser.add_argument('-o', '--out_file', type=str, help='Output PNG file for the plot')
    
    #Mutually exclusive group for MSC and SNV options
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument('-msc', help="Input must be a dataframe to run MSC function", action = 'store_true')
    group.add_argument('-snv', help="Input must be a dataframe to run SNV function", action = 'store_true')
    
    #Derivative can run after choosing SNV or MSC preprocessing
    parser.add_argument('-der', nargs='+', help="Input must be a dataframe to run derivative function", choices=['der1', 'der2'])
    
    args = parser.parse_args()
    
     # Reading the data frame
    imported_df = pd.read_csv(args.plot, header=None)
    imported_df = remove_na(imported_df)
    if args.wavelength:
        wavelength = pd.read_csv(args.wavelength, header=None)
        df = combine_wavelength(imported_df, wavelength)
    else:
        df = imported_df
    
    # Determine number of plots to create
    num_plots = 1
    if args.msc or args.snv:
        num_plots += 1  
    if args.der:
        num_plots += len(args.der)
    
    # Create a new figure
    plt.figure(figsize=(12, 4))
    
    # Plotting
    plot_index = 1
    
    # Plot original data
    plt.subplot(1, num_plots, plot_index)
    plotting(df)
    plt.title('Original Data')
    plot_index += 1
    
    #preprocessing execution
    if args.msc:
        preprocess_df = preprocess_msc(df)
        plt.subplot(1, num_plots, plot_index)
        plotting(preprocess_df)
        plt.title('MSC Preprocessed Data')
        plot_index += 1
    elif args.snv:
        preprocess_df = preprocess_snv(df)
        plt.subplot(1, num_plots, plot_index)
        plotting(preprocess_df)
        plt.title('SNV Preprocessed Data')
        plot_index += 1
    
    #derivative execution
    if args.der:
        for der_option in args.der:
            if der_option == "der1":
                der1 = derivate(preprocess_df, d_order=1)
                plt.subplot(1, num_plots, plot_index)
                plotting(der1)
                plt.title('1st Derivative')
                plot_index += 1
            elif der_option == "der2":
                der2 = derivate(preprocess_df, d_order=2)
                plt.subplot(1, num_plots, plot_index)
                plotting(der2)
                plt.title('2nd Derivative')
                plot_index += 1

    # Check if the output location is available
    if args.out_file:
        #output path
        os.makedirs("output_plot", exist_ok=True)
        output_plot_path = os.path.join("output_plot", args.out_file)
        #export the plot
        plt.savefig(output_plot_path)
        print(f"Plot saved as {args.out_file}")
    else:
        plt.tight_layout()
        plt.show()
if __name__ == '__main__':
    main()    
