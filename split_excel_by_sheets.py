"""
    Reads an input Excel file and generates separate output CSV files for each sheet.

    Actions:
        input_excel_path (str): Path to the input Excel file.
        output_folder (str): Path to the folder where output CSV files will be saved.
"""
import pandas as pd
import os

def split_excel_by_sheets(input_excel_path, output_folder):

    # Read the input Excel file into a dictionary of DataFrames (one for each sheet)
    excel_file = pd.ExcelFile(input_excel_path)
    sheet_data = {sheet_name: excel_file.parse(sheet_name) for sheet_name in excel_file.sheet_names}

    # Create separate output CSV files for each sheet
    os.makedirs(output_folder, exist_ok=True)
    for sheet_name, df in sheet_data.items():
        output_csv_path = os.path.join(output_folder, f"{sheet_name}.csv")
        df.to_csv(output_csv_path, index=False)
        print(f"Saved {sheet_name}.csv")
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plotting, Preprocessing and Derivate the NIR Spectra')
    parser.add_argument('-i', '--input_excel', type=str, help='Excel file location', required = True)
    parser.add_argument('-o', '--out_folder', default = "output_csv", type=str, help='Output folder location')
    args = parser.parse_args()
    
    split_excel_by_sheets(args.input_excel, args.out_folder)
