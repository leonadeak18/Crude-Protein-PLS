I am working on developing an efficient partial least square regression (PLSR) model to predict crude protein content in wheat kernel samples using Near Infrared Spectroscopy (NIRS) data. The dataset consists of four sheets: two for calibration purposes and the remaining two for model validation. This datasets is freely available online and was compiled by Wenya Liu, 2016. Here are the three main steps I did:
1. Data separation
    - Created an argparse input that can separate single workbook based on its sheets.
2. Data preprocessing and spectra plotting
    - Created an argparse input that can
        a. remove sample data that contain null value
        b. process preprocessing data, such as Multiplicative scatter correction (MSC), Standard Normal Variate (SNV), first and second derivative.
        c. plot all spectra, before and after pretreatment
        d. export the plot as an image file
3. Build PLSR models and determine the most effective one.
    - Create functions that can use to fit a model and plot it.
    - By comparing the performance of different models, I determined the most effective one.
    
Source:
Wenya, L. (2016). Wheat kernel dataset: Figshare. Retrieved from https://figshare.com/articles/wheat_kernel_dataset/4252217/1