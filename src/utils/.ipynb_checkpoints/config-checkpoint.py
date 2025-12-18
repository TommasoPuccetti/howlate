"""
This module contains configuration settings.
It defines global constants, file paths, and desired False Positive Rate (FPR) thresholds 
used during the evaluation of the Intrusion Detection model.

### Configuration Overview

The following configuration settings are used to control various aspects of the evaluation pipeline:

1. **PRINT_SEPARATOR:** A string used for visual separation in printed output, aiding readability.
2. **ROOT:** The root directory for input/output data files..
3. **DESIRED_FPRS:** A list of False Positive Rate (FPR) thresholds at which the detection model's performance 
   will be evaluated.

### Main Configuration Variables

#### PRINT_SEPARATOR
A long separator string that is used to format printed output, making it easier to identify sections in the terminal or logs.

#### ROOT
Root directory path for data storage and access. This is the base directory for input datasets and output results.

#### DESIRED_FPRS
A list of desired False Positive Rate (FPR) thresholds.
"""

PRINT_SEPARATOR = "--------------------------------------------------------------------------------------------------------------------"
ROOT = "./data"
DESIRED_FPRS = [ 0.0000015, 0.000003027, 0.000018164, 0.000036327, 0.000072655, 0.000217964, 0.001089819 ]
#DESIRED_FPRS = [0.000018164 ]