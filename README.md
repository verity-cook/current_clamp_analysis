# Current Clamp Analysis
for the analysis of whole-cell current clamp data in the .abf format

## How to use

### File format
Data should be organised as follows

"data/\[YYYYMMDD]/\[YYYYMMDD]\_Cell\[n]\_\[XXXX].abf" 

where YYYYMMDD is the date of the experiment, n is the cell number and XXXX is the number of the experiment. For example

"data/20220718/20220718\_Cell3_0015.abf"

### Current clamp protocols 
The type of protocol saved in each file is got from the meta data but the analysis picks the files based on the name of the protocol. For example, the resting membrane potential is calculated from files with the protocol name "L_RMP", and the rheobase is calculated from files with "L_Ic_rheobase_delta5pA". If the name of the protocols in your experiment is different, then these should be changed in ```cell_analysis_methods.py```

### Cell ID
For comparing between different conditions the identiy of the cell should be recorded in a csv in the form 

|   Date   | Cell |     ID    |
|   ____   | ____ |    ___    | 
| YYYYMMDD |   n  | condition |

### Run code
```cell_analysis_methods.py``` contains the fucntions for the analysis
```cell_analysis.ipynb``` run the cells to create a pickle file containing a pandas DataFrame with all of the data from the analysis, ready to be plotted. 
```plot_cell_data.ipynb``` run the cells to create and save the plots in a folder caled "plots".  
