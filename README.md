# agency_pymdp
Riddhi J. Pitliya Feb 2024

### Description:
An active inference simulation of the two-agent (participant + simulated participant) agency task. The agency task is the one implemented with the Italy team. Button Pressing. Shapes. PS_ZO, NS_ZO, ZS_ZO, ZS_PO, ZS_NO.

### To run experiments from scratch: 
1. Empty the "experiments" folder and "csvlog" folder.
2. Optionally delete the "results.txt" file. Remember to save the previous version for record-keeping if you choose to delete the file. If you choose not to delete the file, the results from your experiments will be appended.
3. Alter the parameters in config.yaml 
4. Go to the directory where your files are in the Mac terminal (via cd), and run "./sweep.sh config.yaml 30", where the number determines the number of experiments you want to run in a set of experiments. The individual results are saved in the "csvlog" folder (and "experiments" folder but in a different format), and the collated results are saved in the "results.txt" file.
