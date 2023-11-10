# EUMESAT-proc
github repo: https://github.com/quasur/EUMESAT-proc  
For the IDP  2 week project related to exploring the EUMESAT Data  
We managed to remove clouds and implement a crude biome detection algorithm  

In nearly all scripts you must specify where your dataset images are and where your mean and std arrays are. These are typically done in the first lines with in the mypath and datapath variables.  
If not supplied: mean and std data along with the EUMETSAT images can be found here - https://uniofnottm-my.sharepoint.com/:f:/g/personal/ppyac18_nottingham_ac_uk/EtGwgYg2k2JOhVmH_0eihxgBg3RlRaqcHbUxF-eJCt1SoA?e=EXhoMR
  
## What each file and folder does
### Files  

IDP EUMETSAT.pdf -The project report  

Contribution statement.txt - A statement on how workload was distributed and what aspects of the projects we worked on.   

Cloud detection.py - Produces a set of 23 images of the weighted average image of the earth over 15 days to remove clouds.  

Cloud detection graphs.py - The working code used to produce some of the graphs seen in the report. This script is poorly maintained and commented.  

biome.py - Produces graphs of how a pixel changes biome over time in lat vs time and lat vs long. Other variables can be used to extract information about SLIC clustering or biome identification.  

biome graphs.py - like biome.py but with additional plots for graphs we used in the report, this code is poorly maintained and commented and variables may need to be manually changed for successful plots. Please inspect version history of the AdamCloud branch if you are intrested in a sepcific function.  

mask.png - An image used to cut out africa from the monthly mean images.  

Yearvid.mp4 - A video showing the monthly means and biome classification over the year.  

README.md - This file  

### Folders

Biome classes - Contains images showing the biome classification throughout the year  

monthimg- 23 Monthly average images  

RepIm - Figures used or that may have been used at some point in the report  

Old code - Code kept here purely for archival purposes. Here be dragons. Poorly maintained code, with sometimes unclear purposes. Some plotting functions within were used to create figures for the report.  
