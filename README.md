# NCCT-atlas-for-ASPECTS-scoring
This repository includes the age-specific non-contrast CT ASPECTS atlases across four age groups: 10-30 years, 30-50 years, 50-70 years, and 70-90 years constructed based on 281 NCCT images of healthy subjects from clinical practice. 

Keywords: Non-contrast computed tomography, ASPECTS regions, age-specific atlas. 

## Preprocessing
The preprocessing include brain tissue extraction and image alignment. The former extract brain tissue at the selected ASPECTS layers based on threshold and connected domain, the latter align the image to keep the brain region in the center of the image and horizontally symmetrical using the Symmetry-Enhanced Attention Network (Ni, H. et al. (2022). Asymmetry Disentanglement Network for Interpretable Acute Ischemic Stroke Infarct Segmentation in Non-contrast CT Scans. In: Wang, L., Dou, Q., Fletcher, P.T., Speidel, S., Li, S. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2022. MICCAI 2022. Lecture Notes in Computer Science, vol 13438. Springer, Cham. https://doi.org/10.1007/978-3-031-16452-1_40).

## Atlas construction
We utilized the workflow defined in the $ antsMultivariateTemplateConstruction2.sh $ script from the Advanced Normalization Tools (ANTs, http://stnava.github.io/ANTs/). The iteration number was set to 4, and the remaining parameters were set to default values.

## Data provided 
* ASPECTS atlases
  * Average intensity brain (skull stripped)
  * ASPECTS regions (skull stripped) 
  
## Usage Notes 
The constructed age-specific ASPECTS atlases are a series of two-dimensional atlases corresponding to the level of the thalamus and basal ganglia and the rostral to ganglionic structures of different age groups. In the usage process, users need to manually select relevant NCCT image layers to map the ASPECTS regions and extract the brain tissue. These atlases not only can be used for rapid ASPECTS scoring, but the relevant neuroscience research.
  
