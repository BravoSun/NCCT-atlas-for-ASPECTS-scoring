# NCCT-atlas-for-ASPECTS-scoring
This repository includes the age-specific non-contrast CT ASPECTS atlases across four age groups: 10-29 years, 30-49 years, 50-69 years, and 70-89 years constructed based on 281 NCCT images of healthy subjects from clinical practice. 

Keywords: Non-contrast computed tomography, ASPECTS regions, age-specific atlas. 

## Preprocessing
The preprocessing include brain tissue extraction and image alignment. The former extract brain tissue at the selected ASPECTS layers based on threshold and connected domain, the latter align the image to keep the brain region in the center of the image and horizontally symmetrical using the Symmetry-Enhanced Attention Network (Ni, H. et al. (2022). Asymmetry Disentanglement Network for Interpretable Acute Ischemic Stroke Infarct Segmentation in Non-contrast CT Scans. In: Wang, L., Dou, Q., Fletcher, P.T., Speidel, S., Li, S. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2022. MICCAI 2022. Lecture Notes in Computer Science, vol 13438. Springer, Cham. https://doi.org/10.1007/978-3-031-16452-1_40).

## Atlas construction
We utilized the workflow defined in the $ antsMultivariateTemplateConstruction2.sh $ script from the Advanced Normalization Tools (ANTs, http://stnava.github.io/ANTs/). The iteration number was set to 4, and the remaining parameters were set to default values.

## ASPECTS key slices localization
A neural network (ASPECTSLoc-Net) was developed to automatically localize the subcortical and cortical regions from a single brain NCCT image.  

## ASPECTS region mapping
After brain tissue extraction, the cortical and subcortical regions were registered with the corresponding atlas to achieve ASPECTS region mapping.

## Data provided 
* ASPECTS atlases
  * Average intensity brain (skull stripped)
  * ASPECTS regions (skull stripped)
Our final constructed ASPECTS atlas has been released in https://doi.org/10.6084/m9.figshare.26819290.
  
## Usage Notes 
The constructed age-specific ASPECTS atlases are a series of two-dimensional atlases corresponding to the level of the thalamus and basal ganglia and the rostral to ganglionic structures of different age groups. In the usage process, users can either utilize our ASPECTSLoc-Net or manually determine the ASPECTS slices to map the corresponding ASPECTS regions. These atlases not only can be used for rapid ASPECTS scoring, but the relevant neuroscience research.
  
## Citation
Sun, Q., Wang, G., Yang, J. et al. Age-specific ASPECTS atlas of Chinese subjects across different age groups for assessing acute ischemic stroke. Sci Data 11, 1132 (2024). https://doi.org/10.1038/s41597-024-03973-y
