#!/bin/bash 
image_folder=$(dirname $(readlink -f "$0"))
echo $image_folder

export ANTSPATH=${HOME}/projects/ANTs/bin/
export PATH=${ANTSPATH}:$PATH

while getopts i:o:z:m:t: option
do
 case "${option}"
 in
 i) INPUT_PREFIX=${OPTARG};;
 o) OUTPUT_PREFIX=${OPTARG};;
 z) INIT_TEMPLATE=${OPTARG};;
 m) METRIC=${OPTARG};;
 t) TRANS=${OPTARG};;
 esac
done

template_folder='template'
transformation_folder='transformation_field'
warped_folder='warped_image'

cd ${image_folder}
mkdir -p ${image_folder}/${template_folder}
mkdir -p ${image_folder}/${transformation_folder}
mkdir -p ${image_folder}/${warped_folder}

template_name=${OUTPUT_PREFIX}'_template0.nii.gz'
affine_field='*GenericAffine.mat'
warped_field='*Warp.nii.gz'
warped_image='*WarpedToTemplate.nii.gz'

bash ${ANTSPATH}antsMultivariateTemplateConstruction2.sh -d 2 -m ${METRIC} -a 0 -t ${TRANS} -n 0 -c 2 -j 8 -z ${image_folder}/${INIT_TEMPLATE} -o ${OUTPUT_PREFIX}_ ${image_folder}/${INPUT_PREFIX}*

mv ${image_folder}/${template_name} ${image_folder}/${template_folder}
mv ${image_folder}/${affine_field} ${image_folder}/${transformation_folder}
mv ${image_folder}/${warped_field} ${image_folder}/${transformation_folder}
mv ${image_folder}/${warped_image} ${image_folder}/${warped_folder}

