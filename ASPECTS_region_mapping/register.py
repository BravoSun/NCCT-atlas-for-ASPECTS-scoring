import os.path

import ants 
import SimpleITK as sitk 
import scipy.io as io
import numpy as np


def wrapper(fixed_img_path, mov_img_path, mov_lable_path, img_save_dir, label_save_dir,
            reg_type='SyN', aff_metric='meansquares', output_jac=False, prefix=''):

    f_img = ants.image_read(fixed_img_path)
    m_img = ants.image_read(mov_img_path)

    if mov_lable_path != '':
        m_label = ants.image_read(mov_lable_path)

    wrapper = ants.registration(fixed=f_img, moving=m_img,
                                type_of_transform=reg_type, aff_metric=aff_metric, random_seed=1)

    fwd_transforms = wrapper['fwdtransforms']
    inv_transforms = wrapper['invtransforms']

    warped_img = ants.apply_transforms(fixed=f_img, moving=m_img, transformlist=fwd_transforms,
                                       interpolator="linear")

    if mov_lable_path != '':
        warped_label = ants.apply_transforms(fixed=f_img, moving=m_label, transformlist=fwd_transforms,
                                             interpolator="genericLabel")

    warped_img.set_direction(f_img.direction)
    warped_img.set_origin(f_img.origin)
    warped_img.set_spacing(f_img.spacing)
    img_name = os.path.join(img_save_dir, reg_type + '_' + prefix + mov_img_path.split('/')[-1])
    ants.image_write(warped_img, img_name)

    if mov_lable_path != '':
        warped_label.set_direction(f_img.direction)
        warped_label.set_origin(f_img.origin)
        warped_label.set_spacing(f_img.spacing)
        label_name = os.path.join(label_save_dir, reg_type + '_' + prefix + mov_lable_path.split('/')[-1])
        ants.image_write(warped_label, label_name)

    if mov_lable_path != '':
        return warped_img, warped_label

    return warped_img