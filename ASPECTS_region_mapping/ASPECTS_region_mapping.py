#coding=utf-8
import os
import shutil

import SimpleITK as sitk
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transform_net import PlaneFinder
import math
from matplotlib import pyplot as plt
import register

def ASPECTS_score(wraped_nuclear_label_path, wraped_nuclear_above_label_path,
                  subject_nuclear_label_path, subject_nuclear_above_label_path):
    # 统计DICE参数
    wraped_nuclear_label = sitk.ReadImage(wraped_nuclear_label_path)
    wraped_nuclear_label_array = sitk.GetArrayFromImage(wraped_nuclear_label)
    wraped_nuclear_label_array = np.array(wraped_nuclear_label_array, dtype='uint8')

    subject_nuclear_label = sitk.ReadImage(subject_nuclear_label_path)
    subject_nuclear_label_array = sitk.GetArrayFromImage(subject_nuclear_label)
    subject_nuclear_label_array = np.array(subject_nuclear_label_array, dtype='uint8')

    wraped_nuclear_above_label = sitk.ReadImage(wraped_nuclear_above_label_path)
    wraped_nuclear_above_label_array = sitk.GetArrayFromImage(wraped_nuclear_above_label)
    wraped_nuclear_above_label_array = np.array(wraped_nuclear_above_label_array, dtype='uint8')

    subject_nuclear_above_label = sitk.ReadImage(subject_nuclear_above_label_path)
    subject_nuclear_above_label_array = sitk.GetArrayFromImage(subject_nuclear_above_label)
    subject_nuclear_above_label_array = np.array(subject_nuclear_above_label_array, dtype='uint8')

    spacing = subject_nuclear_label.GetSpacing()[0]

    all_score = metric_claculation(wraped_nuclear_label_array, subject_nuclear_label_array, region_dict, 'all', spacing)
    C_score = metric_claculation(wraped_nuclear_label_array, subject_nuclear_label_array, region_dict, 'C', spacing)
    L_score = metric_claculation(wraped_nuclear_label_array, subject_nuclear_label_array, region_dict, 'L', spacing)
    IC_score = metric_claculation(wraped_nuclear_label_array, subject_nuclear_label_array, region_dict, 'IC', spacing)
    I_score = metric_claculation(wraped_nuclear_label_array, subject_nuclear_label_array, region_dict, 'I', spacing)
    M1_score = metric_claculation(wraped_nuclear_label_array, subject_nuclear_label_array, region_dict, 'M1', spacing)
    M2_score = metric_claculation(wraped_nuclear_label_array, subject_nuclear_label_array, region_dict, 'M2', spacing)
    M3_score = metric_claculation(wraped_nuclear_label_array, subject_nuclear_label_array, region_dict, 'M3', spacing)
    M4_score = metric_claculation(wraped_nuclear_above_label_array, subject_nuclear_above_label_array, region_dict, 'M4', spacing)
    M5_score = metric_claculation(wraped_nuclear_above_label_array, subject_nuclear_above_label_array, region_dict, 'M5', spacing)
    M6_score = metric_claculation(wraped_nuclear_above_label_array, subject_nuclear_above_label_array, region_dict, 'M6', spacing)

    score_dict = {'all': 0,
                   'C': 0,
                   'L': 0,
                   'IC': 0,
                   'I': 0,
                   'M1': 0,
                   'M2': 0,
                   'M3': 0,
                   'M4': 0,
                   'M5': 0,
                   'M6': 0}

    score_dict['all'] = all_score
    score_dict['C'] = C_score
    score_dict['L'] = L_score
    score_dict['IC'] = IC_score
    score_dict['I'] = I_score
    score_dict['M1'] = M1_score
    score_dict['M2'] = M2_score
    score_dict['M3'] = M3_score
    score_dict['M4'] = M4_score
    score_dict['M5'] = M5_score
    score_dict['M6'] = M6_score

    return score_dict

def DICE_statics(score_list):

    region_dice_dict = {'all': 0,
                        'C': 0,
                        'L': 0,
                        'IC': 0,
                        'I': 0,
                        'M1': 0,
                        'M2': 0,
                        'M3': 0,
                        'M4': 0,
                        'M5': 0,
                        'M6': 0}

    DICE_list = []
    subject_DICE_list = []
    for subject_score in score_list:
        region_dice_dict['all'] += subject_score['score']['all']['dice']
        region_dice_dict['C'] += subject_score['score']['C']['dice']
        region_dice_dict['L'] += subject_score['score']['L']['dice']
        region_dice_dict['IC'] += subject_score['score']['IC']['dice']
        region_dice_dict['I'] += subject_score['score']['I']['dice']
        region_dice_dict['M1'] += subject_score['score']['M1']['dice']
        region_dice_dict['M2'] += subject_score['score']['M2']['dice']
        region_dice_dict['M3'] += subject_score['score']['M3']['dice']
        region_dice_dict['M4'] += subject_score['score']['M4']['dice']
        region_dice_dict['M5'] += subject_score['score']['M5']['dice']
        region_dice_dict['M6'] += subject_score['score']['M6']['dice']

        subject_DICE_list.append(subject_score['subject'])
        subject_DICE_list.append(subject_score['age'])
        subject_DICE_list.append(subject_score['score']['C']['dice'])
        subject_DICE_list.append(subject_score['score']['L']['dice'])
        subject_DICE_list.append(subject_score['score']['IC']['dice'])
        subject_DICE_list.append(subject_score['score']['I']['dice'])
        subject_DICE_list.append(subject_score['score']['M1']['dice'])
        subject_DICE_list.append(subject_score['score']['M2']['dice'])
        subject_DICE_list.append(subject_score['score']['M3']['dice'])
        subject_DICE_list.append(subject_score['score']['M4']['dice'])
        subject_DICE_list.append(subject_score['score']['M5']['dice'])
        subject_DICE_list.append(subject_score['score']['M6']['dice'])

        DICE_list.append(subject_DICE_list.copy())
        subject_DICE_list.clear()

    return DICE_list, region_dice_dict

def difference_map(warped_label_path, subject_label_path, region_label, output_folder):
    # 重合部分用蓝色（3），过分割部分用红色（1），欠分割部分用绿色（2）
    warped_label = sitk.GetArrayFromImage(sitk.ReadImage(warped_label_path))
    subject_label = sitk.GetArrayFromImage(sitk.ReadImage(subject_label_path))

    # subject_spacing = sitk.ReadImage(subject_label_path)
    # warped_spacing = sitk.ReadImage(subject_label_path)
    #
    # subject_label = ndimage.zoom(subject_label, (subject_spacing[1] / warped_spacing[1],
    #                                              subject_spacing[0] / warped_spacing), order=1, mode='nearest')

    warped_label_diff = np.copy(warped_label)
    subject_label_diff = np.copy(subject_label)

    warped_label_diff[warped_label != region_dict[region_label]] = 0
    warped_label_diff[warped_label == region_dict[region_label]] = 3

    subject_label_diff[subject_label != region_dict[region_label]] = 0
    subject_label_diff[subject_label == region_dict[region_label]] = 3

    diff_label = np.copy(subject_label_diff)
    over_mask = (warped_label_diff == 3) & (subject_label_diff != 3)
    under_mask = (warped_label_diff != 3) & (subject_label_diff == 3)

    # 过分割
    diff_label[over_mask] = 1
    diff_label[under_mask] = 2

    diff_label = sitk.GetImageFromArray(diff_label)
    diff_label.SetOrigin(sitk.ReadImage(subject_label_path).GetOrigin())
    diff_label.SetSpacing(sitk.ReadImage(subject_label_path).GetSpacing())
    diff_label.SetDirection(sitk.ReadImage(subject_label_path).GetDirection())

    output_path = os.path.join(output_folder, 'diff_' + region_label + '.nii.gz')
    sitk.WriteImage(diff_label, output_path)

def get_transform_matrices(view):
    b = view.size(0)
    if view.size(1) == 6:
        rx = view[:, 0]
        ry = view[:, 1]
        rz = view[:, 2]
        trans_xyz = view[:, 3:].reshape(b, 1, 3)
    elif view.size(1) == 5:
        rx = view[:, 0]
        ry = view[:, 1]
        rz = view[:, 2]
        delta_xy = view[:, 3:].reshape(b, 1, 2)
        trans_xyz = torch.cat([delta_xy, torch.zeros(b, 1, 1).to(view.device)], 2)
    elif view.size(1) == 3:
        rx = view[:, 0]
        ry = view[:, 1]
        rz = view[:, 2]
        trans_xyz = torch.zeros(b, 1, 3).to(view.device)
    rot_mat = get_rotation_matrix(rx, ry, rz)
    # change rot_mat to [4, 4]
    R = torch.zeros((rot_mat.shape[0], 4, 4)).to(device=rot_mat.device)
    R[:, :3, :3] = rot_mat
    R[:, 3, 3] = 1
    T = get_translation_matrix(trans_xyz)
    M = torch.matmul(T, R)
    # compute inverse matrix
    R_inv = R.transpose(1, 2)
    T_inv = get_translation_matrix(-trans_xyz)
    M_inv = torch.matmul(R_inv, T_inv)
    return M, M_inv

def get_rotation_matrix(tx, ty, tz):
    m_x = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_y = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_z = torch.zeros((len(tx), 3, 3)).to(tx.device)

    m_x[:, 1, 1], m_x[:, 1, 2] = tx.cos(), -tx.sin()
    m_x[:, 2, 1], m_x[:, 2, 2] = tx.sin(), tx.cos()
    m_x[:, 0, 0] = 1

    m_y[:, 0, 0], m_y[:, 0, 2] = ty.cos(), ty.sin()
    m_y[:, 2, 0], m_y[:, 2, 2] = -ty.sin(), ty.cos()
    m_y[:, 1, 1] = 1

    m_z[:, 0, 0], m_z[:, 0, 1] = tz.cos(), -tz.sin()
    m_z[:, 1, 0], m_z[:, 1, 1] = tz.sin(), tz.cos()
    m_z[:, 2, 2] = 1
    return torch.matmul(m_z, torch.matmul(m_y, m_x))

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)
    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def stn(x, theta, mode='nearest'):
    # theta must be (Bs, 3, 4) = [R|t]
    a = x.size()
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    out = F.grid_sample(x, grid, padding_mode='border', align_corners=False, mode=mode)
    return out

def Alignment(image_array, label_array):
    origin_image_array = image_array.copy()
    origin_label_array = label_array.copy()

    image_array[image_array < 0] = 0
    image_array[image_array > 80] = 80

    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    image_tensor = torch.tensor(image_array)
    nor_size = [24, 256, 256]

    image_tensor = image_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
    image_tensor = torch.nn.functional.interpolate(image_tensor, nor_size, mode='trilinear', align_corners=False)

    image_tensor = image_tensor.cuda().type(torch.float32)

    net = PlaneFinder().cuda()
    net.load_state_dict(torch.load('B0050.pth')['state_dict'])
    net.eval()

    transform = net(image_tensor)

    # show_transform = transform[0, :].data.cpu().numpy()
    #
    transform = torch.cat([transform[:, :1] * math.pi / 180 * 0,
                          transform[:, 1:2] * math.pi / 180 * 0,
                          transform[:, 2:3] * math.pi / 180 * 40,
                          transform[:, 3:4] * 0.5,
                          transform[:, 4:5] * 0,
                          transform[:, 5:] * 0], 1)
    #
    # rx = transform[0][0]
    # ry = transform[0][1]
    # rz = transform[0][2]
    # tx = transform[0][3]
    # ty = transform[0][4]
    # tz = transform[0][5]
    # print('rx:%.8f ry:%.8f rz:%.8f tx:%.8f ty:%.8f tz:%.8f' % (rx, ry, rz, tx, ty, tz))

    # 生成变换矩阵
    M, M_inv = get_transform_matrices(transform)

    # 图像变换
    origin_image_tensor = torch.tensor(origin_image_array)
    origin_image_tensor = origin_image_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
    origin_image_tensor = origin_image_tensor.cuda().type(torch.float32)
    images_t = stn(origin_image_tensor, M[:, :3, :])

    origin_label_tensor = torch.tensor(origin_label_array.astype(np.uint8))
    origin_label_tensor = origin_label_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
    origin_label_tensor = origin_label_tensor.cuda().type(torch.float32)
    labels_t = stn(origin_label_tensor, M[:, :3, :])

    images_t = images_t.squeeze(dim=0).squeeze(dim=0)
    image_t_array = images_t.data.cpu().numpy()

    labels_t = labels_t.squeeze(dim=0).squeeze(dim=0)
    label_t_array = labels_t.data.cpu().numpy()

    return image_t_array.astype(np.int16), label_t_array.astype(np.uint16)

# # 将选定的验证集复制到指定文件夹中
# origin_data_folder = '/home/sunq/projects/ASPECTS/preprocess'
# test_folder = '/home/sunq/projects/ASPECTS/atlas/paper/TestingData'
# test_data_file = '/home/sunq/projects/ASPECTS/atlas/paper/testing_dataset.csv'
#
# df = pd.read_csv(test_data_file, encoding='utf-8-sig')
# subject_list = df.values.tolist()
#
# for subject in subject_list:
#     print(subject)
#     name = subject[0]
#     age = subject[1]
#
#     rater_expert_subject_folder = os.path.join(origin_data_folder, 'rater_expert', age, name)
#     rater1_subject_folder = os.path.join(origin_data_folder, 'rater1', age, name)
#     rater2_subject_folder = os.path.join(origin_data_folder, 'rater2', age, name)
#
#     test_rater_expert_subject_folder = os.path.join(test_folder, 'rater_expert', age, name)
#     test_rater1_subject_folder = os.path.join(test_folder, 'rater1', age, name)
#     test_rater2_subject_folder = os.path.join(test_folder, 'rater2', age, name)
#
#     shutil.copytree(rater_expert_subject_folder, test_rater_expert_subject_folder)
#     shutil.copytree(rater1_subject_folder, test_rater1_subject_folder)
#     shutil.copytree(rater2_subject_folder, test_rater2_subject_folder)



# 统计两个rater的直接区域覆盖率，以及每个rater和图谱的覆盖率（配准）
data_folder = ''

nucleus_img_tag = 'image_subcortical.nii.gz'
nucleus_label_tag = 'label_subcortical.nii.gz'
nucleus_above_img_tag = 'image_cortical.nii.gz'
nucleus_above_label_tag = 'label_cortical.nii.gz'

age_list = os.listdir(data_folder)
age_list.sort()

cortical
subcortical

for age in age_list:
    print('-------------------------------- ' + age + ' --------------------------------')
    if age == '10-30':
        atlas_cortical_image_path = os.path.join(atlas_folder, 'Pad_nucleus_image_1030.nii.gz')
        atlas_cortical_label_path = os.path.join(atlas_folder, 'Pad_nucleus_label_1030.nii.gz')
        atlas_subcortical_image_path = os.path.join(atlas_folder, 'Pad_nucleus_above_image_1030.nii.gz')
        atlas_subcortical_label_path = os.path.join(atlas_folder, 'Pad_nucleus_above_label_1030.nii.gz')
    elif age == '30-50':
        atlas_cortical_image_path = os.path.join(atlas_folder, 'Pad_nucleus_image_3050.nii.gz')
        atlas_cortical_label_path = os.path.join(atlas_folder, 'Pad_nucleus_label_3050.nii.gz')
        atlas_subcortical_image_path = os.path.join(atlas_folder, 'Pad_nucleus_above_image_3050.nii.gz')
        atlas_subcortical_label_path = os.path.join(atlas_folder, 'Pad_nucleus_above_label_3050.nii.gz')
    elif age == '50-70':
        atlas_cortical_image_path = os.path.join(atlas_folder, 'Pad_nucleus_image_5070.nii.gz')
        atlas_cortical_label_path = os.path.join(atlas_folder, 'Pad_nucleus_label_5070.nii.gz')
        atlas_subcortical_image_path = os.path.join(atlas_folder, 'Pad_nucleus_above_image_5070.nii.gz')
        atlas_subcortical_label_path = os.path.join(atlas_folder, 'Pad_nucleus_above_label_5070.nii.gz')
    elif age == '70-90':
        atlas_cortical_image_path = os.path.join(atlas_folder, 'Pad_nucleus_image_7090.nii.gz')
        atlas_cortical_label_path = os.path.join(atlas_folder, 'Pad_nucleus_label_7090.nii.gz')
        atlas_subcortical_image_path = os.path.join(atlas_folder, 'Pad_nucleus_above_image_7090.nii.gz')
        atlas_subcortical_label_path = os.path.join(atlas_folder, 'Pad_nucleus_above_label_7090.nii.gz')

    expert_age_folder = os.path.join(expert_folder, age)
    expert_subject_list = os.listdir(expert_age_folder)
    expert_subject_list.sort()

    score_rater1_rater2_list = []
    score_rater1_expert_list = []
    score_rater2_expert_list = []
    score_atlas_expert_list = []

    for subject in tqdm(expert_subject_list):
        expert_subject_folder = os.path.join(expert_age_folder, subject)

        # 统计两个rater和expert标记的DICE
        expert_nuclues_img_path = os.path.join(expert_subject_folder, nucleus_img_tag)
        expert_nuclues_above_img_path = os.path.join(expert_subject_folder, nucleus_above_img_tag)
        expert_nuclues_label_path = os.path.join(expert_subject_folder, nucleus_label_tag)
        expert_nuclues_above_label_path = os.path.join(expert_subject_folder, nucleus_above_label_tag)
        expert_img_path = os.path.join(expert_subject_folder, img_tag)

        rater1_nuclues_img_path = os.path.join(expert_subject_folder.replace('expert', 'rater1'), nucleus_img_tag)
        rater1_nuclues_above_img_path = os.path.join(expert_subject_folder.replace('expert', 'rater1'), nucleus_above_img_tag)
        rater1_nuclues_label_path = os.path.join(expert_subject_folder.replace('expert', 'rater1'), nucleus_label_tag)
        rater1_nuclues_above_label_path = os.path.join(expert_subject_folder.replace('expert', 'rater1'), nucleus_above_label_tag)


        rater2_nuclues_img_path = os.path.join(expert_subject_folder.replace('expert', 'rater1'), nucleus_img_tag)
        rater2_nuclues_above_img_path = os.path.join(expert_subject_folder.replace('expert', 'rater1'),
                                                     nucleus_above_img_tag)
        rater2_nuclues_label_path = os.path.join(expert_subject_folder.replace('expert', 'rater1'),
                                                 nucleus_label_tag)
        rater2_nuclues_above_label_path = os.path.join(expert_subject_folder.replace('expert', 'rater1'),
                                                       nucleus_above_label_tag)

        # rater1和expert
        rater1_expert_score_dict = ASPECTS_score(rater1_nuclues_label_path, rater1_nuclues_above_label_path,
                                                 expert_nuclues_label_path, expert_nuclues_above_label_path)

        # rater2和expert
        rater2_expert_score_dict = ASPECTS_score(rater2_nuclues_label_path, rater2_nuclues_above_label_path,
                                                 expert_nuclues_label_path, expert_nuclues_above_label_path)

        # if rater1_expert_score_dict['all']['dice'] > 0.99 or rater2_expert_score_dict['all']['dice'] > 0.99:
        #     print(subject + "_same data error")
        #     continue

        rater1_expert_subject_score_dict = {'subject': subject, 'age': age, 'score': rater1_expert_score_dict}
        score_rater1_expert_list.append(rater1_expert_subject_score_dict)

        rater2_expert_subject_score_dict = {'subject': subject, 'age': age, 'score': rater2_expert_score_dict}
        score_rater2_expert_list.append(rater2_expert_subject_score_dict)

        # rater1和rater2
        rater1_rater2_score_dict = ASPECTS_score(rater1_nuclues_label_path, rater1_nuclues_above_label_path,
                                                 rater2_nuclues_label_path, rater2_nuclues_above_label_path)

        rater1_rater2_subject_score_dict = {'subject': subject, 'age': age, 'score': rater1_rater2_score_dict}
        score_rater1_rater2_list.append(rater1_rater2_subject_score_dict)

        # 统计图谱和两个rater标记的DICE（配准结果保存在rater1中）
        reg_type = 'SyN'
        aff_metric = 'mattes'
        # aff_metric = 'meansquares'
        register.wrapper(expert_nuclues_img_path, atlas_nuclear_image_path, atlas_nuclear_label_path,
                         expert_subject_folder, expert_subject_folder,
                         reg_type=reg_type, aff_metric=aff_metric, output_jac=False, prefix='')

        register.wrapper(expert_nuclues_above_img_path, atlas_nuclear_above_image_path,
                         atlas_nuclear_above_label_path,
                         expert_subject_folder, expert_subject_folder,
                         reg_type=reg_type, aff_metric=aff_metric, output_jac=False, prefix='')

        wraped_nuclues_image_name = 'SyN_Pad_nucleus_image_' + age.replace('-', '') + '.nii.gz'
        wraped_nuclues_above_image_name = 'SyN_Pad_nucleus_above_image_' + age.replace('-', '') + '.nii.gz'
        wraped_nuclues_label_name = 'SyN_Pad_nucleus_label_' + age.replace('-', '') + '.nii.gz'
        wraped_nuclues_above_label_name = 'SyN_Pad_nucleus_above_label_' + age.replace('-', '') + '.nii.gz'

        wraped_nuclear_label_path = os.path.join(expert_subject_folder, wraped_nuclues_label_name)
        wraped_nuclear_above_label_path = os.path.join(expert_subject_folder, wraped_nuclues_above_label_name)

        # 输出差值图像
        difference_map(wraped_nuclear_label_path, expert_nuclues_label_path, 'C',
                       expert_subject_folder)
        difference_map(wraped_nuclear_label_path, expert_nuclues_label_path, 'L',
                       expert_subject_folder)
        difference_map(wraped_nuclear_label_path, expert_nuclues_label_path, 'IC',
                       expert_subject_folder)
        difference_map(wraped_nuclear_label_path, expert_nuclues_label_path, 'I',
                       expert_subject_folder)
        difference_map(wraped_nuclear_label_path, expert_nuclues_label_path, 'M1',
                       expert_subject_folder)
        difference_map(wraped_nuclear_label_path, expert_nuclues_label_path, 'M2',
                       expert_subject_folder)
        difference_map(wraped_nuclear_label_path, expert_nuclues_label_path, 'M3',
                       expert_subject_folder)
        difference_map(wraped_nuclear_above_label_path, expert_nuclues_above_label_path, 'M4',
                       expert_subject_folder)
        difference_map(wraped_nuclear_above_label_path, expert_nuclues_above_label_path, 'M5',
                       expert_subject_folder)
        difference_map(wraped_nuclear_above_label_path, expert_nuclues_above_label_path, 'M6',
                       expert_subject_folder)

        # atlas和expert
        atlas_expert_score_dict = ASPECTS_score(wraped_nuclear_label_path, wraped_nuclear_above_label_path,
                                   expert_nuclues_label_path, expert_nuclues_above_label_path)

        atlas_expert_subject_score_dict = {'subject': subject, 'age': age, 'score': atlas_expert_score_dict}
        score_atlas_expert_list.append(atlas_expert_subject_score_dict)

        os.remove(os.path.join(expert_subject_folder, wraped_nuclues_image_name))
        os.remove(os.path.join(expert_subject_folder, wraped_nuclues_above_image_name))
        os.remove(os.path.join(expert_subject_folder, wraped_nuclues_label_name))
        os.remove(os.path.join(expert_subject_folder, wraped_nuclues_above_label_name))

    dice_rate1_rater2_list, region_dice_rate1_rater2_dict = DICE_statics(score_rater1_rater2_list)
    dice_rate1_expert_list, region_dice_rate1_expert_dict = DICE_statics(score_rater1_expert_list)
    dice_rate2_expert_list, region_dice_rate2_expert_dict = DICE_statics(score_rater2_expert_list)
    dice_atlas_expert_list, region_dice_atlas_expert_dict = DICE_statics(score_atlas_expert_list)

    # 保存所有subject的dice
    header = ['subject', 'age', 'C', 'L', 'IC', 'I', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']

    # dice_df_output_path = os.path.join(statistic_folder, 'test_dice_rater1_rater2_stats_' + age + '.csv')
    # dice_df = pd.DataFrame(columns=header, data=dice_rate1_rater2_list)
    # dice_df = dice_df.astype(str)
    # dice_df.to_csv(dice_df_output_path, quotechar='"', index=False)

    dice_df_output_path = os.path.join(statistic_folder, 'test_dice_rater1_expert_stats_' + age + '.csv')
    dice_df = pd.DataFrame(columns=header, data=dice_rate1_expert_list)
    dice_df = dice_df.astype(str)
    dice_df.to_csv(dice_df_output_path, quotechar='"', index=False)

    dice_df_output_path = os.path.join(statistic_folder, 'test_dice_rater2_expert_stats_' + age + '.csv')
    dice_df = pd.DataFrame(columns=header, data=dice_rate2_expert_list)
    dice_df = dice_df.astype(str)
    dice_df.to_csv(dice_df_output_path, quotechar='"', index=False)

    dice_df_output_path = os.path.join(statistic_folder, 'test_dice_atlas_expert_stats_' + age + '.csv')
    dice_df = pd.DataFrame(columns=header, data=dice_atlas_expert_list)
    dice_df = dice_df.astype(str)
    dice_df.to_csv(dice_df_output_path, quotechar='"', index=False)

    # 保存所有subject每个区域的平均dice
    for key in region_dice_rate1_rater2_dict.keys():
        region_dice_rate1_rater2_dict[key] /= len(expert_subject_list)

    for key in region_dice_rate1_expert_dict.keys():
        region_dice_rate1_expert_dict[key] /= len(expert_subject_list)

    for key in region_dice_rate2_expert_dict.keys():
        region_dice_rate2_expert_dict[key] /= len(expert_subject_list)

    for key in region_dice_atlas_expert_dict.keys():
        region_dice_atlas_expert_dict[key] /= len(expert_subject_list)

    header = ['all', 'C', 'L', 'IC', 'I', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']

    dice_region_rate1_rate2_list = list(region_dice_rate1_rater2_dict.values())
    dice_region_rate1_expert_list = list(region_dice_rate1_expert_dict.values())
    dice_region_rate2_expert_list = list(region_dice_rate2_expert_dict.values())
    dice_region_atlas_expert_list = list(region_dice_atlas_expert_dict.values())

    # dice_df_output_path = os.path.join(statistic_folder, 'test_dice_region_rater1_rater2_stats_' + age + '.csv')
    # dice_df = pd.DataFrame(columns=header, data=[dice_region_rate1_rate2_list])
    # dice_df = dice_df.astype(str)
    # dice_df.to_csv(dice_df_output_path, quotechar='"', index=False)

    dice_df_output_path = os.path.join(statistic_folder, 'test_dice_region_rater1_expert_stats_' + age + '.csv')
    dice_df = pd.DataFrame(columns=header, data=[dice_region_rate1_expert_list])
    dice_df = dice_df.astype(str)
    dice_df.to_csv(dice_df_output_path, quotechar='"', index=False)

    dice_df_output_path = os.path.join(statistic_folder, 'test_dice_region_rater2_expert_stats_' + age + '.csv')
    dice_df = pd.DataFrame(columns=header, data=[dice_region_rate2_expert_list])
    dice_df = dice_df.astype(str)
    dice_df.to_csv(dice_df_output_path, quotechar='"', index=False)

    dice_df_output_path = os.path.join(statistic_folder, 'test_dice_region_atlas_expert_stats_' + age + '.csv')
    dice_df = pd.DataFrame(columns=header, data=[dice_region_atlas_expert_list])
    dice_df = dice_df.astype(str)
    dice_df.to_csv(dice_df_output_path, quotechar='"', index=False)



