import SimpleITK as sitk
import numpy as np
import os
import scipy.ndimage as ndimage
from tqdm import tqdm
import torch
from transform_net import Alignment
import math
import torch.nn.functional as F
import cv2

def brain_tissue_ext(img_arr):
    img = sitk.GetImageFromArray(img_arr)

    seed_region = img_arr[156:356, 156:356]
    seed_pts = np.argwhere((seed_region > 20) & (seed_region < 30))

    if seed_pts.shape[0] == 0:
        print('Brain tissue extraction failed!')

    seed_pts_idx = 0
    seed_pts = seed_pts[seed_pts_idx]
    seed_pts = [(int(seed_pts[0] + 156), int(seed_pts[1] + 156))]

    brain_mask = sitk.ConfidenceConnected(img, seedList=seed_pts, numberOfIterations=0, multiplier=12,
                                          initialNeighborhoodRadius=5, replaceValue=1)

    BMC = sitk.BinaryMorphologicalClosingImageFilter()
    BMC.SetKernelType(sitk.sitkBall)
    BMC.SetKernelRadius(10)
    BMC.SetForegroundValue(1)
    brain_mask = BMC.Execute(brain_mask)

    brain_mask = sitk.GetArrayFromImage(brain_mask)
    brain_tissue_arr = img_arr * brain_mask

    dim = brain_tissue_arr.shape
    brain_tissue_upper = brain_tissue_arr[0:int(dim[0]/2), 0:dim[1]]
    brain_tissue_upper[brain_tissue_upper > 90] = 0
    brain_tissue_arr[0:int(dim[0]/2), 0:dim[1]] = brain_tissue_upper

    brain_mask[img_arr < 0] = 0
    brain_mask[img_arr > 120] = 0

    brain_mask = select_max_region(brain_mask)
    brain_mask = fill_hole(brain_mask)
    brain_tissue_arr = img_arr * brain_mask

    _, _, stats, _ = cv2.connectedComponentsWithStats(brain_mask, connectivity=8)
    stats = stats[stats[:, 4].argsort()]
    bbox = stats[:-1]

    return brain_tissue_arr, bbox

def select_max_region(mask):
    num, labels = cv2.connectedComponents(mask, connectivity=8)

    max_label_idx = np.max(labels)

    if max_label_idx == 1:
        return mask

    max_region_id = -1
    max_region = 0
    if max_label_idx > 1:
        for i in range(1, max_label_idx + 1):
            region = np.sum(labels == i)
            if region > max_region:
                max_region_id = i
                max_region = region

    mask[labels != max_region_id] = 0

    return mask

def fill_hole(label_arr):
    tissue_idx = np.unique(label_arr)
    for id in tissue_idx:
        if id == 0:
            continue

        binary_label = label_arr.copy()
        binary_label[binary_label != id] = 0
        binary_label[binary_label == id] = 1

        binary_label = ndimage.binary_fill_holes(binary_label)

        label_arr[binary_label == 1] = id

    return label_arr

def sampler(img_arr, label_arr, ori_dpi, sample_dpi):
    img_arr = ndimage.zoom(img_arr, (ori_dpi / sample_dpi, ori_dpi / sample_dpi), order=1)
    label_arr = ndimage.zoom(label_arr, (ori_dpi / sample_dpi, ori_dpi / sample_dpi), order=0)

    return img_arr, label_arr

def get_patient_age(patient_path):
    file_list = os.listdir(patient_path)
    for file in file_list:
        file_path = os.path.join(patient_path, file)
        if os.path.isdir(file_path):
            dcm_list = os.listdir(file_path)
            for dcm in dcm_list:
                if '.dcm' in dcm:
                    dcm_path = os.path.join(file_path, dcm)
                    dcm_img = sitk.ReadImage(dcm_path)
                    age = int(dcm_img.GetMetaData('0010|1010')[1:3])
                    return age

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

def image_alignment(image_array, label_array):
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

    net = Alignment().cuda()
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


patient_folder_path = '/home/usr/Dataset/'
age_spcific_save_folder = '/home/usr/preprocess/preproc_age_group_data/'

patient_list = os.listdir(patient_folder_path)

age_flag = ''
for patient in tqdm(patient_list):

    patient_path = os.path.join(patient_folder_path, patient)

    if not os.path.isdir(patient_path):
        continue

    img_path = patient_path + '/image.nii.gz'
    label_path = patient_path + '/label.nii.gz'

    vol = sitk.ReadImage(img_path)
    vol_arr = sitk.GetArrayFromImage(vol)

    label = sitk.ReadImage(label_path)
    label_arr = sitk.GetArrayFromImage(label)

    # alignment
    vol_arr, label_arr = image_alignment(vol_arr, label_arr)

    dim = vol_arr.shape
    name_idx = 0

    origin = vol.GetOrigin()
    direction = vol.GetDirection()
    spacing = vol.GetSpacing()

    for z in range(0, dim[0]):
        if np.all(label_arr[z, :, :] == 0):
            continue

        img_arr = vol_arr[z, :, :]
        refined_label = label_arr[z, :, :]

        cnt = len(np.where(refined_label == 1)[0])

        if cnt != 0:
            name = 'nucleus'
        else:
            name = 'nucleus_above'

        age = get_patient_age(patient_path)

        if age < 30:
            continue
            age_flag = '10-30'
            age_1030_list.append(age)
            age_1030_name_list.append(patient)
            average_age_1030 += age
            age_spcific_save_path = os.path.join(age_spcific_save_folder, '10-30')
            if not os.path.exists(age_spcific_save_path):
                os.makedirs(age_spcific_save_path)
        elif age >= 30 and age < 50:
            continue
            age_flag = '30-50'
            age_3050_list.append(age)
            age_3050_name_list.append(patient)
            average_age_3050 += age
            age_spcific_save_path = os.path.join(age_spcific_save_folder, '30-50')
            if not os.path.exists(age_spcific_save_path):
                os.makedirs(age_spcific_save_path)
        elif age >= 50 and age < 70:
            continue
            age_flag = '50-70'
            age_5070_list.append(age)
            age_5070_name_list.append(patient)
            average_age_5070 += age
            age_spcific_save_path = os.path.join(age_spcific_save_folder, '50-70')
            if not os.path.exists(age_spcific_save_path):
                os.makedirs(age_spcific_save_path)
        elif age >= 70:
            age_flag = '70-90'
            age_7090_list.append(age)
            age_7090_name_list.append(patient)
            average_age_7090 += age
            age_spcific_save_path = os.path.join(age_spcific_save_folder, '70-90')
            if not os.path.exists(age_spcific_save_path):
                os.makedirs(age_spcific_save_path)
        else:
            print(patient)

        if not os.path.exists(os.path.join(age_spcific_save_path, patient)):
            os.makedirs(os.path.join(age_spcific_save_path, patient))

        img_save_path = os.path.join(age_spcific_save_path, patient) + '/image_' + name + '.nii.gz'
        label_save_path = os.path.join(age_spcific_save_path, patient) + '/label_' + name + '.nii.gz'

        brain_tissue_arr, bbox = brain_tissue_ext(img_arr)

        height = bbox[0, 3] * spacing[1]
        height_dict[age_flag].append(height)

        # fill hole
        refined_label = fill_hole(refined_label)

        brain_tissue_arr_nor = brain_tissue_arr
        refined_label_nor = refined_label

        brain_tissue = sitk.GetImageFromArray(brain_tissue_arr_nor)
        brain_tissue.SetOrigin(origin)
        brain_tissue.SetSpacing(spacing)

        refined_label = sitk.GetImageFromArray(refined_label_nor)
        refined_label.SetOrigin(origin)
        refined_label.SetSpacing(spacing)

        sitk.WriteImage(brain_tissue, img_save_path)
        sitk.WriteImage(refined_label, label_save_path)

        name_idx = name_idx + 1
        if name_idx > 2:
            print('{} label mistake'.format(patient))
