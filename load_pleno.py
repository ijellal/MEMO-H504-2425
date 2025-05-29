import torch
import os
import cv2
import math
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
# import colmap_read_model as read_model
# pip install pycolmap

###################################################
"""
MATRICE PROJECTION + TRANSLATION
"""
def KM(F):
    return torch.Tensor([
        [F, 0, 0, 0],
        [0, F, 0, 0],
        [0, 0, F, 0],
        [0, 0, -1, F]
    ]).float()

def Km(s, pix, cx, cy):
    return torch.Tensor([
        [(s/pix), 0, (cx/pix)],
        [0, (s/pix), (cy/pix)],
        [0, 0, 1]
    ]).float()

def Tm(F, M, cx, cy):
    return torch.Tensor([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, F + M]
    ]).float()

def Tm_4x4_v1(rotation, translation):

    qw,qx,qy,qz = rotation
    tx,ty,tz = translation

    if qx == 0 and qy == 0 and qz == 0 and qw == 0:
        rotation_matrix_w2c = np.eye(3)
    else:
        rotation_matrix_w2c = R.from_quat([qx, qy, qz, qw]).as_matrix()
    
    Tm = torch.eye(4)
    Tm[:3, :3] = torch.tensor(rotation_matrix_w2c, dtype=torch.float32)
    Tm[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float32)

    return Tm


def Tm_4x4_v2(rotation, translation):

    qw,qx,qy,qz = rotation
    tx,ty,tz = translation

    if qx == 0 and qy == 0 and qz == 0 and qw == 0:
        rotation_matrix_w2c = np.eye(3)
    else:
        rotation_matrix_w2c = R.from_quat([qx, qy, qz, qw]).as_matrix()
    
    w2c = np.eye(4)
    w2c[:3, :3] = rotation_matrix_w2c
    w2c[:3, 3]  = np.array([tx, ty, tz])

    c2w = np.linalg.inv(w2c)

    # Correction possible
    c2w[:3, 0] *= -1 # inversion de l'axe x
    c2w[:3, 1] *= -1 # inversion de l'axe y
    # c2w[:3, 2] *= -1 # inversion de l'axe z

    pose_c2w = torch.tensor(c2w, dtype=torch.float32)


    return pose_c2w


###################################################
"""
CHARGEMENT DES DONNEES:
"""
def load_camera_params(camera_file):
    
    params = {}

    with open(camera_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.split()

        if len(parts) > 0:
            params['camera_id'] = int(parts[0])
            params['model'] = (parts[1])
            params['width'] = int(parts[2])
            params['height'] = int(parts[3])
            params['mla_center_x'] = float(parts[4])
            params['mla_center_y'] = float(parts[5])
            params['mla_rotation'] = float(parts[6])
            params['mla_diameter'] = float(parts[7])
            params['pix'] = float(parts[8]) 
            params['F'] = float(parts[9])
            params['M'] = float(parts[10])
            params['s'] = float(parts[11])
            params['cx'] = float(parts[12])
            params['cy'] = float(parts[13])

    return params

def load_image_data(image_file):
    image_data = []
    with open(image_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) > 0:
            image_info = {
                'image_id': int(parts[0]),
                'qw': float(parts[1]),
                'qx': float(parts[2]),
                'qy': float(parts[3]),
                'qz': float(parts[4]),
                'tx': float(parts[5]),
                'ty': float(parts[6]),
                'tz': float(parts[7]),
                'camera_id': int(parts[8]),
                'image_name': parts[9]
            }
            image_data.append(image_info)
    
    return image_data


def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

#####################################################################

"""
GENERATION DES MATRICES ET DES POSES:
"""
def load_plenoptic_camera_matrices(camera_file):
    params = load_camera_params(camera_file)
    print("Params loaded:", params)
    
    F = params['F']
    M = params['M']
    s = params['s']
    pix = params['pix']
    cx = params['cx']
    cy = params['cy']
    
    KM_matrix = KM(F)
    Km_matrix = Km(s, pix, cx, cy)
    Tm_matrix = Tm(F, M, cx, cy)    
    return KM_matrix, Km_matrix, Tm_matrix


def get_number_lens(width, height, mla_diameter, rotation):

    if rotation == 0:
        num_lenses_x = math.floor(width / mla_diameter)
        num_lenses_y = math.floor(height / (mla_diameter * math.sqrt(3)/2))
        
    elif rotation == 90:
        num_lenses_x = math.floor(width / (mla_diameter * math.sqrt(3)/2))
        num_lenses_y = math.floor(height / mla_diameter)

       
    else:
        raise ValueError("La rotation doit être de 0 ou 90 degrés.")
    
    return num_lenses_x, num_lenses_y


def generate_poses(F, M, center_x, center_y, mla_diameter, num_lenses_x, num_lenses_y, rotation, camera_rotation, camera_translation):
    poses = []
    micro_lens_positions = []
    if rotation == 0:
        vertical_spacing = mla_diameter * math.sqrt(3) / 2
        horizontal_spacing = mla_diameter
        for row in range(num_lenses_y):
            y_ij = center_y - (num_lenses_y // 2) * vertical_spacing + row * vertical_spacing
            for col in range(num_lenses_x):
                x_ij = center_x - (num_lenses_x//2) * horizontal_spacing + col * horizontal_spacing
                if row % 2 == 1:
                    x_ij += horizontal_spacing / 2

                micro_lens_positions.append((x_ij, y_ij))
                pose = Tm_4x4_v1(camera_rotation, camera_translation)
                poses.append(pose)
    
    elif rotation == 90:
        horizontal_spacing = mla_diameter * math.sqrt(3) / 2
        vertical_spacing = mla_diameter
        for col in range(num_lenses_x):
            x_ij = center_x - (num_lenses_x//2) * horizontal_spacing + col * horizontal_spacing
            for row in range(num_lenses_y):
                y_ij = center_y - (num_lenses_y // 2) * vertical_spacing + row * vertical_spacing
                if col % 2 == 1:
                    y_ij += vertical_spacing / 2

                micro_lens_positions.append((x_ij, y_ij))
                pose = Tm_4x4_v1(camera_rotation, camera_translation)
                poses.append(pose)
    else:
        raise ValueError("La rotation doit être soit 0° (décalage horizontal), soit 90° (décalage vertical).")
    
    return micro_lens_positions, poses

#######################################################################
"""
DECOUPAGE DE MON IMAGE EN MICRO-IMAGES
"""
def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist2 = (X - center[0])**2 + (Y - center[1])**2
    mask = dist2 <= radius**2
    return mask


def cut_micro_images(image, mla_centers, lens_diameter, output_folder, image_number):
    target_size = math.floor(lens_diameter)
    img = image
    os.makedirs(output_folder, exist_ok=True)
    image_height, image_width = img.shape[:2]
    lens_diameter = math.floor(lens_diameter)
    lens_radius = lens_diameter / 2
    micro_images = []

    i = 0
    
    for (cx, cy) in mla_centers:
        x_start = max(0, int(cx - lens_radius))
        y_start = max(0, int(cy - lens_radius))
        x_end = min(image_width, int(cx + lens_radius))
        y_end = min(image_height, int(cy + lens_radius))

        micro_image = img[y_start:y_end, x_start:x_end]  
        micro_image = cv2.resize(micro_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        micro_image_with_alpha = cv2.cvtColor(micro_image, cv2.COLOR_BGR2RGBA)
        mask = create_circular_mask(micro_image.shape[0], micro_image.shape[1], (lens_radius, lens_radius), lens_radius)

        micro_image_with_alpha[~mask] = [0, 0, 0, 0]
        micro_images.append(micro_image_with_alpha)
        micro_image_filename = f"{output_folder}/micro_image__{image_number}_{i}.png"
        cv2.imwrite(micro_image_filename, micro_image_with_alpha)

        i += 1 

    return micro_images

#######################################################################
"""
DECOUPAGE DE MES MICRO-IMAGE EN TRAIN_SET & TEST_SET
"""
def save_indices_to_txt(i_train, i_test, filename):
    with open(filename, "w") as f:
        f.write(",".join(map(str, i_train.tolist())) + "\n")
        f.write(",".join(map(str, i_test.tolist())) + "\n")

def load_indices_from_txt(filename):
    if not os.path.exists(filename):
        return None, None
    with open(filename, "r") as f:
        lines = f.readlines()
        i_train = list(map(int, lines[0].strip().split(","))) if lines[0].strip() else []
        i_test = list(map(int, lines[1].strip().split(","))) if lines[1].strip() else []
    return torch.tensor(i_train), torch.tensor(i_test)


def split_train_test(micro_images, test_set, train_set, test_ratio, indices_file):

    num_total_images = micro_images.shape[0]
    all_indices = list(range(num_total_images))

    i_train, i_test = load_indices_from_txt(indices_file)
    if i_train is not None and i_test is not None:
        print("Chargement des indices depuis le fichier existant...")
        return i_train, i_test

    num_test_images = int(num_total_images * test_ratio)

    test_indices = random.sample(all_indices, num_test_images)
    train_indices = [i for i in all_indices if i not in test_indices]

    test_images = micro_images[test_indices]
    train_images = micro_images[train_indices]

    os.makedirs(train_set, exist_ok=True)
    os.makedirs(test_set, exist_ok=True)

    for i, img_array in enumerate(train_images):
        img = Image.fromarray(img_array.astype(np.uint8))
        img_name = f"train_image_{i}.png"
        img.save(os.path.join(train_set, img_name))
    for i, img_array in enumerate(test_images):
        img = Image.fromarray(img_array.astype(np.uint8))
        img_name = f"test_image_{i}.png"
        img.save(os.path.join(test_set, img_name))

    i_train = torch.tensor(train_indices)
    i_test  = torch.tensor(test_indices)

    save_indices_to_txt(i_train, i_test, indices_file)

                                        
    return i_train, i_test

#############################################################
"""
ENCAPS
"""
def load_pleno_data(camera_file, image_file, images_folder, micro_images_folder, test_set, train_set, indices_file):
     
    params = load_camera_params(camera_file)
    width, height = params['width'], params['height']
    mla_diameter = params['mla_diameter']
    mla_rotation = params['mla_rotation']
    center_x, center_y = params['mla_center_x'], params['mla_center_y']
    F, M, s, pix = params['F'], params['M'], params['s'], params['pix']

    rotation = int(math.degrees(mla_rotation))
    test_ratio = 0.1
    i = 1

    num_lenses_x, num_lenses_y = get_number_lens(width, height, mla_diameter, rotation)
    
    camera_poses = {}
    with open(image_file, 'r') as f:
        next(f)
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 10:
                continue
            image_name = tokens[-1]
            qw, qx, qy, qz = map(float, tokens[1:5]) 
            tx, ty, tz = map(float, tokens[5:8])
            camera_poses[image_name] = {'rotation': (qw, qx, qy, qz), 'translation': (tx, ty, tz)}
    
    all_micro_images = []
    all_poses = []
    all_mla_centers = []

    for img_name in sorted(os.listdir(images_folder)):
 
        img_path = os.path.join(images_folder, img_name)
        if img_name not in camera_poses:
            continue
        
        color_image = np.array(Image.open(img_path))
      
        camera_rotation = camera_poses[img_name]['rotation']
        camera_translation = camera_poses[img_name]['translation']

        mla_centers, poses = generate_poses(F, M, center_x, center_y, mla_diameter, num_lenses_x, num_lenses_y, rotation,
                                            camera_rotation, camera_translation)

        micro_images = cut_micro_images(color_image, mla_centers, mla_diameter, micro_images_folder, i)
        all_micro_images.append(micro_images)
        all_poses.append(poses)
        all_mla_centers.append(mla_centers)
        i += 1

    all_micro_images = np.array(all_micro_images)
    
    num_categories, num_images, height, width, channels = all_micro_images.shape 
    all_micro_images = all_micro_images.reshape(-1, height, width, channels)


    i_train, i_test = split_train_test(all_micro_images, test_set, train_set, test_ratio, indices_file)
    all_micro_images = np.array(all_micro_images).astype(np.float32)/255.0
    micro_images = torch.tensor(all_micro_images) 

    intrinsic_params= {

        'F':F,
        'M':M,
        's':s,
        'pix':pix,
        'mla_centers':all_mla_centers,
        'lens_diameter':mla_diameter
    }
    return micro_images, all_poses, intrinsic_params, i_test, i_train

def main():

    camera_file = 'param/fujita_cameras.txt'
    image_file = 'param/fujita_images.txt'
    images_folder = 'color/NagoyaFujita_1im'
    output_folder = './1im/micro-images'

    test_set = './1im/output/test'
    train_set = './1im/output/train'
    indices_file = './1im/output/indices.txt' 
    
    micro_images, poses, intrinsic_params, i_test, i_train = load_pleno_data(camera_file= camera_file, 
                                                                             image_file = image_file, 
                                                                             images_folder= images_folder, 
                                                                             micro_images_folder=output_folder, 
                                                                             test_set=test_set, 
                                                                             train_set=train_set, 
                                                                             indices_file=indices_file )

if __name__ == '__main__':
    main()
