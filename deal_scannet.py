import numpy as np
import os
import cv2
import sys
import re


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    print(filename)
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))
    # print("write pfm")

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


# 这里生成深度图
nerf_depth = "/home/jxh/NerfingMVS-dhn/logs/scene_000_test1/nerf/results"
output_dir = "/home/jxh/NerfingMVS-dhn/depth_fused/scene_000_test1/depth"
all_depth = os.listdir(nerf_depth)
all_depth.sort()
h, w = 0, 0

i = 0
for depth in all_depth:
    if depth.endswith('.png') or depth.endswith('.txt'):
        continue

    ori_depth = os.path.join(nerf_depth, depth)

    # 处理原始scannet的部分

    # depth_image = cv2.imread(ori_depth, cv2.IMREAD_UNCHANGED)
    # h, w =depth_image.shape
    # depth_image = depth_image.astype(np.float32)
    # print(ori_depth)

    depth_image, _ = read_pfm(ori_depth)
    depth_image *= 1000

    # .pfm版本
    depth_name = format(str(i), "0>8s") + ".pfm"
    depth_name = os.path.join(output_dir, depth_name)
    print(depth_name)
    save_pfm(depth_name, depth_image)
    i += 1

#这里处理pose
pose_dir = "/home/jxh/NerfingMVS-dhn/data/scene_000/pose"
output_dir = "/home/jxh/NerfingMVS-dhn/depth_fused/scene_000_test1/cams"
all_pose = os.listdir(pose_dir)
all_pose.sort()

intrinsic_dir = "/home/jxh/NerfingMVS-dhn/data/scene_000/intrinsic_color.txt"

intrinsic = ""

with open(intrinsic_dir, 'r') as f:
    intrinsic = f.readlines()
    intrinsic = [line.rstrip() for line in intrinsic]
    intrinsic = np.fromstring(" ".join(intrinsic), dtype=np.float32, sep=" ").reshape(4,4)
    intrinsic = intrinsic[:3,:3]

i = 0
for pose in all_pose:
    with open(os.path.join(pose_dir, pose), 'r') as f:
        P = f.readlines()
        cams_name =  format(str(i), "0>8s") + "_cam.txt"
        with open(os.path.join(output_dir, cams_name), 'w') as F:
            F.write("extrinsic" + '\n')
            for line in P:
                F.write(line)
            F.write('\n')
            F.write('intrinsic'+ '\n')
            np.savetxt(F, intrinsic)
            # for line in intrinsic:
            #     F.write(str(line))
            F.write('\n')
            F.write("0.0 0.0" + '\n')
    i+=1

i = 0
nerf_rgb = "/home/jxh/NerfingMVS-dhn/logs/scene_000_test1/nerf/results"
img_files = os.listdir(nerf_rgb)
output_dir = "/home/jxh/NerfingMVS-dhn/depth_fused/scene_000_test1/images"
img_files.sort()

for image in img_files:
    if image.endswith('depth.png') or image.endswith('.txt') or image.endswith('.pfm'):
        continue
    ori_rgb = os.path.join(nerf_rgb, image)
    rgb_name = format(str(i), "0>8s") + ".jpg"
    color_image_dir = os.path.join(output_dir, rgb_name)
    rgb = cv2.imread(ori_rgb)
    # rgb = cv2.resize(rgb, (w,h))
    cv2.imwrite(color_image_dir, rgb)
    i += 1