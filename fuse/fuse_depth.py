import argparse, os, time, sys, gc, cv2
from ast import arg
import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image

from data_io import read_pfm, save_pfm

mathematics_pytorch_tutorial

from multiprocessing import Pool
from functools import partial
import signal

parser = argparse.ArgumentParser(description="Predict depth, filter, and fuse")

parser.add_argument("--dense_folder", default="./outputs", help="output dir")
parser.add_argument("--name", default="scene_01", help="output dir")
parser.add_argument("--fusion", default="nerf", help="output dir")

parser.add_argument("--num_view", type=int, default=5, help="num of view")
parser.add_argument("--max_h", type=int, default=900, help="testing max h")
parser.add_argument("--max_w", type=int, default=1200, help="testing max w")
parser.add_argument("--fix_res",
                    action="store_true",
                    help="scene all using same res")

parser.add_argument("--num_worker",
                    type=int,
                    default=4,
                    help="depth_filer worker")
parser.add_argument("--save_freq",
                    type=int,
                    default=5000,
                    help="save freq of local pcd")

parser.add_argument(
    "--filter_method",
    type=str,
    default="normal",
    choices=["gipuma", "normal"],
    help="filter method",
)

# filter
parser.add_argument("--conf", type=float, default=0.9, help="prob confidence")
parser.add_argument("--thres_view",
                    type=int,
                    default=5,
                    help="threshold of num view")

# # filter by gimupa
# parser.add_argument("--fusibile_exe_path",
#                     type=str,
#                     default="../fusibile/fusibile")
# parser.add_argument("--prob_threshold", type=float, default="0.9")
# parser.add_argument("--disp_threshold", type=float, default="0.25")
# parser.add_argument("--num_consistent", type=float, default="4")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32,
                               sep=" ").reshape((4, 4))

    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(" ".join(lines[7:10]),
                               dtype=np.float32,
                               sep=" ").reshape((3, 3))

    intrinsics[:2, :] /= 7.5

    # intrinsics[:2, :] /= 2
    return intrinsics, extrinsics


# read an image
def read_img(filename, is_nerf=False):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    w, h = img.size
    if is_nerf:
        w //= 2
        h //= 2
    else:
        w *= 2
        h *= 2
        w //= 15
        h //= 15

    np_img = np.array(img, dtype=np.float32) / 255.0
    np_img = cv2.resize(np_img, (w, h))
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


def write_cam(file, cam):
    f = open(file, "w")
    f.write("extrinsic\n")
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + " ")
        f.write("\n")
    f.write("\n")

    f.write("intrinsic\n")
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + " ")
        f.write("\n")

    f.write("\n" + str(cam[1][3][0]) + " " + str(cam[1][3][1]) + " " +
            str(cam[1][3][2]) + " " + str(cam[1][3][3]) + "\n")

    f.close()


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src,
                         intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(
        np.linalg.inv(intrinsics_ref),
        np.vstack(
            (x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]),
    )
    # source 3D space
    xyz_src = np.matmul(
        np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
        np.vstack((xyz_ref, np.ones_like(x_ref))),
    )[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src,
                                  x_src,
                                  y_src,
                                  interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(
        np.linalg.inv(intrinsics_src),
        np.vstack(
            (xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]),
    )
    # reference 3D space
    xyz_reprojected = np.matmul(
        np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
        np.vstack((xyz_src, np.ones_like(x_ref))),
    )[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height,
                                                    width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height,
                                               width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height,
                                               width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def generate_pointcloud(rgb, depth, ply_file, intr, scale=1.0):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.

    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file

    """
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    points = []
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v, u]  # rgb.getpixel((u, v))
            Z = depth[v, u] / scale
            if Z == 0:
                continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append("%f %f %f %d %d %d 0\n" %
                          (X, Y, Z, color[0], color[1], color[2]))
    file = open(ply_file, "w")
    file.write("""ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            """ % (len(points), "".join(points)))
    file.close()
    print("save ply, fx:{}, fy:{}, cx:{}, cy:{}".format(fx, fy, cx, cy))


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref,
                                depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    (
        depth_reprojected,
        x2d_reprojected,
        y2d_reprojected,
        x2d_src,
        y2d_src,
    ) = reproject_with_depth(
        depth_ref,
        intrinsics_ref,
        extrinsics_ref,
        depth_src,
        intrinsics_src,
        extrinsics_src,
    )
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref)**2 + (y2d_reprojected - y_ref)**2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(pair_folder, scan_folder, out_folder, plyfilename):
    # the pair file
    if os.path.exists(os.path.join(scan_folder, "pair.txt")):
        use_pair = True
        pair_file = os.path.join(scan_folder, "pair.txt")
    else:
        use_pair = False

    # for the final point cloud
    vertexs = []
    vertex_colors = []

    is_nerf = args.fusion == "nerf"

    if use_pair:
        pair_data = read_pair_file(pair_file)
        nviews = len(pair_data)
    else:
        pair_data = []
        files = os.listdir(os.path.join(scan_folder, "images"))
        nviews = len(files)
        for view_idx in range(nviews):
            if view_idx == 0:
                src_views = [1, 2, 3, 4, 5, 6]
                pair_data.append((view_idx, src_views))
            elif view_idx == 1:
                src_views = [0, 2, 3, 4, 5, 6]
                pair_data.append((view_idx, src_views))
            elif view_idx == 2:
                src_views = [0, 1, 3, 4, 5, 6]
                pair_data.append((view_idx, src_views))
            elif view_idx == 3:
                src_views = [0, 1, 2, 4, 5, 6]
                pair_data.append((view_idx, src_views))
            elif view_idx == nviews - 1:
                src_views = [
                    nviews - 7, nviews - 6, nviews - 5, nviews - 4, nviews - 3,
                    nviews - 2
                ]
                pair_data.append((view_idx, src_views))
            elif view_idx == nviews - 2:
                src_views = [
                    nviews - 7, nviews - 6, nviews - 5, nviews - 4, nviews - 3,
                    nviews - 1
                ]
                pair_data.append((view_idx, src_views))
            elif view_idx == nviews - 3:
                src_views = [
                    nviews - 7, nviews - 6, nviews - 5, nviews - 4, nviews - 2,
                    nviews - 1
                ]
                pair_data.append((view_idx, src_views))
            elif view_idx == nviews - 4:
                src_views = [
                    nviews - 7, nviews - 6, nviews - 5, nviews - 3, nviews - 2,
                    nviews - 1
                ]
                pair_data.append((view_idx, src_views))
            else:
                src_views = [
                    view_idx - 3, view_idx - 2, view_idx - 1, view_idx + 1,
                    view_idx + 2, view_idx + 3
                ]
                pair_data.append((view_idx, src_views))

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # src_views = src_views[:args.num_view]
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, "cams/{:0>8}_cam.txt".format(ref_view)))
        # load the reference image
        ref_img = read_img(
            os.path.join(scan_folder, "images/{:0>8}.jpg".format(ref_view)),
            is_nerf)
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(
            os.path.join(out_folder, "depth/{:0>8}.pfm".format(ref_view)))[0]

        # ref_depth_est = cv2.resize(ref_depth_est, (1920, 1440))
        if ref_view % args.save_freq == 0:
            generate_pointcloud(
                ref_img * 255,
                ref_depth_est,
                os.path.join(out_folder, "local_{:0>8}.ply").format(ref_view),
                ref_intrinsics,
            )

        photo_mask = np.ones_like(ref_depth_est) > 0.5

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            if not (src_view % 10):
                continue
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder,
                             "cams/{:0>8}_cam.txt".format(src_view)))
            # the estimated depth of the source view
            src_depth_est = read_pfm(
                os.path.join(out_folder,
                             "depth/{:0>8}.pfm".format(src_view)))[0]
            # src_depth_est = cv2.resize(src_depth_est, (1920, 1440))

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(
                ref_depth_est,
                ref_intrinsics,
                ref_extrinsics,
                src_depth_est,
                src_intrinsics,
                src_extrinsics,
            )
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) +
                              ref_depth_est) / (geo_mask_sum + 1)
        # at least 3 source views matched
        geo_mask = geo_mask_sum >= args.thres_view
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)),
            photo_mask,
        )
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)),
            geo_mask)
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)),
            final_mask,
        )

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".
              format(
                  scan_folder,
                  ref_view,
                  photo_mask.mean(),
                  geo_mask.mean(),
                  final_mask.mean(),
              ))

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[
            valid_points]
        # color = ref_img[1:-16:4, 1::4, :][valid_points]  # hardcoded for DTU dataset

        color = ref_img[valid_points]

        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

        # # set used_mask[ref_view]
        # used_mask[ref_view][...] = True
        # for idx, src_view in enumerate(src_views):
        #     src_mask = np.logical_and(final_mask, all_srcview_geomask[idx])
        #     src_y = all_srcview_y[idx].astype(np.int)
        #     src_x = all_srcview_x[idx].astype(np.int)
        #     used_mask[src_view][src_y[src_mask], src_x[src_mask]] = True

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs],
                       dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex_colors = np.array(
        [tuple(v) for v in vertex_colors],
        dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )

    vertex_all = np.empty(len(vertexs),
                          vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, "vertex")
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def init_worker():
    """
    Catch Ctrl+C signal to termiante workers
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def pcd_filter_worker(scan):
    save_name = "{}.ply".format(scan)
    pair_folder = args.dense_folder
    scan_folder = args.dense_folder
    out_folder = args.dense_folder
    filter_depth(pair_folder, scan_folder, out_folder,
                 os.path.join(out_folder, save_name))


def pcd_filter(testlist, number_worker):

    partial_func = partial(pcd_filter_worker)

    p = Pool(number_worker, init_worker)
    try:
        p.map(partial_func, testlist)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()


if __name__ == "__main__":
    pcd_filter_worker(args.name)
    # pcd_filter(args.name, args.num_worker)
