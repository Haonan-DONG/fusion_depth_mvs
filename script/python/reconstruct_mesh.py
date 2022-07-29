import sys
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import colmap2mvsnet as IO3D
import os


class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj


def create_mesh_from_rgbd_dataset(path):
    camera_poses = read_trajectory(os.path.join(path, "trajectory.log"))

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 / 512.0,
        sdf_trunc=0.02,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    rgb_files = os.listdir(os.path.join(path, "color"))
    rgb_files.sort()

    depth_files = os.listdir(os.path.join(path, "depth"))
    depth_files.sort()

    data_number = len(depth_files)

    for i in range(data_number):
        if (i % 5 != 0):
            continue
        print("Integrate {:d}-th image into the volume.".format(i))
        color = o3d.io.read_image(os.path.join(path, "color", rgb_files[i]))
        depth = o3d.io.read_image(os.path.join(path, "depth", depth_files[i]))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(256, 192, 208.834, 208.834, 128,
                                              96),
            np.linalg.inv(camera_poses[i].pose))

    print("Extract a triangle mesh from the volume and visualize it.")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("mesh.ply", mesh)

    # o3d.visualization.draw_geometries([mesh],
    #                                   front=[0.5297, -0.1873, -0.8272],
    #                                   lookat=[2.0712, 2.0312, 1.7251],
    #                                   up=[-0.0558, -0.9809, 0.1864],
    #                                   zoom=0.47)


# =============== Read Single rgb-d point cloud ================
def read_single_rgbd():
    print("Read Redwood dataset")
    color_raw = o3d.io.read_image(
        "/home/mvs18/Desktop/dhn/precise_reconstruction_rgbd/data/output_img/261RGB.jpg"
    )
    depth_raw = o3d.io.read_image(
        "/home/mvs18/Desktop/dhn/precise_reconstruction_rgbd/data/output_img/263Depth.jpg"
    )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_trunc=10)
    print(rgbd_image)
    #
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(320, 180, 514.61963, 514.61963, 160,
                                          90))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.io.write_point_cloud("copy_of_fragment.ply", pcd)
    # o3d.visualization.draw_geometries([pcd], zoom=0.35)


if __name__ == "__main__":
    print("Start to processing RGB-D data.\n")
    path = sys.argv[1]
    #create_mesh_from_rgbd_dataset(path)
    read_single_rgbd()
