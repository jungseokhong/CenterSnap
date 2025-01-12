import argparse
import pathlib
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import matplotlib.pyplot as plt
import os
import time
import pytorch_lightning as pl
import _pickle as cPickle
import os, sys
from simnet.lib.net import common
from simnet.lib import camera
from simnet.lib.net.panoptic_trainer import PanopticModel
from simnet.lib.net.models.auto_encoder import PointCloudAE
from utils.nocs_utils import load_img_NOCS, create_input_norm
from utils.viz_utils import depth2inv, viz_inv_depth
from utils.transform_utils import get_gt_pointclouds, transform_coordinates_3d, calculate_2d_projections
from utils.transform_utils import project
from utils.viz_utils import save_projected_points, draw_bboxes, line_set_mesh, display_gird, draw_geometries, show_projected_points

## 1. Instantiate CenterSnap Model

sys.argv = ['', '@configs/net_config.txt']
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
common.add_train_args(parser)
app_group = parser.add_argument_group('app')
app_group.add_argument('--app_output', default='inference', type=str)
app_group.add_argument('--result_name', default='centersnap_nocs', type=str)
app_group.add_argument('--data_dir', default='nocs_test_subset', type=str)
hparams = parser.parse_args()
min_confidence = 0.50
use_gpu=False
hparams.checkpoint = 'nocs_test_subset/checkpoint/centersnap_real.ckpt'
model = PanopticModel(hparams, 0, None, None)
model.eval()
if use_gpu:
    model.cuda()
data_path = open(os.path.join(hparams.data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
_CAMERA = camera.NOCS_Real()

def get_auto_encoder(model_path):
    emb_dim = 128
    n_pts = 2048
    ae = PointCloudAE(emb_dim, n_pts)
    ae.cuda()
    ae.load_state_dict(torch.load(model_path))
    ae.eval()
    return ae

def load_depth(depth_path):
    """Load depth image from img_path."""
    # depth_path = depth_path + '_depth.png'
    # print("depth_path", depth_path)
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1] * 256 + depth[:, :, 2]
        depth16 = np.where(depth16 == 32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == "uint16":
        depth16 = depth
    else:
        assert False, "[ Error ]: Unsupported depth type."
    return depth16

## 2. Perform inference using NOCS Real Subset

# RUN THIS ONE FOR OUR DATA
num = 0
img_name = "3"

left_linear = (np.load("640_480/rgb/rgb_" + img_name + ".npy"))
left_linear = cv2.cvtColor(left_linear, cv2.COLOR_BGR2RGB)
img_vis = left_linear

actual_depth = (np.load("640_480/depth/depth_" + img_name +".npy")).astype(int)
depth = np.array(actual_depth, dtype=np.float32) / 255.0

input = create_input_norm(left_linear, depth)[None, :, :, :]

auto_encoder_path = os.path.join(hparams.data_dir, 'ae_checkpoints', 'model_50_nocs.pth')
ae = get_auto_encoder(auto_encoder_path)
    
if use_gpu:
    input = input.to(torch.device('cuda:0'))
_, _, _ , pose_output = model.forward(input)
with torch.no_grad():
    latent_emb_outputs, abs_pose_outputs, peak_output, _, _ = pose_output.compute_pointclouds_and_poses(min_confidence,is_target = False)

### 2.1 Visualize Peaks output and Depth output

display_gird(img_vis, depth, peak_output)

## 2.2 Decode shape from latent embeddings

our_k = True

if our_k:
  camera_k = np.diag((0,0,0,1))
  camera_k[:3,:3] = np.load("640_480/rgb_k.npy")
  # camera_k[0,0] = 386.786
  # camera_k[1,1] = 386.786
  # camera_k[0,2] = 317.492
  # camera_k[1,2] = 248.319
  # camera_k[2,2] = 1
  print(camera_k)

else:
  camera_k = _CAMERA.K_matrix
  print(camera_k)


write_pcd = False
rotated_pcds = []
points_2d = []
box_obb = []
axes = []
colors_array = []
boxes = []
for j in range(len(latent_emb_outputs)):
    emb = latent_emb_outputs[j]
    emb = latent_emb_outputs[j]
    emb = torch.FloatTensor(emb).unsqueeze(0)
    emb = emb.cuda()
    _, shape_out = ae(None, emb)
    shape_out = shape_out.cpu().detach().numpy()[0]
    rotated_pc, rotated_box, _ = get_gt_pointclouds(abs_pose_outputs[j], shape_out, camera_model = _CAMERA)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rotated_pc)
    print("rotated_pc", rotated_pc.shape)
    rotated_pcds.append(pcd)
    pcd.paint_uniform_color((1.0, 0.0, 0.0))
    colors_array.append(pcd.colors)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    T = abs_pose_outputs[j].camera_T_object
    mesh_frame = mesh_frame.transform(T)
    rotated_pcds.append(mesh_frame)
    cylinder_segments = line_set_mesh(rotated_box)
    for k in range(len(cylinder_segments)):
      rotated_pcds.append(cylinder_segments[k])
    points_mesh = camera.convert_points_to_homopoints(rotated_pc.T)
    points_2d_mesh = project(camera_k, points_mesh)
    points_2d_mesh = points_2d_mesh.T
    points_2d.append(points_2d_mesh)
    #2D output
    points_obb = camera.convert_points_to_homopoints(np.array(rotated_box).T)
    points_2d_obb = project(camera_k, points_obb)
    points_2d_obb = points_2d_obb.T
    box_obb.append(points_2d_obb)
    xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
    sRT = abs_pose_outputs[j].camera_T_object @ abs_pose_outputs[j].scale_matrix
    transformed_axes = transform_coordinates_3d(xyz_axis, sRT)
    projected_axes = calculate_2d_projections(transformed_axes, camera_k[:3,:3])
    axes.append(projected_axes)

draw_geometries(rotated_pcds)


## 2.3 Project 3D Pointclouds and 3D bounding boxes on 2D image

color_img = np.copy(img_vis) 
print(color_img.shape, len(points_2d))
projected_points_img = show_projected_points(color_img, points_2d)
colors_box = [(63, 237, 234)]
im = np.array(np.copy(img_vis)).copy()
for k in range(len(colors_box)):
    for points_2d, axis in zip(box_obb, axes):
        points_2d = np.array(points_2d)
        im = draw_bboxes(im, points_2d, axis, colors_box[k])

plt.gca().invert_yaxis()
plt.axis('off')
plt.imshow(im[...,::-1])
plt.savefig('my_projection.png')
