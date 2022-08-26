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
import pickle
import zstandard as zstd

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

def emb_pose_generator(numpy_left_img, numpy_depth_img):

    left_linear = (np.load(numpy_left_img))
    left_linear = cv2.cvtColor(left_linear, cv2.COLOR_BGR2RGB)
    img_vis = left_linear
    actual_depth = (np.load(numpy_depth_img)).astype(int)
    depth = np.array(actual_depth, dtype=np.float32) / 255.0

    input = create_input_norm(left_linear, depth)[None, :, :, :]

    auto_encoder_path = os.path.join(hparams.data_dir, 'ae_checkpoints', 'model_50_nocs.pth')
    ae = get_auto_encoder(auto_encoder_path)
        
    if use_gpu:
        input = input.to(torch.device('cuda:0'))
    seg_output, _, _ , pose_output = model.forward(input)
    seg_pred = seg_output.get_prediction()
    print(seg_pred[0].shape, seg_pred[0])
    img = seg_output.get_visualization_img(np.load(numpy_left_img))
    cv2.imwrite('masks.png',img)
    # cv2.waitKey(0)
    seg_pred = np.argmax(seg_pred, axis=0).astype(np.uint8)

    with torch.no_grad():
        latent_emb_outputs, abs_pose_outputs, peak_output, scores, indices = pose_output.compute_pointclouds_and_poses(min_confidence,is_target = False)

    # print(seg_pred)
    pred_cls_ids = []
    print(indices)
    for indice in indices:
        pred_cls_ids.append(seg_pred[indice[0], indice[1]])
    pred_scores = scores
    print(pred_cls_ids, pred_scores)


    return ae, latent_emb_outputs, abs_pose_outputs, peak_output, img_vis, depth


def shape_decoder(ae, latent_emb_outputs, abs_pose_outputs, our_k=True):
    if our_k:
        camera_k = np.diag((0,0,0,1))
        camera_k[:3,:3] = np.load("640_480/rgb_k.npy")
        # camera_k[0,0] = 580#386.786
        # camera_k[1,1] = 580#386.786
        # camera_k[0,2] = 320#317.492
        # camera_k[1,2] = 240#248.319
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
        np.save("object"+str(j)+".npy", shape_out)
        rotated_pc, rotated_box, _ = get_gt_pointclouds(abs_pose_outputs[j], shape_out, camera_model = _CAMERA)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(rotated_pc)
        print("rotated_pc", rotated_pc.shape)
        np.save("rot_object"+str(j)+".npy", rotated_pc)
        # np.savetxt("rot_object"+str(j)+".xyz", rotated_pc)
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

    return rotated_pcds, points_2d, box_obb, axes

def project_pcd_2_img(img_vis, points_2d, box_obb, axes):
    color_img = np.copy(img_vis)
    projected_points_img = show_projected_points(color_img, points_2d)
    colors_box = [(63, 237, 234)]
    im = np.array(np.copy(img_vis)).copy()
    for k in range(len(colors_box)):
        for points_2d, axis in zip(box_obb, axes):
            points_2d = np.array(points_2d)
            im = draw_bboxes(im, points_2d, axis, colors_box[k])

    plt.gca() #.invert_yaxis()
    plt.axis('off')
    plt.imshow(im[...,::-1])
    plt.savefig('my_projection.png')
    return


def decompress_datapoint(cbuf):
    cctx = zstd.ZstdDecompressor()
    buf = cctx.decompress(cbuf)
    x = pickle.loads(buf)
    return x


def _datapoint_path(dataset_path, uid):
    return f'{dataset_path}/{uid}.pickle.zstd'

def read(dataset_path, uid):
    path = _datapoint_path(dataset_path, uid)
    with open(path, 'rb') as fh:
      dp = decompress_datapoint(fh.read())
    # TODO: remove this, once old datasets without UID are out of use
    if not hasattr(dp, 'uid'):
      dp.uid = uid
    assert dp.uid == uid
    return dp


def main(numpy_left_img, numpy_depth_img):
    ae, latent_emb_outputs, abs_pose_outputs, peak_output, img_vis, depth = emb_pose_generator(numpy_left_img, numpy_depth_img)
    ### 2.1 Visualize Peaks output and Depth output
    display_gird(img_vis, depth, peak_output)
    ## 2.2 Decode shape from latent embeddings
    rotated_pcds, points_2d, box_obb, axes = shape_decoder(ae, latent_emb_outputs, abs_pose_outputs, our_k=True)
    ## draw pcds
    # draw_geometries(rotated_pcds)
    ## 2.3 Project 3D Pointclouds and 3D bounding boxes on 2D image
    project_pcd_2_img(img_vis, points_2d, box_obb, axes)
    return


img_name = "3"
numpy_left_img = "640_480/rgb_more/rgb_" + img_name + ".npy"
numpy_depth_img = "640_480/depth_more/depth_" + img_name +".npy"

# main(numpy_left_img, numpy_depth_img)

### Data extraction from the dataset
# test_data = read('test', '2AvmqwfHvbpiuhG68xjipa')
test_data = read('test', '2DmwnekiwnQQ5msHeZDf5X')

print(test_data.stereo.left_color.shape)
cv2.imwrite('stereo_left.png',test_data.stereo.left_color)
print(test_data.depth.shape)
# cv2.imwrite('test_depth.png', img)
plt.imshow(test_data.depth)
plt.savefig('test_depth.png')
print(test_data.segmentation.shape)
plt.imshow(test_data.segmentation)
plt.savefig('test_seg.png')
print(test_data.instance_mask.shape)
