import torch
from torch import nn, optim, utils
from typing import Any
import lightning.pytorch as pl
from lib.model.DSTformer import DSTformer
import lib.model.loss as losses
from functools import partial
from dataclasses import dataclass
import numpy as np
from collections import OrderedDict
import tqdm
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from trackman.posetools.data.datasets import AbstractImageDataset


def get_wandb_video_from_joints(joints, fps=30):
    frames = []
    for pose in joints:
        frames.append(visualize_3d_pose(pose))
    frames = np.asarray(frames)

    # Flip axis to (time, channel, height, width)
    frames = np.transpose(frames, (0, 3, 1, 2))

    return wandb.Video(frames, fps=fps, format="gif")


def visualize_3d_pose(j3d):
    """
    Plots a 3D skeleton.

    Args:
        j3d (np.array): 3D pose array of shape (J, 3).
    """
    # Viz colors
    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"

    # Human3.6M joint pairs
    joint_pairs = [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
        [0, 7],
        [7, 8],
        [8, 9],
        [8, 11],
        [8, 14],
        [9, 10],
        [11, 12],
        [12, 13],
        [14, 15],
        [15, 16],
    ]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]

    fig = plt.figure(0, figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=12.0, azim=80)
    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    for i in range(len(joint_pairs)):
        limb = joint_pairs[i]
        xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
        if joint_pairs[i] in joint_pairs_left:
            ax.plot(
                -xs,
                -zs,
                -ys,
                color=color_left,
                lw=3,
                marker="o",
                markerfacecolor="w",
                markersize=3,
                markeredgewidth=2,
            )  # axis transformation for visualization
        elif joint_pairs[i] in joint_pairs_right:
            ax.plot(
                -xs,
                -zs,
                -ys,
                color=color_right,
                lw=3,
                marker="o",
                markerfacecolor="w",
                markersize=3,
                markeredgewidth=2,
            )  # axis transformation for visualization
        else:
            ax.plot(
                -xs,
                -zs,
                -ys,
                color=color_mid,
                lw=3,
                marker="o",
                markerfacecolor="w",
                markersize=3,
                markeredgewidth=2,
            )  # axis transformation for visualization

    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Close figure to prevent memory leak
    plt.close(fig)

    return data


def coco2h36m(x):
    """
    COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}

    H36M:
    0: 'root',
    1: 'rhip',
    2: 'rkne',
    3: 'rank',
    4: 'lhip',
    5: 'lkne',
    6: 'lank',
    7: 'belly',
    8: 'neck',
    9: 'nose',
    10: 'head',
    11: 'lsho',
    12: 'lelb',
    13: 'lwri',
    14: 'rsho',
    15: 'relb',
    16: 'rwri'
    """
    if isinstance(x, torch.Tensor):
        y = torch.zeros_like(x)
    else:
        T, V, C = x.shape
        x = x[None, ...]
        y = np.zeros([1, T, 17, C])

    y[:, :, 0, :] = (x[:, :, 11, :] + x[:, :, 12, :]) * 0.5
    y[:, :, 1, :] = x[:, :, 12, :]
    y[:, :, 2, :] = x[:, :, 14, :]
    y[:, :, 3, :] = x[:, :, 16, :]
    y[:, :, 4, :] = x[:, :, 11, :]
    y[:, :, 5, :] = x[:, :, 13, :]
    y[:, :, 6, :] = x[:, :, 15, :]
    y[:, :, 8, :] = (x[:, :, 5, :] + x[:, :, 6, :]) * 0.5
    y[:, :, 7, :] = (y[:, :, 0, :] + y[:, :, 8, :]) * 0.5
    y[:, :, 9, :] = x[:, :, 0, :]
    y[:, :, 10, :] = (x[:, :, 1, :] + x[:, :, 2, :]) * 0.5
    y[:, :, 11, :] = x[:, :, 5, :]
    y[:, :, 12, :] = x[:, :, 7, :]
    y[:, :, 13, :] = x[:, :, 9, :]
    y[:, :, 14, :] = x[:, :, 6, :]
    y[:, :, 15, :] = x[:, :, 8, :]
    y[:, :, 16, :] = x[:, :, 10, :]

    return y


def _infer_box(
    pose3d: torch.tensor, fx: float, fy: float, cx: float, cy: float, rootIdx: int = 0
):
    """
    Infers the bounding box of a 3D pose in 2D space.

    Args:
        pose3d (torch.tensor): A tensor of shape (N, 3) representing the 3D pose.
        intrinsics (dict): A dictionary containing the camera intrinsics.
        rootIdx (int, optional): The index of the root joint. Defaults to 0.

    Returns:
        torch.tensor: A tensor of shape (4,) representing the bounding box in 2D space.
    """
    root_joint = pose3d[rootIdx, :]

    # Top left
    tl_joint = root_joint.clone()
    tl_joint[:2] -= 1000.0

    # Bottom right
    br_joint = root_joint.clone()
    br_joint[:2] += 1000.0
    tl_joint = torch.reshape(tl_joint, (1, 3))
    br_joint = torch.reshape(br_joint, (1, 3))

    # Project to 2D
    tl2d = _weak_project(tl_joint, fx, fy, cx, cy).flatten()

    br2d = _weak_project(br_joint, fx, fy, cx, cy).flatten()

    return torch.tensor([tl2d[0], tl2d[1], br2d[0], br2d[1]])


def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d


def camera2image_coordinates(
    pose3d: torch.tensor,
    box: torch.tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rootIdx: int = 0,
):
    rectangle_3d_size = 2000.0
    ratio = (box[2] - box[0] + 1) / rectangle_3d_size
    pose3d_image_frame = torch.zeros_like(pose3d)
    pose3d_image_frame[:, :2] = _weak_project(pose3d.clone(), fx, fy, cx, cy)
    pose3d_depth = ratio * (pose3d[:, 2] - pose3d[rootIdx, 2])
    pose3d_image_frame[:, 2] = pose3d_depth
    return pose3d_image_frame


def rotate_3d_pose_90_clockwise(
    pose3d: torch.tensor,
    numtimesrot90clockwise: int,
):
    """
    Rotates 3D pose in camera coordinates by 90 degrees clockwise while accounting for image dimensions.

    Args:
        pose3d (torch.tensor): 3D pose tensor of shape (S, J, 3).
        numtimesrot90clockwise (int): Number of times to rotate the pose by 90 degrees clockwise.

    Returns:
        torch.tensor: 3D pose tensor of shape (S, J, 3) after rotation.
    """
    pose3d_rotated = pose3d.clone()
    for i in range(numtimesrot90clockwise):
        x = pose3d_rotated[..., 0].clone()
        y = pose3d_rotated[..., 1].clone()
        pose3d_rotated[..., 0] = -y  # + image_height
        pose3d_rotated[..., 1] = x

    return pose3d_rotated


def sportspose2h36m(
    j3d_world_spostspose: torch.Tensor,
    camera_info: dict,
    times_rot90: int = 0,
    res: tuple = (1920, 1080),
    return_camera_and_image_coordinates: bool = False,
) -> torch.Tensor:
    """
    Converts 3D pose from world coordinates to camera coordinates and then to H36M coordinates.

    Args:
        j3d_world_spostspose (torch.Tensor): 3D pose in world coordinates, of shape (B, S, J, 3) or (S, J, 3) .
        camera_info (dict): Camera information dictionary containing extrinsic parameters.
        times_rot90 (int, optional): Number of times the image is rotated by 90 degrees. Defaults to 0.
        res (tuple, optional): Resolution of the image, of the form (width, height). Defaults to (1920, 1080).
        return_camera_and_image_coordinates (bool, optional): Whether to return the pose in camera and image coordinates. Defaults to False.

    Returns:
        torch.Tensor: 3D pose in H36M coordinates, of shape (B, S, J, 3).
    """
    # Convert to camera coordinates from world coordinates
    rot = camera_info["extrinsic"]["R"].to(j3d_world_spostspose)
    trans = camera_info["extrinsic"]["T"].to(j3d_world_spostspose)

    # Add batch dimension if not present to unify the code
    if len(j3d_world_spostspose.shape) != 4:
        j3d_world_spostspose = j3d_world_spostspose[None, ...]
    b, s, j, _ = j3d_world_spostspose.shape

    # Transform 3d pose to camera coordinate
    world_to_cam = torch.zeros((b, 4, 4), dtype=torch.float64).to(j3d_world_spostspose)
    world_to_cam[:, :3, :3] = rot
    world_to_cam[:, :3, 3] = trans
    world_to_cam[:, 3, 3] = 1.0

    # Pad the world coordinate to 4d homogenous coordinate
    ones_column = torch.ones(*j3d_world_spostspose.shape[:-1], 1).to(
        j3d_world_spostspose
    )
    j3d_world_hom = torch.cat([j3d_world_spostspose, ones_column], dim=-1)

    # Transform to camera coordinate
    j3d_camera = torch.matmul(world_to_cam, j3d_world_hom.view(b, -1, 4).mT).mT
    j3d_camera = (j3d_camera[..., :3] / j3d_camera[..., 3].unsqueeze(-1)).view(
        b, s, j, 3
    )

    # Scale from meter to millimeter
    j3d_camera = j3d_camera * 1000.0

    # Reorder coordinates to match H36M
    j3d_camera = coco2h36m(j3d_camera)

    # Go from camera coordinates to image coordinates
    j3d_image = torch.zeros_like(j3d_camera)
    j3d_image_org = torch.zeros_like(j3d_camera)
    j3d_scaled_image = j3d_image.clone()

    for batch_num in range(b):
        for frame in range(s):
            box = _infer_box(
                j3d_camera[batch_num, frame, :, :],
                fx=camera_info["intrinsic"]["f"][0, 0],
                fy=camera_info["intrinsic"]["f"][0, 1],
                cx=camera_info["intrinsic"]["c"][0, 0],
                cy=camera_info["intrinsic"]["c"][0, 1],
            )

            j3d_image[batch_num, frame] = camera2image_coordinates(
                j3d_camera[batch_num, frame, :, :],
                box,
                fx=camera_info["intrinsic"]["f"][0, 0],
                fy=camera_info["intrinsic"]["f"][0, 1],
                cx=camera_info["intrinsic"]["c"][0, 0],
                cy=camera_info["intrinsic"]["c"][0, 1],
                rootIdx=0,
            )
        j3d_image_org[batch_num, ...] = j3d_image[batch_num, ...].clone()
        res_w, res_h = res[batch_num, ...]
        if times_rot90[batch_num] % 2 == 1:
            res_h, res_w = res[batch_num, ...]
            j3d_image[batch_num, :, :, :] = rotate_3d_pose_90_clockwise(
                j3d_image[batch_num, :, :, :], 1
            )
            j3d_image[batch_num, :, :, 0] += res_w

        # Scale to be within [-1, 1]
        j3d_scaled_image[batch_num, :, :, :] = j3d_image[batch_num, :, :, :]
        j3d_scaled_image[batch_num, :, :, :2] = j3d_scaled_image[
            batch_num, :, :, :2
        ] / res_w * 2 - torch.tensor(
            [
                1,
                res_h / res_w,
            ]
        ).to(
            j3d_scaled_image
        )
        j3d_scaled_image[batch_num, :, :, 2:] = (
            j3d_scaled_image[batch_num, :, :, 2:] / res_w * 2
        )

    if return_camera_and_image_coordinates:
        return j3d_scaled_image.float(), j3d_camera.float(), j3d_image.float()
    return j3d_scaled_image.float()


def denormalize_dections(j3d, res, times_rot90):
    """
    Denormalizes model detections in order to be able to compute a MPJPE score.
    Note that it denormalizes the detections to millimeters and not meters as the original dataset.

    Args:
        j3d (torch.tensor): 3D pose tensor of shape (B, S, J, 3) or (S, J, 3).
        res (tuple): Image resolution of the form (width, height).
        times_rot90 (int): Number of times the image is rotated by 90 degrees.

    Returns:
        torch.tensor: 3D pose tensor of shape (S, J, 3) after denormalization.
    """
    j3d_denorm = j3d.clone()

    # If there is no batch dimension, add one to unify the code
    if len(j3d_denorm.shape) != 4:
        j3d_denorm = j3d_denorm[None, ...]

    for batch_num in range(j3d.shape[0]):
        if times_rot90[batch_num] % 2 == 1:
            res_h, res_w = res[batch_num, ...]
        else:
            res_w, res_h = res[batch_num, ...]

        j3d_denorm[batch_num, :, :, :2] = (
            (
                j3d_denorm[batch_num, :, :, :2]
                + torch.tensor([1, res_h / res_w]).to(j3d_denorm)
            )
            * res_w
            / 2
        )
        j3d_denorm[batch_num, :, :, 2:] = j3d_denorm[batch_num, :, :, 2:] * res_w / 2

    return j3d_denorm


def reorder_dataparallel_checkpoint(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


@dataclass
class TrainingConfig:
    lambda_scale: float = 1.0
    lambda_3d_velocity: float = 1.0
    lambda_lv: float = 1.0
    lambda_lg: float = 1.0
    lambda_a: float = 1.0
    lambda_av: float = 1.0


class MotionBertSportPose(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        debug_images=False,
        config=None,
        ortho_project=False,
        pretrain_path=None,
        view="FO",
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.model = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=512,
            dim_rep=512,
            depth=5,
            num_heads=8,
            mlp_ratio=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            maxlen=243,
            num_joints=17,
        )
        self.ortho_project = ortho_project

        if config is None:
            self.config = TrainingConfig()
        else:
            self.config = config

        if pretrain_path is not None:
            self.model.load_state_dict(
                reorder_dataparallel_checkpoint(torch.load(pretrain_path)["model_pos"]),
                strict=True,
            )

        self.debug_images = debug_images

        self.view = view

        # Init vars for validation and test metric calculation
        self.val_step_denorm_outputs = []
        self.val_step_gt = []
        self.test_step_denorm_outputs = []
        self.test_step_gt = []

    def training_step(self, batch, batch_idx):
        # Get camera and 3d pose in world coordinate
        target3d, j3d_cam, j3d_image = sportspose2h36m(
            batch["joints_3d"]["data_points"],
            batch["video"]["calibration"][self.view],
            batch["video"][self.view]["numtimesrot90clockwise"],
            batch["video"][self.view]["img_dims"],
            return_camera_and_image_coordinates=True,
        )

        # Ortho project 3D pose to 2D pose for 2D gt - set z to 1 as being confidence in 2D
        pose2d = target3d.clone()
        pose2d[..., 2] = 1.0

        # Augment / Mask 2D poses
        # TODO: Add augmentation and masking

        # Get 3D pose from model
        output = self.model(pose2d)

        # 3D Loses
        loss_3d_pos = losses.loss_mpjpe(output, target3d)
        loss_3d_scale = losses.n_mpjpe(output, target3d)
        loss_3d_velocity = losses.loss_velocity(output, target3d)
        loss_limb_variation = losses.loss_limb_var(output)
        loss_limb_gt = losses.loss_limb_gt(output, target3d)
        loss_angle = losses.loss_angle(output, target3d)
        loss_angle_velocity = losses.loss_angle_velocity(output, target3d)

        loss_total = (
            loss_3d_pos
            + self.config.lambda_scale * loss_3d_scale
            + self.config.lambda_3d_velocity * loss_3d_velocity
        )
        (
            +self.config.lambda_lv * loss_limb_variation
            + self.config.lambda_lg * loss_limb_gt
            + self.config.lambda_a * loss_angle
        )
        +self.config.lambda_av * loss_angle_velocity

        # Log 3D losses
        self.log("train/loss_3d_pos", loss_3d_pos, on_epoch=True)
        self.log("train/loss_3d_scale", loss_3d_scale, on_epoch=True)
        self.log("train/loss_3d_velocity", loss_3d_velocity, on_epoch=True)
        self.log("train/loss_limb_variation", loss_limb_variation, on_epoch=True)
        self.log("train/loss_limb_gt", loss_limb_gt, on_epoch=True)
        self.log("train/loss_angle", loss_angle, on_epoch=True)
        self.log("train/loss_angle_velocity", loss_angle_velocity, on_epoch=True)
        self.log("train/loss_total", loss_total, on_epoch=True)

        if self.trainer.is_last_batch:
            # Log sample of one of the predicted 3D poses from the trainset
            with torch.no_grad():
                # Log pose sequence as gif
                wandb_gif = get_wandb_video_from_joints(
                    output[0, :, :, :].cpu().numpy()
                )
                self.logger.experiment.log(
                    {
                        "train/pose3d_pred_seq": wandb_gif,
                    },
                )

        return loss_total

    def validation_step(self, batch, batch_idx):
        # Get camera and 3d pose in world coordinate
        target3d, j3d_cam, j3d_image = sportspose2h36m(
            batch["joints_3d"]["data_points"],
            batch["video"]["calibration"][self.view],
            batch["video"][self.view]["numtimesrot90clockwise"],
            batch["video"][self.view]["img_dims"],
            return_camera_and_image_coordinates=True,
        )

        # Ortho project 3D pose to 2D pose for 2D gt - set z to 1 as being confidence in 2D
        pose2d = target3d.clone()
        pose2d[..., 2] = 1.0

        # Get 3D pose from model
        output = self.model(pose2d)

        # Get denormalized values for MPJPE
        output_denorm = denormalize_dections(
            output,
            batch["video"][self.view]["img_dims"],
            batch["video"][self.view]["numtimesrot90clockwise"],
        )

        # Append values
        self.val_step_denorm_outputs.append(output_denorm)
        self.val_step_gt.append(j3d_image)

        # 3D Loses
        loss_3d_pos = losses.loss_mpjpe(output, target3d)
        loss_3d_scale = losses.n_mpjpe(output, target3d)
        loss_3d_velocity = losses.loss_velocity(output, target3d)
        loss_limb_variation = losses.loss_limb_var(output)
        loss_limb_gt = losses.loss_limb_gt(output, target3d)
        loss_angle = losses.loss_angle(output, target3d)
        loss_angle_velocity = losses.loss_angle_velocity(output, target3d)

        loss_total = (
            loss_3d_pos
            + self.config.lambda_scale * loss_3d_scale
            + self.config.lambda_3d_velocity * loss_3d_velocity
        )
        (
            +self.config.lambda_lv * loss_limb_variation
            + self.config.lambda_lg * loss_limb_gt
            + self.config.lambda_a * loss_angle
        )
        +self.config.lambda_av * loss_angle_velocity

        # Log 3D losses
        self.log("val/loss_3d_pos", loss_3d_pos, on_epoch=True)
        self.log("val/loss_3d_scale", loss_3d_scale, on_epoch=True)
        self.log("val/loss_3d_velocity", loss_3d_velocity, on_epoch=True)
        self.log("val/loss_limb_variation", loss_limb_variation, on_epoch=True)
        self.log("val/loss_limb_gt", loss_limb_gt, on_epoch=True)
        self.log("val/loss_angle", loss_angle, on_epoch=True)
        self.log("val/loss_angle_velocity", loss_angle_velocity, on_epoch=True)
        self.log("val/loss_total", loss_total, on_epoch=True)

        if self.trainer.is_last_batch:
            # Log sample of one of the predicted 3D poses from the trainset
            with torch.no_grad():
                # Log pose sequence as gif
                wandb_gif = get_wandb_video_from_joints(
                    output[0, :, :, :].cpu().numpy()
                )
                self.logger.experiment.log(
                    {
                        "val/pose3d_pred_seq": wandb_gif,
                    },
                )

        return j3d_image, output_denorm

    def on_validation_epoch_end(self) -> None:
        val_preds = torch.cat(self.val_step_denorm_outputs)
        gt = torch.cat(self.val_step_gt)

        # Align by root joint
        val_preds = val_preds - val_preds[:, :, 0:1, :]
        gt = gt - gt[:, :, 0:1, :]

        val_preds = val_preds.cpu().numpy()
        gt = gt.cpu().numpy()

        # Calculate MPJPE and Procrustes aligned MPJPE
        mpjpe = losses.mpjpe(val_preds, gt)
        pampjpe = losses.p_mpjpe(val_preds, gt)

        # Log MPJPE and PAMPJPE
        self.log("val/mpjpe", mpjpe)
        self.log("val/pampjpe", pampjpe)

        # Clear variables
        self.val_step_denorm_outputs.clear()
        self.val_step_gt.clear()

    def test_step(self, batch, batch_idx):
        # Get camera and 3d pose in world coordinate
        target3d, j3d_cam, j3d_image = sportspose2h36m(
            batch["joints_3d"]["data_points"],
            batch["video"]["calibration"][self.view],
            batch["video"][self.view]["numtimesrot90clockwise"],
            batch["video"][self.view]["img_dims"],
            return_camera_and_image_coordinates=True,
        )

        # Ortho project 3D pose to 2D pose for 2D gt - set z to 1 as being confidence in 2D
        pose2d = target3d.clone()
        pose2d[..., 2] = 1.0

        # Get 3D pose from model
        output = self.model(pose2d)

        # Get denormalized values for MPJPE
        output_denorm = denormalize_dections(
            output,
            batch["video"][self.view]["img_dims"],
            batch["video"][self.view]["numtimesrot90clockwise"],
        )

        # Append values
        self.test_step_denorm_outputs.append(output_denorm)
        self.test_step_gt.append(j3d_image)

        # 3D Loses
        loss_3d_pos = losses.loss_mpjpe(output, target3d)
        loss_3d_scale = losses.n_mpjpe(output, target3d)
        loss_3d_velocity = losses.loss_velocity(output, target3d)
        loss_limb_variation = losses.loss_limb_var(output)
        loss_limb_gt = losses.loss_limb_gt(output, target3d)
        loss_angle = losses.loss_angle(output, target3d)
        loss_angle_velocity = losses.loss_angle_velocity(output, target3d)

        loss_total = (
            loss_3d_pos
            + self.config.lambda_scale * loss_3d_scale
            + self.config.lambda_3d_velocity * loss_3d_velocity
        )
        (
            +self.config.lambda_lv * loss_limb_variation
            + self.config.lambda_lg * loss_limb_gt
            + self.config.lambda_a * loss_angle
        )
        +self.config.lambda_av * loss_angle_velocity

        # Log 3D losses
        self.log("test/loss_3d_pos", loss_3d_pos, on_epoch=True)
        self.log("test/loss_3d_scale", loss_3d_scale, on_epoch=True)
        self.log("test/loss_3d_velocity", loss_3d_velocity, on_epoch=True)
        self.log("test/loss_limb_variation", loss_limb_variation, on_epoch=True)
        self.log("test/loss_limb_gt", loss_limb_gt, on_epoch=True)
        self.log("test/loss_angle", loss_angle, on_epoch=True)
        self.log("test/loss_angle_velocity", loss_angle_velocity, on_epoch=True)
        self.log("test/loss_total", loss_total, on_epoch=True)

        if self.trainer.is_last_batch:
            # Log sample of one of the predicted 3D poses from the trainset
            with torch.no_grad():
                # Log pose sequence as gif
                wandb_gif = get_wandb_video_from_joints(
                    output[0, :, :, :].cpu().numpy()
                )
                self.logger.experiment.log(
                    {
                        "test/pose3d_pred_seq": wandb_gif,
                    },
                )

        return loss_total

    def on_test_epoch_end(self) -> None:
        test_preds = torch.cat(self.test_step_denorm_outputs)
        gt = torch.cat(self.test_step_gt)

        # Align by root joint
        test_preds = test_preds - test_preds[:, :, 0:1, :]
        gt = gt - gt[:, :, 0:1, :]

        test_preds = test_preds.cpu().numpy()
        gt = gt.cpu().numpy()

        # Calculate MPJPE and Procrustes aligned MPJPE
        mpjpe = losses.mpjpe(test_preds, gt)
        pampjpe = losses.p_mpjpe(test_preds, gt)

        # Log MPJPE and PAMPJPE
        self.log("test/mpjpe", mpjpe)
        self.log("test/pampjpe", pampjpe)

        # Clear variables
        self.test_step_denorm_outputs.clear()
        self.test_step_gt.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def main():
    torch.set_float32_matmul_precision("medium")

    # Init train dataset
    root_datapath = "/work3/ckin/bigdata/SportsPose/"
    data_path = os.path.join(root_datapath, "MarkerlessEndBachelor_withVideoPaths")
    video_path = os.path.join(root_datapath, "videos")
    include_debug_images = False

    # Init log and checkpoint data dir
    checkpoint_dir = "/work3/ckin/motionbert_data"

    print("Loading dataset...")
    test_subjects = ["mje", "mzm", "shs"]
    val_subjects = ["orb", "mhp", "mhs", "cin", "ufh"]

    train_dataset = AbstractImageDataset(
        data_dir=data_path,
        dataset_type="sportsPose",
        video_root_dir=video_path,
        views=["FO"],
        sample_level="video",
        return_preset={
            "joints_2d": True,
            "joints_3d": {
                "data_points": True,
            },
            "metadata": {
                "file_name": True,
                "person_id": True,
            },
            "video": {
                "view": {
                    "camera": False,
                    "img_dims": True,
                    "numtimesrot90clockwise": True,
                },
                "image": include_debug_images,
                "calibration": True,
            },
        },
        blacklist={"metadata": {"person_id": test_subjects + val_subjects}},
        seq_size=243,
    )

    # Val
    val_dataset = AbstractImageDataset(
        data_dir=data_path,
        dataset_type="sportsPose",
        video_root_dir=video_path,
        views=["FO"],
        sample_level="video",
        return_preset={
            "joints_2d": True,
            "joints_3d": {
                "data_points": True,
            },
            "metadata": {
                "file_name": True,
                "person_id": True,
            },
            "video": {
                "view": {
                    "camera": False,
                    "img_dims": True,
                    "numtimesrot90clockwise": True,
                },
                "image": include_debug_images,
                "calibration": True,
            },
        },
        whitelist={"metadata": {"person_id": val_subjects}},
        seq_size=243,
        validation_dataset=True,
    )

    # Test
    test_dataset = AbstractImageDataset(
        data_dir=data_path,
        dataset_type="sportsPose",
        video_root_dir=video_path,
        views=["FO"],
        sample_level="video",
        return_preset={
            "joints_2d": True,
            "joints_3d": {
                "data_points": True,
            },
            "metadata": {
                "file_name": True,
                "person_id": True,
            },
            "video": {
                "view": {
                    "camera": False,
                    "img_dims": True,
                    "numtimesrot90clockwise": True,
                },
                "image": include_debug_images,
                "calibration": True,
            },
        },
        whitelist={"metadata": {"person_id": test_subjects}},
        seq_size=243,
        validation_dataset=True,
    )

    print("Dataset loaded.")
    print("Train dataset length: ", len(train_dataset))
    print("Validation dataset length: ", len(val_dataset))
    print("Test dataset length: ", len(test_dataset))

    train_loader = utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )
    test_loader = utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )

    # Init model
    motionbert = MotionBertSportPose(
        learning_rate=0.0002,
        debug_images=include_debug_images,
        pretrain_path="/zhome/0c/6/109332/Projects/MotionBERT/models/model.bin",
    )

    # Init wandb logging
    wandb_logger = pl.loggers.WandbLogger(
        project="motionbert_sportspose", save_dir=checkpoint_dir
    )

    # Init trainer
    print("Start training...")
    trainer = pl.Trainer(
        max_epochs=30, logger=wandb_logger, default_root_dir=checkpoint_dir
    )

    # Test baseline model
    trainer.test(
        dataloaders=test_loader,
        verbose=True,
        model=motionbert,
    )

    # Train model
    trainer.fit(
        model=motionbert,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Test results
    trainer.test(dataloaders=test_loader)


if __name__ == "__main__":
    print("Hello from main")
    main()
