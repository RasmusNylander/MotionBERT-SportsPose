import torch
from torch import nn, optim, utils
from typing import Any
import lightning.pytorch as pl
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
from trackman.posetools.models.DSTformer import DSTformer
from lightning.pytorch.cli import LightningCLI
import itertools


def get_wandb_video_from_joints(joints, fps=30):
    frames = np.asarray([visualize_3d_pose(pose) for pose in joints])
    # Flip axis to (time, channel, height, width)
    frames = frames.transpose((0, 3, 1, 2))

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
    pose3d: torch.Tensor, fx: float, fy: float, cx: float, cy: float, rootIdx: int = 0
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
    pose3d: torch.Tensor,
    box: torch.Tensor,
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
    pose3d: torch.Tensor,
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
    lambda_consistency: float = 1.0


class MotionBertSportPose(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        views,
        test_views,
        batch_size,
        debug_images=False,
        config=None,
        ortho_project=False,
        pretrain_path=None,
        lambda_scale=1.0,
        lambda_3d_velocity=1.0,
        lambda_lv=1.0,
        lambda_lg=1.0,
        lambda_a=1.0,
        lambda_av=1.0,
        lambda_consistency=1.0,
        lambda_3d_pos=1.0,
        lambda_2d_pos=1.0,
        use_3d_data=True,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.views = views
        self.test_views = test_views
        self.batch_size = batch_size
        self.lambda_scale = lambda_scale
        self.lambda_3d_velocity = lambda_3d_velocity
        self.lambda_lv = lambda_lv
        self.lambda_lg = lambda_lg
        self.lambda_a = lambda_a
        self.lambda_av = lambda_av
        self.lambda_consistency = lambda_consistency
        self.lambda_3d_pos = lambda_3d_pos
        self.lambda_2d_pos = lambda_2d_pos
        self.use_3d_data = use_3d_data

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

        if pretrain_path is not None:
            self.model.load_state_dict(
                reorder_dataparallel_checkpoint(torch.load(pretrain_path)["model_pos"]),
                strict=True,
            )

        self.debug_images = debug_images

        # Init vars for validation and test metric calculation
        self.val_step_denorm_outputs = []
        self.val_step_gt = []
        self.test_step_denorm_outputs = []
        self.test_step_gt = []
        self.full_meta_data_gt = {}
        self.full_meta_data_denorm = {}
        self.partial_meta_data_gt = {}
        self.partial_meta_data_denorm = {}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Get camera and 3d pose in world coordinate
        target3ds = []
        j3d_cams = []
        j3d_images = []
        pose2ds = []
        outputs = []
        for view in self.views:
            target3d, j3d_cam, j3d_image = sportspose2h36m(
                batch["joints_3d"]["data_points"],
                batch["video"]["calibration"][view],
                batch["video"][view]["numtimesrot90clockwise"],
                batch["video"][view]["img_dims"],
                return_camera_and_image_coordinates=True,
            )
            target3ds.append(target3d)
            j3d_cams.append(j3d_cam)
            j3d_images.append(j3d_image)

            # Ortho project 3D pose to 2D pose for 2D gt - set z to 1 as being confidence in 2D
            pose2d = target3d.clone()
            pose2d[..., 2] = 1.0

            pose2ds.append(pose2d)

            # Augment / Mask 2D poses
            # TODO: Add augmentation and masking

            # Get 3D pose from model
            output = self.model(pose2d)
            outputs.append(output)

        # Consistency loss
        loss_consistency = torch.tensor(0.0).to(outputs[0])
        if len(outputs) > 1:
            # Loop through all pairs of outputs using itertools
            for output1, output2 in itertools.combinations(outputs, 2):
                loss_consistency += losses.loss_consistency(output1, output2)

            # Take average of all consistency losses between all pairs of outputs
            loss_consistency /= len(list(itertools.combinations(outputs, 2)))

        # Recombine values for easier loss calculation
        output = torch.cat(outputs, dim=0)
        target3d = torch.cat(target3ds, dim=0)
        j3d_cam = torch.cat(j3d_cams, dim=0)
        j3d_image = torch.cat(j3d_images, dim=0)
        pose2d = torch.cat(pose2ds, dim=0)

        # 3D Loses
        if self.use_3d_data:
            loss_3d_pos = losses.loss_mpjpe(output, target3d)
            loss_3d_scale = losses.n_mpjpe(output, target3d)
            loss_3d_velocity = losses.loss_velocity(output, target3d)
            loss_limb_variation = losses.loss_limb_var(output)
            loss_limb_gt = losses.loss_limb_gt(output, target3d)
            loss_angle = losses.loss_angle(output, target3d)
            loss_angle_velocity = losses.loss_angle_velocity(output, target3d)

            loss_total = (
                loss_3d_pos * self.lambda_3d_pos
                + self.lambda_scale * loss_3d_scale
                + self.lambda_3d_velocity * loss_3d_velocity
                + self.lambda_lv * loss_limb_variation
                + self.lambda_lg * loss_limb_gt
                + self.lambda_a * loss_angle
                + self.lambda_av * loss_angle_velocity
                + self.lambda_consistency * loss_consistency
            )
        else:
            loss_2d_pos = losses.loss_2d_weighted(output, target3d, pose2d[..., 2])
            loss_total = (
                loss_2d_pos * self.lambda_2d_pos
                + self.lambda_consistency * loss_consistency
            )

        if False:
            print("Losses:")
            print(loss_3d_pos)
            print(self.config.lambda_scale * loss_3d_scale)
            print(self.lambda_lv * loss_3d_velocity)
            print(self.config.lambda_consistency * loss_consistency)

            print("Total loss:")
            print(
                f"Sum of losses: {loss_3d_pos + self.config.lambda_scale * loss_3d_scale + self.config.lambda_3d_velocity * loss_3d_velocity + self.config.lambda_consistency * loss_consistency}"
            )

            print(f"Logged loss: {loss_total}")

            print(
                f"Whole computation: {loss_3d_pos + self.config.lambda_scale * loss_3d_scale + self.config.lambda_3d_velocity * loss_3d_velocity + self.config.lambda_lv * loss_limb_variation + self.config.lambda_lg * loss_limb_gt + self.config.lambda_a * loss_angle + self.config.lambda_av * loss_angle_velocity + self.config.lambda_consistency * loss_consistency}"
            )

            print(
                f"Whole computation: {loss_3d_pos} + {self.config.lambda_scale * loss_3d_scale} + {self.config.lambda_3d_velocity} * {loss_3d_velocity} + {self.config.lambda_lv} * {loss_limb_variation} + {self.config.lambda_lg} * {loss_limb_gt} + {self.config.lambda_a} * {loss_angle} + {self.config.lambda_av} * {loss_angle_velocity} + {self.config.lambda_consistency} * {loss_consistency}"
            )

            print(loss_total)

        # Log 3D losses
        if self.use_3d_data:
            self.log("train/loss_3d_pos", loss_3d_pos, on_epoch=True)
            self.log("train/loss_3d_scale", loss_3d_scale, on_epoch=True)
            self.log("train/loss_3d_velocity", loss_3d_velocity, on_epoch=True)
            self.log("train/loss_limb_variation", loss_limb_variation, on_epoch=True)
            self.log("train/loss_limb_gt", loss_limb_gt, on_epoch=True)
            self.log("train/loss_angle", loss_angle, on_epoch=True)
            self.log("train/loss_angle_velocity", loss_angle_velocity, on_epoch=True)
        else:
            self.log("train/loss_2d_pos", loss_2d_pos, on_epoch=True)

        self.log("train/loss_total", loss_total, on_epoch=True)
        self.log("train/loss_consistency", loss_consistency, on_epoch=True)

        # Debug log consistency times weights
        self.log(
            "train/loss_consistency_times_weights",
            loss_consistency * self.lambda_consistency,
            on_epoch=True,
        )

        if False:
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
        target3ds = []
        j3d_cams = []
        j3d_images = []
        pose2ds = []
        outputs = []
        output_denorms = []
        for view in self.test_views:
            target3d, j3d_cam, j3d_image = sportspose2h36m(
                batch["joints_3d"]["data_points"],
                batch["video"]["calibration"][view],
                batch["video"][view]["numtimesrot90clockwise"],
                batch["video"][view]["img_dims"],
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
                batch["video"][view]["img_dims"],
                batch["video"][view]["numtimesrot90clockwise"],
            )

            # Append values
            self.val_step_denorm_outputs.append(output_denorm)
            self.val_step_gt.append(j3d_image)

            target3ds.append(target3d)
            j3d_cams.append(j3d_cam)
            j3d_images.append(j3d_image)
            pose2ds.append(pose2d)
            outputs.append(output)
            output_denorms.append(output_denorm)

        # Consistency loss
        if len(outputs) > 1:
            loss_consistency = losses.loss_consistency(outputs[1], outputs[0])
        else:
            loss_consistency = torch.tensor(0.0).to(outputs[0])

        # Recombine values for easier loss calculation
        output = torch.cat(outputs, dim=0)
        output_denorm = torch.cat(output_denorms, dim=0)
        target3d = torch.cat(target3ds, dim=0)
        j3d_cam = torch.cat(j3d_cams, dim=0)
        j3d_image = torch.cat(j3d_images, dim=0)
        pose2d = torch.cat(pose2ds, dim=0)

        # 3D Loses
        loss_3d_pos = losses.loss_mpjpe(output, target3d)
        loss_3d_scale = losses.n_mpjpe(output, target3d)
        loss_3d_velocity = losses.loss_velocity(output, target3d)
        loss_limb_variation = losses.loss_limb_var(output)
        loss_limb_gt = losses.loss_limb_gt(output, target3d)
        loss_angle = losses.loss_angle(output, target3d)
        loss_angle_velocity = losses.loss_angle_velocity(output, target3d)

        loss_total = (
            loss_3d_pos * self.lambda_3d_pos
            + self.lambda_scale * loss_3d_scale
            + self.lambda_3d_velocity * loss_3d_velocity
            + self.lambda_lv * loss_limb_variation
            + self.lambda_lg * loss_limb_gt
            + self.lambda_a * loss_angle
            + self.lambda_av * loss_angle_velocity
            + self.lambda_consistency * loss_consistency
        )

        # Log 3D losses
        self.log("val/loss_3d_pos", loss_3d_pos, on_epoch=True)
        self.log("val/loss_3d_scale", loss_3d_scale, on_epoch=True)
        self.log("val/loss_3d_velocity", loss_3d_velocity, on_epoch=True)
        self.log("val/loss_limb_variation", loss_limb_variation, on_epoch=True)
        self.log("val/loss_limb_gt", loss_limb_gt, on_epoch=True)
        self.log("val/loss_angle", loss_angle, on_epoch=True)
        self.log("val/loss_angle_velocity", loss_angle_velocity, on_epoch=True)
        self.log("val/loss_total", loss_total, on_epoch=True)
        self.log("val/loss_consistency", loss_consistency, on_epoch=True)

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
        target3ds = []
        j3d_cams = []
        j3d_images = []
        pose2ds = []
        outputs = []
        output_denorms = []
        for view in self.test_views:
            target3d, j3d_cam, j3d_image = sportspose2h36m(
                batch["joints_3d"]["data_points"],
                batch["video"]["calibration"][view],
                batch["video"][view]["numtimesrot90clockwise"],
                batch["video"][view]["img_dims"],
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
                batch["video"][view]["img_dims"],
                batch["video"][view]["numtimesrot90clockwise"],
            )

            # Append values
            self.test_step_denorm_outputs.append(output_denorm)
            self.test_step_gt.append(j3d_image)

            """
            for person_id in batch["metadata"]["person_id"]:
                if person_id not in self.full_meta_data_gt.keys():
                    self.full_meta_data_gt[batch["metadata"]["person_id"]] = {}
                    self.full_meta_data_denorm[batch["metadata"]["person_id"]] = {}

                for activity in batch["metadata"]["activity"]:
                    if activity not in self.full_meta_data_gt[person_id].keys():
                        self.full_meta_data_gt[person_id][activity] = []
                        self.full_meta_data_denorm[person_id][activity] = []
            """
            for batch_num, activity in enumerate(batch["metadata"]["activity"]):
                if activity not in self.partial_meta_data_gt.keys():
                    self.partial_meta_data_gt[activity] = []
                    self.partial_meta_data_denorm[activity] = []

                self.partial_meta_data_gt[activity].append(
                    j3d_image[batch_num, ...].unsqueeze(0)
                )
                self.partial_meta_data_denorm[activity].append(
                    output_denorm[batch_num, ...].unsqueeze(0)
                )

            target3ds.append(target3d)
            j3d_cams.append(j3d_cam)
            j3d_images.append(j3d_image)
            pose2ds.append(pose2d)
            outputs.append(output)
            output_denorms.append(output_denorm)

        # Consistency loss
        if len(outputs) > 1:
            loss_consistency = losses.loss_consistency(outputs[1], outputs[0])
        else:
            loss_consistency = torch.tensor(0.0).to(outputs[0])

        # Recombine values for easier loss calculation
        output = torch.cat(outputs, dim=0)
        output_denorm = torch.cat(output_denorms, dim=0)
        target3d = torch.cat(target3ds, dim=0)
        j3d_cam = torch.cat(j3d_cams, dim=0)
        j3d_image = torch.cat(j3d_images, dim=0)
        pose2d = torch.cat(pose2ds, dim=0)

        # 3D Loses
        loss_3d_pos = losses.loss_mpjpe(output, target3d)
        loss_3d_scale = losses.n_mpjpe(output, target3d)
        loss_3d_velocity = losses.loss_velocity(output, target3d)
        loss_limb_variation = losses.loss_limb_var(output)
        loss_limb_gt = losses.loss_limb_gt(output, target3d)
        loss_angle = losses.loss_angle(output, target3d)
        loss_angle_velocity = losses.loss_angle_velocity(output, target3d)

        loss_total = (
            loss_3d_pos * self.lambda_3d_pos
            + self.lambda_scale * loss_3d_scale
            + self.lambda_3d_velocity * loss_3d_velocity
            + self.lambda_lv * loss_limb_variation
            + self.lambda_lg * loss_limb_gt
            + self.lambda_a * loss_angle
            + self.lambda_av * loss_angle_velocity
            + self.lambda_consistency * loss_consistency
        )

        # Log 3D losses
        self.log("test/loss_3d_pos", loss_3d_pos, on_epoch=True)
        self.log("test/loss_3d_scale", loss_3d_scale, on_epoch=True)
        self.log("test/loss_3d_velocity", loss_3d_velocity, on_epoch=True)
        self.log("test/loss_limb_variation", loss_limb_variation, on_epoch=True)
        self.log("test/loss_limb_gt", loss_limb_gt, on_epoch=True)
        self.log("test/loss_angle", loss_angle, on_epoch=True)
        self.log("test/loss_angle_velocity", loss_angle_velocity, on_epoch=True)
        self.log("test/loss_total", loss_total, on_epoch=True)
        self.log("test/loss_consistency", loss_consistency, on_epoch=True)

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

        # Do the same computations for the partial metadata, i.e compute metrics for each activity
        partial_mpjpe = {}
        partial_pampjpe = {}
        for activity in self.partial_meta_data_gt.keys():
            # Align by root joint
            partial_test_preds = torch.cat(self.partial_meta_data_denorm[activity])
            partial_gt = torch.cat(self.partial_meta_data_gt[activity])

            partial_test_preds = partial_test_preds - partial_test_preds[:, :, 0:1, :]
            partial_gt = partial_gt - partial_gt[:, :, 0:1, :]

            partial_test_preds = partial_test_preds.cpu().numpy()
            partial_gt = partial_gt.cpu().numpy()

            # Calculate MPJPE and Procrustes aligned MPJPE
            partial_mpjpe[activity] = losses.mpjpe(partial_test_preds, partial_gt)
            partial_pampjpe[activity] = losses.p_mpjpe(partial_test_preds, partial_gt)

            # Log MPJPE and PAMPJPE
            self.log(f"test/{activity}/mpjpe", partial_mpjpe[activity])
            self.log(f"test/{activity}/pampjpe", partial_pampjpe[activity])

        # Clear variables
        self.test_step_denorm_outputs.clear()
        self.test_step_gt.clear()
        self.full_meta_data_gt.clear()
        self.full_meta_data_denorm.clear()
        self.partial_meta_data_gt.clear()
        self.partial_meta_data_denorm.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class SportsPoseDataModule(pl.LightningDataModule):
    def __init__(
        self, data_path, video_path, views, test_views, batch_size, include_debug_images
    ):
        super().__init__()
        self.data_path = data_path
        self.video_path = video_path
        self.views = views
        self.test_views = test_views
        self.batch_size = batch_size
        self.include_debug_images = include_debug_images

        self.test_batch_size = (
            int(len(self.views) / len(self.test_views)) * self.batch_size
        )

        # Define data split
        self.test_subjects = ["mje", "mzm", "shs"]
        self.val_subjects = ["orb", "mhp", "mhs", "cin", "ufh"]

        # Return preset
        self.return_preset = {
            "joints_2d": True,
            "joints_3d": {
                "data_points": True,
            },
            "metadata": {
                "file_name": True,
                "person_id": True,
                "activity": True,
            },
            "video": {
                "view": {
                    "camera": False,
                    "img_dims": True,
                    "numtimesrot90clockwise": True,
                },
                "image": self.include_debug_images,
                "calibration": True,
            },
        }

    def setup(self, stage=None):
        self.train_dataset = AbstractImageDataset(
            data_dir=self.data_path,
            dataset_type="sportsPose",
            video_root_dir=self.video_path,
            views=self.views,
            sample_level="video",
            return_preset=self.return_preset,
            blacklist={
                "metadata": {"person_id": self.test_subjects + self.val_subjects}
            },
            seq_size=243,
        )
        self.val_dataset = AbstractImageDataset(
            data_dir=self.data_path,
            dataset_type="sportsPose",
            video_root_dir=self.video_path,
            views=self.test_views,
            sample_level="video",
            return_preset=self.return_preset,
            whitelist={"metadata": {"person_id": self.val_subjects}},
            seq_size=243,
            validation_dataset=True,
        )
        self.test_dataset = AbstractImageDataset(
            data_dir=self.data_path,
            dataset_type="sportsPose",
            video_root_dir=self.video_path,
            views=self.test_views,
            sample_level="video",
            return_preset=self.return_preset,
            whitelist={"metadata": {"person_id": self.test_subjects}},
            seq_size=243,
            validation_dataset=True,
        )

        print("Train dataset length: ", len(self.train_dataset))
        print("Validation dataset length: ", len(self.val_dataset))
        print("Test dataset length: ", len(self.test_dataset))

    def train_dataloader(self):
        return utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.batch_size,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.test_batch_size,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.test_batch_size,
            persistent_workers=True,
            pin_memory=True,
        )


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--sweep", default=False)
        parser.link_arguments("model.views", "data.views")
        parser.link_arguments("model.test_views", "data.test_views")
        parser.link_arguments("model.batch_size", "data.batch_size")


def main():
    # Set precision for A100 efficiency
    torch.set_float32_matmul_precision("medium")

    # Init logging
    checkpoint_dir = "/work3/ckin/motionbert_data/logs"
    wandb_logger = pl.loggers.WandbLogger(
        project="motionbert_sportspose", save_dir=checkpoint_dir
    )

    # Init cli (Note that with the CLI we dont need a trainer module)
    cli = CustomLightningCLI(
        MotionBertSportPose,
        SportsPoseDataModule,
        seed_everything_default=42,
        run=False,
        trainer_defaults={"logger": wandb_logger, "default_root_dir": checkpoint_dir},
        save_config_callback=None,
    )

    wandb_logger.experiment.config.update(dict(cli.config.model))

    # Run test before training for baseline
    cli.trainer.test(cli.model, datamodule=cli.datamodule, verbose=True)

    # Run training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # Run test after training
    cli.trainer.test(datamodule=cli.datamodule, verbose=True)


if __name__ == "__main__":
    print("Hello from main")
    main()
