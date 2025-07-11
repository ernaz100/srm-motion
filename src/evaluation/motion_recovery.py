"""
Comprehensive motion recovery utilities for SRM.

This module provides various motion recovery functions adapted from the STMC repo,
including HumanML3D, SMPL-based, and other motion representations.
"""

import numpy as np
import torch
import einops
from typing import Optional, Dict, Any, Tuple
import warnings

# Import local utilities
from .motion_visualization import (
    quaternion_to_matrix, qrot, qinv, quaternion_to_cont6d,
    recover_root_rot_pos, recover_from_ric
)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle rotations to rotation matrices.
    
    Args:
        axis_angle: Axis-angle rotations as tensor of shape (..., 3).
    
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # Rodrigues' rotation formula
    R = cos_angle * torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype) + \
        sin_angle * torch.cross(torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype), axis, dim=-1) + \
        (1 - cos_angle) * torch.einsum('...i,...j->...ij', axis, axis)
    
    return R


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to 6D continuous rotation representation.
    
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    
    Returns:
        6D rotation representation as tensor of shape (..., 6).
    """
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D continuous rotation representation to rotation matrices.
    
    Args:
        d6: 6D rotation representation as tensor of shape (..., 6).
    
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]
    
    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    
    matrix = torch.stack([x, y, z], dim=-1)
    return matrix


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str = "ZYX") -> torch.Tensor:
    """
    Convert rotation matrices to Euler angles.
    
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Euler angle convention (default: "ZYX").
    
    Returns:
        Euler angles as tensor of shape (..., 3).
    """
    if convention == "ZYX":
        # Extract Euler angles from rotation matrix
        sy = torch.sqrt(matrix[..., 0, 0] * matrix[..., 0, 0] + matrix[..., 1, 0] * matrix[..., 1, 0])
        singular = sy < 1e-6
        
        x = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
        y = torch.atan2(-matrix[..., 2, 0], sy)
        z = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
        
        # Handle singular case
        x = torch.where(singular, torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1]), x)
        y = torch.where(singular, torch.atan2(-matrix[..., 2, 0], sy), y)
        z = torch.where(singular, torch.zeros_like(z), z)
        
        return torch.stack([z, y, x], dim=-1)
    else:
        raise NotImplementedError(f"Euler angle convention {convention} not implemented")


def axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return rotation matrices for rotations about specified axis.
    
    Args:
        axis: Axis label "X", "Y", or "Z".
        angle: Euler angles in radians as tensor of any shape.
    
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    
    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"Invalid axis {axis}")
    
    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to axis-angle representation.
    
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    
    Returns:
        Axis-angle rotations as tensor of shape (..., 3).
    """
    # Extract rotation angle
    trace = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    
    # Extract rotation axis
    axis = torch.stack([
        matrix[..., 2, 1] - matrix[..., 1, 2],
        matrix[..., 0, 2] - matrix[..., 2, 0],
        matrix[..., 1, 0] - matrix[..., 0, 1]
    ], dim=-1)
    
    # Normalize axis
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    axis = axis / (axis_norm + 1e-8)
    
    # Handle zero rotation case
    zero_rotation = angle < 1e-6
    axis = torch.where(zero_rotation[..., None], torch.tensor([1., 0., 0.], device=matrix.device), axis)
    
    return axis * angle[..., None]


def smplrifkefeats_to_smpldata(
    features: torch.Tensor,
    first_angle: float = 0.0,
    abs_root: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert SMPL-RIFKE features back to SMPL data.
    
    This function is adapted from the STMC repo's smplrifke_feats.py.
    
    Args:
        features: SMPL-RIFKE features as tensor of shape (..., 205).
        first_angle: Initial angle for integration (default: 0.0).
        abs_root: Whether features contain absolute root values (default: False).
    
    Returns:
        Dictionary containing SMPL data:
        - poses: SMPL poses as tensor of shape (..., 66)
        - trans: Translations as tensor of shape (..., 3)
        - joints: Joint positions as tensor of shape (..., 22, 3)
    """
    if features.shape[-1] != 205:
        raise ValueError(f"Expected SMPL-RIFKE feature dimension 205, got {features.shape[-1]}")
    
    # Ungroup features
    (
        root_grav_axis,
        vel_trajectory_local,
        vel_angles,
        poses_local,
        joints_local,
    ) = ungroup_smplrifke(features)
    
    poses_mat_local = rotation_6d_to_matrix(poses_local)
    global_orient_local = poses_mat_local[..., 0, :, :]
    
    if abs_root:
        # Channels already contain absolute values
        yaw = vel_angles
        rotZ = axis_angle_rotation("Z", yaw)
        
        # Absolute trajectory in world coordinates
        trajectory = vel_trajectory_local
        
        # Rotate local joints back to world
        joints_xy = torch.einsum("...jk,...lk->...lj", rotZ[..., :2, :2], joints_local[..., [0, 1]])
        joints_z = joints_local[..., 2:3]
        joints = torch.cat([joints_xy, joints_z], dim=-1)
    else:
        # Original velocity-based representation - integrate
        # Remove the dummy last angle and integrate the angles
        angles = torch.cumsum(vel_angles[..., :-1], dim=-1)
        # The first angle is zero (canonicalization)
        angles = first_angle + torch.cat((torch.zeros_like(angles[..., [0]]), angles), dim=-1)
        
        # Construct the rotation matrix
        rotZ = axis_angle_rotation("Z", angles)
        
        # Rotate the velocity back to world frame
        vel_trajectory = torch.einsum(
            "...jk,...k->...j", rotZ[..., :2, :2], vel_trajectory_local
        )
        
        joints_xy = torch.einsum("...jk,...lk->...lj", rotZ[..., :2, :2], joints_local[..., [0, 1]])
        joints_z = joints_local[..., 2:3]
        joints = torch.cat([joints_xy, joints_z], dim=-1)
        
        # Integrate trajectory (velocities)
        trajectory = torch.cumsum(vel_trajectory[..., :-1, :], dim=-2)
        # First position is zero
        trajectory = torch.cat((torch.zeros_like(trajectory[..., [0], :]), trajectory), dim=-2)
    
    # Add the pelvis (which is still zero)
    joints = torch.cat((torch.zeros_like(joints[..., [0], :]), joints), dim=-2)
    
    # Add back global translation and height to every joint
    joints[..., :, 2] += root_grav_axis[..., None]
    joints[..., :, :2] += trajectory[..., None, :]
    
    # Get back the translation for SMPL data
    trans = torch.cat([trajectory, root_grav_axis[..., None]], dim=-1)
    
    # Remove the predicted Z rotation inside global_orient_local
    global_euler_local = matrix_to_euler_angles(global_orient_local, "ZYX")
    _, rotY_angle, rotX_angle = torch.unbind(global_euler_local, -1)
    rotY = axis_angle_rotation("Y", rotY_angle)
    rotX = axis_angle_rotation("X", rotX_angle)
    
    # Replace it with the one computed with velocities
    global_orient = rotZ @ rotY @ rotX
    poses_mat = torch.cat(
        (global_orient[..., None, :, :], poses_mat_local[..., 1:, :, :]), dim=-3
    )
    
    poses = matrix_to_axis_angle(poses_mat)
    # Flatten back
    poses = einops.rearrange(poses, "... l t -> ... (l t)")
    
    return {"poses": poses, "trans": trans, "joints": joints}


def ungroup_smplrifke(features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Ungroup SMPL-RIFKE features into their components.
    
    Args:
        features: SMPL-RIFKE features as tensor of shape (..., 205).
    
    Returns:
        Tuple of (root_grav_axis, vel_trajectory_local, vel_angles, poses_local, joints_local).
    """
    if features.shape[-1] != 205:
        raise ValueError(f"Expected SMPL-RIFKE feature dimension 205, got {features.shape[-1]}")
    
    # Extract without keeping singleton dimensions
    root_grav_axis = features[..., 0]  # (...)
    vel_trajectory_local = features[..., 1:3]  # (..., 2)
    vel_angles = features[..., 3]  # (...)
    poses_local_flatten = features[..., 4:136]  # (..., 132)
    joints_local_flatten = features[..., 136:205]  # (..., 69)
    
    poses_local = einops.rearrange(poses_local_flatten, "... (l t) -> ... l t", t=6)
    joints_local = einops.rearrange(joints_local_flatten, "... (l t) -> ... l t", t=3)
    
    return root_grav_axis, vel_trajectory_local, vel_angles, poses_local, joints_local


def recover_from_smplrifke(
    motion: np.ndarray, 
    fps: int = 30,
    first_angle: float = 0.0,
    abs_root: bool = False
) -> Dict[str, np.ndarray]:
    """
    Recover joint positions from SMPL-RIFKE representation.
    
    Args:
        motion: SMPL-RIFKE motion data as numpy array of shape (seq_len, 205).
        fps: Frame rate (default: 30).
        first_angle: Initial angle for integration (default: 0.0).
        abs_root: Whether features contain absolute root values (default: False).
    
    Returns:
        Dictionary containing recovered data:
        - joints: Joint positions as numpy array of shape (seq_len, 22, 3)
        - poses: SMPL poses as numpy array of shape (seq_len, 66)
        - trans: Translations as numpy array of shape (seq_len, 3)
    """
    if motion.shape[-1] != 205:
        raise ValueError(f"Expected SMPL-RIFKE feature dimension 205, got {motion.shape[-1]}")
    
    # Convert to torch tensor
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion).float()
    
    # Recover SMPL data
    smpldata = smplrifkefeats_to_smpldata(
        motion, first_angle=first_angle, abs_root=abs_root
    )
    
    # Convert back to numpy
    result = {
        "joints": smpldata["joints"].numpy(),
        "poses": smpldata["poses"].numpy(),
        "trans": smpldata["trans"].numpy(),
        "mocap_framerate": fps
    }
    
    return result


def recover_from_guoh3dfeats(motion: np.ndarray) -> np.ndarray:
    """
    Recover joint positions from Guo et al. HumanML3D features.
    
    This is a simplified version that uses the RIC recovery method.
    For full functionality, a skeleton model would be required.
    
    Args:
        motion: Guo et al. HumanML3D motion data as numpy array of shape (seq_len, 263).
    
    Returns:
        Joint positions as numpy array of shape (seq_len, 22, 3).
    """
    # For now, use the RIC recovery method
    # In a full implementation, this would use the skeleton-based recovery
    return recover_from_ric(motion, n_joints=22)


def canonicalize_rotation(motion: np.ndarray, motion_type: str = "humanml3d") -> np.ndarray:
    """
    Canonicalize motion by removing global rotation.
    
    Args:
        motion: Motion data as numpy array.
        motion_type: Type of motion representation ("humanml3d", "smplrifke", "joints").
    
    Returns:
        Canonicalized motion as numpy array.
    """
    if motion_type == "humanml3d":
        # For HumanML3D, the representation is already rotation-invariant
        return motion
    elif motion_type == "smplrifke":
        # For SMPL-RIFKE, we can canonicalize by setting first_angle=0
        return recover_from_smplrifke(motion, first_angle=0.0)
    elif motion_type == "joints":
        # For joint positions, we need to implement rotation canonicalization
        # This would involve finding the principal direction and rotating to align with X-axis
        warnings.warn("Joint position canonicalization not fully implemented")
        return motion
    else:
        raise ValueError(f"Unknown motion type: {motion_type}")


def convert_motion_representation(
    motion: np.ndarray, 
    from_type: str, 
    to_type: str,
    **kwargs
) -> np.ndarray:
    """
    Convert motion between different representations.
    
    Args:
        motion: Input motion data as numpy array.
        from_type: Source representation type.
        to_type: Target representation type.
        **kwargs: Additional arguments for conversion.
    
    Returns:
        Converted motion data as numpy array.
    """
    # First, convert to joint positions as intermediate representation
    if from_type == "humanml3d":
        joints = recover_from_ric(motion)
    elif from_type == "smplrifke":
        smpldata = recover_from_smplrifke(motion, **kwargs)
        joints = smpldata["joints"]
    elif from_type == "joints":
        joints = motion
    else:
        raise ValueError(f"Unknown source motion type: {from_type}")
    
    # Then convert from joint positions to target representation
    if to_type == "joints":
        return joints
    elif to_type == "humanml3d":
        # This would require implementing the forward conversion
        # For now, return a placeholder
        warnings.warn("Forward conversion to HumanML3D not implemented")
        return motion
    elif to_type == "smplrifke":
        # This would require implementing the forward conversion
        # For now, return a placeholder
        warnings.warn("Forward conversion to SMPL-RIFKE not implemented")
        return motion
    else:
        raise ValueError(f"Unknown target motion type: {to_type}")


def analyze_motion_features(motion: np.ndarray, motion_type: str = "humanml3d") -> Dict[str, Any]:
    """
    Analyze motion features and provide statistics.
    
    Args:
        motion: Motion data as numpy array.
        motion_type: Type of motion representation.
    
    Returns:
        Dictionary containing motion analysis.
    """
    if motion_type == "humanml3d":
        features = extract_humanml3d_features(motion)
        
        analysis = {
            "sequence_length": motion.shape[0],
            "feature_dimension": motion.shape[1],
            "root_rotation_range": (features["root_rot_vel"].min(), features["root_rot_vel"].max()),
            "root_velocity_magnitude": np.linalg.norm(features["root_lin_vel"], axis=1).mean(),
            "root_height_range": (features["root_height"].min(), features["root_height"].max()),
            "ric_data_range": (features["ric_data"].min(), features["ric_data"].max()),
            "foot_contact_summary": {
                "left_foot": features["foot_contact"][:, :2].mean(axis=0),
                "right_foot": features["foot_contact"][:, 2:].mean(axis=0)
            }
        }
        
        return analysis
    else:
        # For other motion types, provide basic statistics
        return {
            "sequence_length": motion.shape[0],
            "feature_dimension": motion.shape[1],
            "data_range": (motion.min(), motion.max()),
            "data_mean": motion.mean(),
            "data_std": motion.std()
        }


# Import the extract_humanml3d_features function from motion_visualization
from .motion_visualization import extract_humanml3d_features 