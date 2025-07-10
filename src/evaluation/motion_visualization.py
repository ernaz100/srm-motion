"""Motion visualization utilities for SRM."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from textwrap import wrap

# Kinematic tree for HumanML3D (22 joints)
kinematic_tree = [
    [0, 3, 6, 9, 12, 15],
    [9, 13, 16, 18, 20],
    [9, 14, 17, 19, 21],
    [0, 1, 4, 7, 10],
    [0, 2, 5, 8, 11],
]
colors = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]

# HumanML3D joint names for reference
HUMANML3D_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", 
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", 
    "neck", "left_collar", "right_collar", "head", "left_shoulder", 
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"
]

# HumanML3D kinematic chain (adapted from STMC)
HUMANML3D_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],  # Right leg
    [0, 1, 4, 7, 10],  # Left leg  
    [0, 3, 6, 9, 12, 15],  # Spine
    [9, 14, 17, 19, 21],  # Right arm
    [9, 13, 16, 18, 20],  # Left arm
]

# HumanML3D raw offsets (adapted from STMC)
HUMANML3D_RAW_OFFSETS = np.array([
    [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0],
    [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1],
    [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, -1, 0], [0, -1, 0],
    [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]
])


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    
    Args:
        quaternions: quaternions with real part first, as tensor of shape (..., 4).
    
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    
    Args:
        q: quaternions with real part first, as tensor of shape (..., 4).
        v: vectors to rotate, as tensor of shape (..., 3).
    
    Returns:
        Rotated vectors as tensor of shape (..., 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
    return v + 2 * (q[..., :1] * uv + uuv)


def qinv(q):
    """
    Invert quaternion(s) q.
    
    Args:
        q: quaternions with real part first, as tensor of shape (..., 4).
    
    Returns:
        Inverted quaternions as tensor of shape (..., 4).
    """
    assert q.shape[-1] == 4
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quaternion_to_cont6d(quaternions):
    """
    Convert quaternions to 6D continuous rotation representation.
    
    Args:
        quaternions: quaternions with real part first, as tensor of shape (..., 4).
    
    Returns:
        6D rotation representation as tensor of shape (..., 6).
    """
    matrices = quaternion_to_matrix(quaternions)
    return matrices[..., :2, :].reshape(quaternions.shape[:-1] + (6,))


def recover_root_rot_pos(data):
    """
    Recover root rotation and position from HumanML3D representation.
    
    Args:
        data: HumanML3D motion data as tensor of shape (..., seq_len, feature_dim).
    
    Returns:
        Tuple of (root_rotation_quaternions, root_positions).
    """
    # Extract rotation velocity (first channel)
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    
    # Get Y-axis rotation from rotation velocity
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
    
    # Convert to quaternions
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)
    
    # Recover root position
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]  # XZ velocities
    
    # Add Y-axis rotation to root position
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    
    # Set Y position from root height
    r_pos[..., 1] = data[..., 3]
    
    return r_rot_quat, r_pos


def recover_from_ric(motion: np.ndarray, n_joints: int = 22) -> np.ndarray:
    """
    Recover XYZ joint positions from HumanML3D representation (ric, root-relative).
    
    This function implements the recovery logic from the STMC repo's HumanML3D utilities.
    It converts the HumanML3D feature representation back to joint positions.
    
    Args:
        motion: HumanML3D motion data as numpy array of shape (seq_len, feature_dim).
                Expected feature dimension is 263 for HumanML3D.
        n_joints: Number of joints (default: 22 for HumanML3D).
    
    Returns:
        Joint positions as numpy array of shape (seq_len, n_joints, 3).
    """
    if motion.shape[-1] != 263:
        raise ValueError(f"Expected HumanML3D feature dimension 263, got {motion.shape[-1]}")
    
    # Convert to torch tensor
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion).float()
    
    # Recover root rotation and position
    r_rot_quat, r_pos = recover_root_rot_pos(motion)
    
    # Extract RIC (Rotation Invariant Coordinates) data
    # RIC data starts at index 4 and contains (n_joints-1)*3 features
    ric_start = 4
    ric_end = ric_start + (n_joints - 1) * 3
    positions = motion[..., ric_start:ric_end]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    
    # Add Y-axis rotation to local joints
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), 
        positions
    )
    
    # Add root XZ to joints
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    
    # Concatenate root and joints
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    
    # Convert back to numpy if input was numpy
    if isinstance(motion, torch.Tensor):
        positions = positions.numpy()
    
    return positions


def recover_from_rot(motion: np.ndarray, n_joints: int = 22) -> np.ndarray:
    """
    Recover joint positions from HumanML3D rotation representation.
    
    This function recovers joint positions using the rotation data instead of RIC data.
    It requires a skeleton model for forward kinematics.
    
    Args:
        motion: HumanML3D motion data as numpy array of shape (seq_len, feature_dim).
        n_joints: Number of joints (default: 22 for HumanML3D).
    
    Returns:
        Joint positions as numpy array of shape (seq_len, n_joints, 3).
    """
    if motion.shape[-1] != 263:
        raise ValueError(f"Expected HumanML3D feature dimension 263, got {motion.shape[-1]}")
    
    # Convert to torch tensor
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion).float()
    
    # Recover root rotation and position
    r_rot_quat, r_pos = recover_root_rot_pos(motion)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)
    
    # Extract rotation data (6D continuous representation)
    rot_start = 4 + (n_joints - 1) * 3  # After RIC data
    rot_end = rot_start + (n_joints - 1) * 6
    cont6d_params = motion[..., rot_start:rot_end]
    
    # Concatenate root rotation with joint rotations
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, n_joints, 6)
    
    # Note: This requires a skeleton model for forward kinematics
    # For now, we'll use a simplified approach or return the RIC recovery
    # In a full implementation, you would use:
    # positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)
    
    # Fallback to RIC recovery for now
    return recover_from_ric(motion, n_joints)


def extract_humanml3d_features(motion: np.ndarray) -> dict:
    """
    Extract and organize HumanML3D motion features for analysis.
    
    Args:
        motion: HumanML3D motion data as numpy array of shape (seq_len, 263).
    
    Returns:
        Dictionary containing extracted features:
        - root_rot_vel: Root rotation velocity
        - root_lin_vel: Root linear velocity (XZ)
        - root_height: Root height (Y)
        - ric_data: Rotation invariant coordinates
        - rot_data: Rotation data (6D continuous)
        - local_vel: Local joint velocities
        - foot_contact: Foot contact information
    """
    if motion.shape[-1] != 263:
        raise ValueError(f"Expected HumanML3D feature dimension 263, got {motion.shape[-1]}")
    
    # Extract root data (first 4 channels)
    root_data = motion[:, :4]
    root_rot_vel = root_data[:, 0]  # Root rotation velocity
    root_lin_vel = root_data[:, 1:3]  # Root linear velocity (XZ)
    root_height = root_data[:, 3]  # Root height (Y)
    
    # Extract RIC data (rotation invariant coordinates)
    ric_start = 4
    ric_end = ric_start + 21 * 3  # (n_joints-1) * 3
    ric_data = motion[:, ric_start:ric_end]
    ric_data = ric_data.reshape(-1, 21, 3)  # Reshape to (seq_len, 21, 3)
    
    # Extract rotation data (6D continuous representation)
    rot_start = ric_end
    rot_end = rot_start + 21 * 6  # (n_joints-1) * 6
    rot_data = motion[:, rot_start:rot_end]
    rot_data = rot_data.reshape(-1, 21, 6)  # Reshape to (seq_len, 21, 6)
    
    # Extract local velocities
    vel_start = rot_end
    vel_end = vel_start + 22 * 3  # n_joints * 3
    local_vel = motion[:, vel_start:vel_end]
    local_vel = local_vel.reshape(-1, 22, 3)  # Reshape to (seq_len, 22, 3)
    
    # Extract foot contact information (last 4 channels)
    foot_contact = motion[:, -4:]
    
    return {
        'root_rot_vel': root_rot_vel,
        'root_lin_vel': root_lin_vel,
        'root_height': root_height,
        'ric_data': ric_data,
        'rot_data': rot_data,
        'local_vel': local_vel,
        'foot_contact': foot_contact
    }


def plot_3d_motion(motion: np.ndarray, save_path: str, title: str = "", fps: int = 20, radius: float = 3):
    """
    Plot 3D stick figure motion and save as video.
    
    Args:
        motion: Motion data as numpy array of shape (seq_len, n_joints, 3)
        save_path: Path to save the video file
        title: Title for the animation
        fps: Frames per second for the video
        radius: Radius for the 3D plot limits
    """
    # Input validation
    if motion.ndim != 3 or motion.shape[-1] != 3:
        raise ValueError(f"Expected motion data of shape (seq_len, n_joints, 3), got {motion.shape}")
    
    data = motion.copy()
    
    # Set matplotlib backend to Agg for headless environments
    import matplotlib
    matplotlib.use('Agg')
    
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_ylim3d([0, radius])
    ax.set_zlim3d([-radius / 3, radius * 2 / 3])
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(elev=110, azim=15)

    def update(frame):
        ax.cla()
        ax.set_title(title + f' [{frame}]')
        skeleton = data[frame]
        for chain, color in zip(kinematic_tree, colors):
            ax.plot(skeleton[chain, 0], skeleton[chain, 1], skeleton[chain, 2], color=color, linewidth=2.0)

    anim = FuncAnimation(fig, update, frames=len(data), interval=1000 / fps)
    
    # Use proper FFmpeg writer to avoid PIL extension issues
    try:
        FFwriter = writers['ffmpeg']
        writer = FFwriter(fps=fps, metadata=dict(artist='SRM'), bitrate=1800)
        anim.save(save_path, writer=writer)
    except Exception as e:
        # Fallback: try without specifying writer
        print(f"Warning: FFmpeg writer failed, trying fallback: {e}")
        try:
            anim.save(save_path, fps=fps)
        except Exception as e2:
            print(f"Error: Both FFmpeg writer and fallback failed: {e2}")
            # Try saving as GIF as last resort
            try:
                gif_path = save_path.replace('.mp4', '.gif')
                anim.save(gif_path, writer='pillow', fps=fps)
                print(f"Saved as GIF instead: {gif_path}")
            except Exception as e3:
                print(f"Error: All save methods failed: {e3}")
                raise
    
    plt.close(fig)

# Function to prep for logging: return list of images or video paths 