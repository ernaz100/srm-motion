import os
import tempfile
from src.tools.extract_joints import extract_joints
from hydra import compose, initialize
from omegaconf import OmegaConf
import numpy as np
import torch
from src.dataset import get_dataset
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import textwrap
import pyrender
from smplx.utils import Struct
from src.tools.smpl_layer import SMPLH
from .smpl_renderer import viz_smpl_seq, Video

def visualize_motion(
    motion_data: torch.Tensor,
    length: int,
    output_path: str,
    text_description: str,
    device,
    fps: int = 20,
    render_type: str = "smpl_mesh",  # "stickman" or "smpl_mesh"
    smpl_model_path: str = "body_models/smplh/SMPLH_NEUTRAL.npz",
):
    """
    Create a video of a single motion.

    Args:
        motion_data (torch.Tensor): The motion data tensor of shape [T, F] or [1, T, F].
        length (int): Number of frames in the motion sequence.
        output_path (str): Path to save the output video.
        text_description (str): Description/title for the video.
        device: Device to use for tensor operations.
        fps (int, optional): Frames per second for the video. Defaults to 20.
        render_type (str, optional): Type of rendering - "stickman" or "smpl_mesh". Defaults to "smpl_mesh".
        smpl_model_path (str, optional): Path to SMPL model for mesh rendering. Required if render_type is "smpl_mesh".
    """
    
    if render_type == "smpl_mesh":
        visualize_motion_smpl_mesh(
            motion_data, length, output_path, text_description, device, fps, smpl_model_path
        )
    else:
        visualize_motion_stickman(
            motion_data, length, output_path, text_description, device, fps
        )


def visualize_motion_stickman(
    motion_data: torch.Tensor,
    length: int,
    output_path: str,
    text_description: str,
    device,
    fps: int = 20,
):
    """
    Create a stickman video of a single motion.

    Args:
        motion_data (torch.Tensor): The motion data tensor of shape [T, F] or [1, T, F].
        length (int): Number of frames in the motion sequence.
        output_path (str): Path to save the output video.
        text_description (str): Description/title for the video.
        device: Device to use for tensor operations.
        fps (int, optional): Frames per second for the video. Defaults to 20.
    """
    n_frames = length
    mean = torch.load(os.path.join("datasets/humanml3d/stats/motion_stats_abs", 'mean.pt')).float()
    std = torch.load(os.path.join("datasets/humanml3d/stats/motion_stats_abs", 'std.pt')).float()

    # Move motion_data to the specified device and ensure float type
    motion_data = motion_data * std.to(motion_data.device) + mean.to(motion_data.device)
    motion_tensor = motion_data.to(device).float()
    
    # The motion features are expected to be in smplrifke format.
    # extract_joints will convert them to 3D joint positions.
    joints = extract_joints(motion_tensor, featsname="smplrifke", fps=fps, abs_root = True)["joints"]

    title = text_description
    
    # Wrap the title to prevent overflow
    wrapped_title = '\n'.join(textwrap.wrap(title, width=60))
    
    # Set up the figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(wrapped_title, fontsize=16)
    
    # Set axis limits based on the data
    x_min, x_max = joints[:, :, 0].min(), joints[:, :, 0].max()
    y_min, y_max = joints[:, :, 1].min(), joints[:, :, 1].max()
    z_min, z_max = joints[:, :, 2].min(), joints[:, :, 2].max()
    
    # Add some padding
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # SMPL joint connections for skeleton visualization
    connections = [
        [0, 1], [0, 2], [0, 3],
        [1, 4], [2, 5], [3, 6],
        [4, 7], [5, 8], [6, 9],
        [7, 10], [8, 11],
        [9, 12], [9, 13], [9, 14],
        [12, 15], [13, 16], [14, 17],
        [16, 18], [17, 19],
        [18, 20], [19, 21]
    ]
    
    def animate(frame):
        # Clear previous frame
        ax.clear()
        
        # Set titles and limits again
        ax.set_title(f"Frame: {frame}/{n_frames-1}", fontsize=14)
        
        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
        ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Get current frame data
        pred_frame = joints[frame]
        
        # Color scheme: green for the motion
        pred_color = 'green'
        
        # Plot joints
        ax.scatter(pred_frame[:, 0], pred_frame[:, 1], pred_frame[:, 2], 
                    c=pred_color, s=50, alpha=0.8)
        
        # Plot skeleton connections
        for connection in connections:
            if connection[0] < len(pred_frame) and connection[1] < len(pred_frame):
                # Predicted skeleton
                ax.plot3D([pred_frame[connection[0], 0], pred_frame[connection[1], 0]],
                            [pred_frame[connection[0], 1], pred_frame[connection[1], 1]],
                            [pred_frame[connection[0], 2], pred_frame[connection[1], 2]],
                            c=pred_color, alpha=0.6, linewidth=2)
        
        fig.suptitle(wrapped_title, fontsize=16)
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000/fps, repeat=False)
    
    # Save animation
    print(f"Saving stickman video to {output_path}")
    FFwriter = writers['ffmpeg']
    writer = FFwriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(output_path, writer=writer)
    print(f"✓ Stickman video saved successfully")
    plt.close(fig)


def visualize_motion_smpl_mesh(
    motion_data: torch.Tensor,
    length: int,
    output_path: str,
    text_description: str,
    device,
    fps: int = 20,
    smpl_model_path: str = None,
):
    """
    Create an SMPL mesh video of a single motion.

    Args:
        motion_data (torch.Tensor): The motion data tensor of shape [T, F] or [1, T, F].
        length (int): Number of frames in the motion sequence.
        output_path (str): Path to save the output video.
        text_description (str): Description/title for the video.
        device: Device to use for tensor operations.
        fps (int, optional): Frames per second for the video. Defaults to 20.
        smpl_model_path (str, optional): Path to SMPL model. If None, will try to find default path.
    """

    # Load motion statistics
    mean = torch.load(os.path.join("datasets/humanml3d/stats/motion_stats_abs", 'mean.pt')).float()
    std = torch.load(os.path.join("datasets/humanml3d/stats/motion_stats_abs", 'std.pt')).float()

    # Denormalize motion data
    motion_data = motion_data * std.to(motion_data.device) + mean.to(motion_data.device)
    motion_tensor = motion_data.to(device).float()
    
    # Initialize SMPL layer
    smpl_layer = SMPLH(
        path=smpl_model_path,
        jointstype="vertices",  # We want vertices for mesh rendering
        input_pose_rep="axisangle",
        batch_size=512,
        gender="neutral"
    ).to(device)

    # Extract SMPL vertices using the SMPL layer
    smpl_output = extract_joints(
        motion_tensor, 
        featsname="smplrifke", 
        fps=fps, 
        abs_root=True,
        value_from="smpl",
        smpl_layer=smpl_layer
    )
    vertices = smpl_output["vertices"]  # Shape: [T, 6890, 3]

    assert len(vertices.shape) == 3
    # Put vertices at floor level
    ground = vertices[..., 2].min()
    vertices[..., 2] -= ground

    # Create output directory for frames
    out_folder = os.path.splitext(output_path)[0]
    os.makedirs(out_folder, exist_ok=True)

    # Get SMPL faces
    faces = smpl_layer.faces

    # Create body structure for rendering
    # Ensure vertices is a tensor (it might already be one)
    if torch.is_tensor(vertices):
        verts = vertices
    else:
        verts = torch.from_numpy(vertices)
    body_pred = Struct(v=verts, f=faces)

    # Render SMPL sequence using pyrender

    print(f"Rendering SMPL mesh video to {output_path}")
    viz_smpl_seq(
        pyrender, 
        out_folder, 
        body_pred, 
        fps=fps,
        progress_bar=None,  # Use default progress bar
        render_body=True,
        render_joints=False,
        render_skeleton=False,
        render_ground=True,
        wireframe=False,
        RGBA=False,
        cam_offset=[0.0, 2.2, 0.9],
        ground_color0=[0.8, 0.9, 0.9],
        ground_color1=[0.6, 0.7, 0.7],
        body_alpha=0.9
    )
    
    # Convert frames to video
    video = Video(out_folder, fps=fps)
    video.save(output_path)
    
    print(f"SMPL mesh video saved to {output_path}")
        


def create_simple_mesh_video(vertices, faces, output_path, text_description, fps=20):
    """
    Create a simple mesh video using matplotlib when pyrender is not available.
    
    Args:
        vertices (np.ndarray): SMPL vertices of shape [T, 6890, 3]
        faces (np.ndarray): SMPL faces
        output_path (str): Path to save the output video
        text_description (str): Description for the video
        fps (int): Frames per second
    """
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("mpl_toolkits not available. Cannot create mesh visualization.")
        return

    n_frames = len(vertices)
    
    # Wrap the title to prevent overflow
    wrapped_title = '\n'.join(textwrap.wrap(text_description, width=60))
    
    # Set up the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(wrapped_title, fontsize=16)
    
    # Set axis limits based on the data
    x_min, x_max = vertices[:, :, 0].min(), vertices[:, :, 0].max()
    y_min, y_max = vertices[:, :, 1].min(), vertices[:, :, 1].max()
    z_min, z_max = vertices[:, :, 2].min(), vertices[:, :, 2].max()
    
    # Add some padding
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    def animate(frame):
        # Clear previous frame
        ax.clear()
        
        # Set titles and limits again
        ax.set_title(f"Frame: {frame}/{n_frames-1}", fontsize=14)
        
        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
        ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Get current frame vertices
        frame_vertices = vertices[frame]
        
        # Create mesh faces for current frame
        mesh_faces = [frame_vertices[face] for face in faces]
        
        # Create 3D polygon collection
        poly3d = Poly3DCollection(mesh_faces, alpha=0.7, facecolor='lightblue', edgecolor='black', linewidth=0.1)
        ax.add_collection3d(poly3d)
        
        fig.suptitle(wrapped_title, fontsize=16)
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000/fps, repeat=False)
    
    # Save animation
    print(f"Saving simple mesh video to {output_path}")
    FFwriter = writers['ffmpeg']
    writer = FFwriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(output_path, writer=writer)
    plt.close(fig)

