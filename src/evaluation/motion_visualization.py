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

def visualize_motion(
    motion_data: torch.Tensor,
    length: int,
    output_path: str,
    text_description: str,
    device,
    fps: int = 20,
):
    """
    Create a video of a single motion.

    Args:
        motion_data (torch.Tensor): The motion data tensor of shape [T, F] or [1, T, F].
        length (int): Number of frames in the motion sequence.
        output_path (str): Path to save the output video.
        text_description (str): Description/title for the video.
        fps (int, optional): Frames per second for the video. Defaults to 20.
        device (str, optional): Device to use for tensor operations. Defaults to "auto".
    """
    n_frames = length
    mean = torch.load(os.path.join("datasets/humanml3d/stats/motion_stats_abs", 'mean.pt')).float()
    std = torch.load(os.path.join("datasets/humanml3d/stats/motion_stats_abs", 'std.pt')).float()

    # Move motion_data to the specified device and ensure float type
    normalized = (motion_data + 1) / 2
    motion_data = normalized * std.to(motion_data.device) + mean.to(motion_data.device)
    motion_tensor = motion_data.to(device).float()
    
    # The motion features are expected to be in smplrifke format.
    # extract_joints will convert them to 3D joint positions.
    # smpl_layer is not needed when value_from is 'joints' (default).
    joints = extract_joints(motion_tensor, featsname="smplrifke", fps=fps, abs_root = True)["joints"]

    title = text_description
    
    # Set up the figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(title, fontsize=16)
    
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
        
        # Color scheme: red for keyframes, blue for regular frames
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
        
        fig.suptitle(f"{title}", fontsize=16)
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000/fps, repeat=False)
    
    # Save animation
    print(f"Saving video to {output_path}")
    FFwriter = writers['ffmpeg']
    writer = FFwriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(output_path, writer=writer)
    plt.close(fig)

