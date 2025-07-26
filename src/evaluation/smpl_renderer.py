"""
SMPL Mesh Renderer for motion visualization.

This module provides functionality to render SMPL mesh sequences using pyrender.
Based on the humor renderer from the STMC project.
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import torch
import trimesh
from tqdm import tqdm
from smplx.utils import Struct
from PIL import Image, ImageDraw, ImageFont
import textwrap


# SMPL-H face connections for skeleton visualization
SMPL_CONNECTIONS = [
    [0, 1], [0, 2], [0, 3],
    [1, 4], [2, 5], [3, 6],
    [4, 7], [5, 8], [6, 9],
    [7, 10], [8, 11],
    [9, 12], [9, 13], [9, 14],
    [12, 15], [13, 16], [14, 17],
    [16, 18], [17, 19],
    [18, 20], [19, 21]
]

# Color scheme for rendering
COLORS = {
    "white": [1.0, 1.0, 1.0],
    "black": [0.0, 0.0, 0.0],
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "lightblue": [0.7, 0.8, 1.0],
    "gray": [0.5, 0.5, 0.5],
    "lightgray": [0.8, 0.8, 0.8],
    "vertex": [0.8, 0.8, 0.8]
}


def add_text_overlay(image_array, text, font_size=24, text_color=(0, 0, 0), bg_color=(255, 255, 255, 180)):
    """
    Add text overlay to a rendered image.
    
    Args:
        image_array: numpy array of the image (H, W, 3) in RGB format
        text: text to overlay
        font_size: font size for the text
        text_color: RGB color for the text
        bg_color: RGBA color for the background rectangle
    
    Returns:
        numpy array with text overlay
    """
    if text is None or text.strip() == "":
        return image_array  # Return original image if no text
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_array)
    
    # Create a drawing object
    draw = ImageDraw.Draw(pil_image, 'RGBA')
    
    # Try to use a default font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except OSError:
            try:
                # Try common Linux font paths
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()
    
    # Wrap text to prevent overflow
    wrapped_text = '\n'.join(textwrap.wrap(text, width=50))
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), wrapped_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position text in top-left corner with some padding
    padding = 10
    x = padding
    y = padding
    
    # Draw background rectangle
    bg_rect = [x - 5, y - 5, x + text_width + 5, y + text_height + 5]
    draw.rectangle(bg_rect, fill=bg_color)
    
    # Draw text
    draw.text((x, y), wrapped_text, font=font, fill=text_color)
    
    # Convert back to numpy array
    return np.array(pil_image)


def look_at(eye, target, up=[0, 0, 1]):
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.eye(4)
    mat[:3, 0] = x
    mat[:3, 1] = y
    mat[:3, 2] = z
    mat[:3, 3] = eye
    return mat


class MeshViewer:
    """
    A simplified mesh viewer for rendering SMPL sequences.
    """
    
    def __init__(
        self,
        pyrender,
        width=1200,
        height=800,
        use_offscreen=True,
        follow_camera=False,
        camera_intrinsics=None,
        img_extn="png",
        default_cam_offset=[0.0, 4.0, 1.25],
        default_cam_rot=None,
        text_overlay=None,  # Add text overlay parameter
    ):
        self.pyrender = pyrender
        self.use_offscreen = use_offscreen
        self.follow_camera = follow_camera
        self.img_extn = img_extn
        self.figsize = (width, height)
        self.text_overlay = text_overlay  # Store text overlay
        
        # Animation settings
        self.animation_len = -1
        self.animation_frame_idx = 0
        self.animation_render_time = 0
        
        # Mesh sequences
        self.animated_seqs = []
        self.animated_seqs_type = []
        self.animated_nodes = []
        self.light_nodes = []
        self.animation_frame_idx = 0
        
        # Scene setup
        self.scene = self.pyrender.Scene(
            bg_color=COLORS["white"], 
            ambient_light=(0.5, 0.5, 0.5)  # Increased ambient lighting
        )
        
        # Camera setup
        self.default_cam_offset = np.array(default_cam_offset)
        self.default_cam_rot = np.array(default_cam_rot) if default_cam_rot is not None else np.array([0, 0, 0])
        
        # Create camera
        if camera_intrinsics is not None:
            fx, fy, cx, cy = camera_intrinsics
            camera = self.pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        else:
            camera = self.pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        
        # Camera pose - identity rotation, only translation
        camera_pose = look_at([0, 4, 1.25], [0, 0, 1])
        print("Camera pose:\n", camera_pose)
        print("Determinant:", np.linalg.det(camera_pose[:3, :3]))
        print("Is finite:", np.isfinite(camera_pose).all())
        
        self.camera_node = self.scene.add(camera, pose=camera_pose)
        
        # Lighting - use STMC's simple approach
        light = self.pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
        self.scene.add(light, pose=camera_pose)
        
        # Renderer
        if self.use_offscreen:
            self.viewer = self.pyrender.OffscreenRenderer(
                *self.figsize, point_size=2.75
            )
        else:
            # For interactive viewing (not implemented in this simplified version)
            raise NotImplementedError("Interactive viewing not implemented")
    
    def use_raymond_lighting(self, intensity=1.0):
        """Add Raymond lighting to the scene."""
        # Remove existing lights
        for node in self.scene.get_nodes():
            if node.light is not None:
                self.scene.remove_node(node)
        
        # Add new lights with better positioning
        light_poses = [
            [0, 0, 2, 0, 0, 0],    # Top light
            [0, 0, -2, 0, 0, 0],   # Bottom light
            [0, 2, 0, 0, 0, 0],    # Front light
            [0, -2, 0, 0, 0, 0],   # Back light
            [2, 0, 0, 0, 0, 0],    # Right light
            [-2, 0, 0, 0, 0, 0],   # Left light
        ]
        
        for i, pose in enumerate(light_poses):
            # Vary light intensity for better visibility
            if i == 0:  # Main top light
                light_intensity = intensity * 1.5
            else:
                light_intensity = intensity * 0.8
            light = self.pyrender.DirectionalLight(color=[light_intensity] * 3)
            light_pose = np.eye(4)
            light_pose[:3, 3] = pose[:3]
            self.scene.add(light, pose=light_pose)
    
    def add_mesh_seq(self, body_mesh_seq, progress_bar=tqdm):
        """Add a sequence of meshes for animation."""
        if len(body_mesh_seq) == 0:
            return
        
        # Set animation length based on first sequence
        if self.animation_len == -1:
            self.animation_len = len(body_mesh_seq)
        else:
            assert len(body_mesh_seq) == self.animation_len, "All sequences must have the same length"
        
        # Convert trimesh to pyrender meshes
        pyrender_mesh_seq = []
        if progress_bar is not None:
            mesh_iter = progress_bar(body_mesh_seq, desc="Converting meshes")
        else:
            mesh_iter = body_mesh_seq
            
        for i, mesh in enumerate(mesh_iter):
            if isinstance(mesh, trimesh.Trimesh):
                # Add material to make mesh visible
                material = self.pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.8, 0.8, 0.8, 1.0],  # Light gray
                    metallicFactor=0.0,
                    roughnessFactor=0.7
                )
                pyrender_mesh = self.pyrender.Mesh.from_trimesh(mesh.copy(), material=material)
                print(f"Created mesh {i}: vertices={mesh.vertices.shape}, faces={mesh.faces.shape}")
                pyrender_mesh_seq.append(pyrender_mesh)
            else:
                pyrender_mesh_seq.append(mesh)
        
        self.animated_seqs.append(pyrender_mesh_seq)
        self.animated_seqs_type.append("mesh")
        
        # Create the corresponding node in the scene (STMC's approach)
        if len(pyrender_mesh_seq) > 0:
            seq_id = len(self.animated_seqs) - 1
            anim_node = self.scene.add(pyrender_mesh_seq[0], f"anim-mesh-{seq_id:02d}")
            self.animated_nodes.append(anim_node)
    
    def update_frame(self):
        """Update frame to show the current animation_frame_idx (STMC's approach)."""
        for seq_idx in range(len(self.animated_seqs)):
            if self.animation_frame_idx < len(self.animated_seqs[seq_idx]):
                cur_mesh = self.animated_seqs[seq_idx][self.animation_frame_idx]
                
                # Replace the old mesh (STMC's approach)
                if seq_idx < len(self.animated_nodes):
                    anim_node = self.animated_nodes[seq_idx]
                    anim_node.mesh = cur_mesh
    
    def add_ground(self, ground_plane=None, xyz_orig=None, color0=None, color1=None, alpha=1.0):
        """Add a ground plane to the scene."""
        if color0 is None:
            color0 = [0.8, 0.9, 0.9]
        if color1 is None:
            color1 = [0.6, 0.7, 0.7]
        
        # Create a simple ground plane
        ground_size = 10.0
        ground_height = 0.0
        
        if xyz_orig is not None:
            ground_height = xyz_orig[2]
        
        # Create ground mesh
        ground_vertices = np.array([
            [-ground_size, -ground_size, ground_height],
            [ground_size, -ground_size, ground_height],
            [ground_size, ground_size, ground_height],
            [-ground_size, ground_size, ground_height]
        ])
        
        ground_faces = np.array([[0, 1, 2], [0, 2, 3]])
        
        # Create ground mesh without face colors to avoid pyrender issues
        ground_mesh = trimesh.Trimesh(
            vertices=ground_vertices,
            faces=ground_faces,
            process=False
        )
        
        # Create a simple material for the ground
        ground_material = self.pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.8, 0.9, 0.9, 1.0],  # Light blue-gray
            metallicFactor=0.0,
            roughnessFactor=0.8
        )
        
        pyrender_ground = self.pyrender.Mesh.from_trimesh(ground_mesh, material=ground_material)
        self.scene.add(pyrender_ground, name="ground")
    
    def set_render_settings(self, out_path, wireframe=False, RGBA=False, single_frame=False):
        """Set rendering settings."""
        self.render_wireframe = wireframe
        self.render_RGBA = RGBA
        self.render_path = out_path
        self.single_frame = single_frame
    
    def animate(self, fps=20, start=None, end=None, progress_bar=tqdm):
        """Animate the mesh sequence and save frames."""
        if self.animation_len == -1:
            print("No animation sequences added")
            return
        
        # Create output directory
        os.makedirs(self.render_path, exist_ok=True)
        
        # Determine frame range
        if start is None:
            start = 0
        if end is None:
            end = self.animation_len
        
        # Render each frame using STMC's approach
        if progress_bar is not None:
            frame_iter = progress_bar(range(start, end), desc="Rendering frames")
        else:
            frame_iter = range(start, end)
            
        print(f"Rendering {len(frame_iter)} frames...")
        
        # Initialize animation frame index
        self.animation_frame_idx = 0
        
        for frame_idx in frame_iter:
            self.animation_frame_idx = frame_idx
            
            # Update frame (STMC's approach)
            self.update_frame()
            
            # Render frame
            if self.render_RGBA:
                color, depth = self.viewer.render(self.scene, flags=self.pyrender.RenderFlags.RGBA)
            else:
                color, depth = self.viewer.render(self.scene)
            
            # Apply text overlay if available
            if self.text_overlay is not None:
                color = add_text_overlay(color, self.text_overlay)
            
            # Save frame
            frame_path = os.path.join(self.render_path, f"frame_{frame_idx:04d}.{self.img_extn}")
            
            if self.render_RGBA:
                import imageio
                imageio.imwrite(frame_path, color)
            else:
                import imageio
                imageio.imwrite(frame_path, color)


def viz_smpl_seq(
    pyrender,
    out_path,
    body,
    start=None,
    end=None,
    imw=720,
    imh=720,
    fps=20,
    use_offscreen=True,
    follow_camera=True,
    progress_bar=tqdm,
    contacts=None,
    render_body=True,
    render_joints=False,
    render_skeleton=False,
    render_ground=True,
    ground_plane=None,
    wireframe=False,
    RGBA=False,
    joints_seq=None,
    joints_vel=None,
    vtx_list=None,
    points_seq=None,
    points_vel=None,
    static_meshes=None,
    camera_intrinsics=None,
    img_seq=None,
    point_rad=0.015,
    skel_connections=SMPL_CONNECTIONS,
    img_extn="png",
    ground_alpha=1.0,
    body_alpha=None,
    mask_seq=None,
    cam_offset=[0.0, 2.2, 0.9],
    ground_color0=[0.8, 0.9, 0.9],
    ground_color1=[0.6, 0.7, 0.7],
    skel_color=[0.5, 0.5, 0.5],
    joint_rad=0.015,
    point_color=[0.0, 0.0, 1.0],
    joint_color=[0.0, 1.0, 0.0],
    contact_color=[1.0, 0.0, 0.0],
    vertex_color=COLORS["vertex"],
    render_bodies_static=None,
    render_points_static=None,
    cam_rot=None,
    text_overlay=None,  # Add text overlay parameter
):
    """
    Visualizes the body model output of an SMPL sequence.
    
    Args:
        pyrender: Pyrender module
        out_path: Output path for the rendered video
        body: Body model output from SMPL forward pass (where the sequence is the batch)
        start: Start frame index
        end: End frame index
        imw: Image width
        imh: Image height
        fps: Frames per second
        use_offscreen: Whether to use offscreen rendering
        follow_camera: Whether to follow camera
        progress_bar: Progress bar function
        render_body: Whether to render the body mesh
        render_joints: Whether to render joints
        render_skeleton: Whether to render skeleton
        render_ground: Whether to render ground
        wireframe: Whether to render in wireframe mode
        RGBA: Whether to render in RGBA mode
        cam_offset: Camera offset
        ground_color0: Ground color 1
        ground_color1: Ground color 2
        body_alpha: Body alpha value
        text_overlay: Text to overlay on each frame (optional)
    """
    
    # Convert tensors to numpy if needed
    c2c = lambda tensor: tensor.detach().to(torch.float32).cpu().numpy() if torch.is_tensor(tensor) else tensor
    
    if contacts is not None and torch.is_tensor(contacts):
        contacts = c2c(contacts)
    
    # Create mesh sequence from body vertices
    if render_body:
        nv = body.v.size(1)
        vertex_colors = np.tile(vertex_color, (nv, 1))
        if body_alpha is not None:
            vtx_alpha = np.ones((vertex_colors.shape[0], 1)) * body_alpha
            vertex_colors = np.concatenate([vertex_colors, vtx_alpha], axis=1)
        
        faces = c2c(body.f)
        body_mesh_seq = []
        for i in range(body.v.size(0)):
            vertices = c2c(body.v[i])
            print(f"Frame {i}: vertices range X[{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}], "
                  f"Y[{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}], "
                  f"Z[{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}]")
            
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors,
                process=False,
            )
            body_mesh_seq.append(mesh)
    else:
        body_mesh_seq = []
    
    # Initialize mesh viewer
    mv = MeshViewer(
        pyrender,
        width=imw,
        height=imh,
        use_offscreen=use_offscreen,
        follow_camera=follow_camera,
        camera_intrinsics=camera_intrinsics,
        img_extn=img_extn,
        default_cam_offset=cam_offset,
        default_cam_rot=cam_rot,
        text_overlay=text_overlay,  # Pass text overlay to MeshViewer
    )
    
    # Add body mesh sequence
    if render_body and render_bodies_static is None:
        mv.add_mesh_seq(body_mesh_seq, progress_bar=progress_bar)
    elif render_body and render_bodies_static is not None:
        # Add static meshes at intervals
        static_meshes = [
            body_mesh_seq[i]
            for i in range(len(body_mesh_seq))
            if i % render_bodies_static == 0
        ]
        mv.add_mesh_seq(static_meshes, progress_bar=progress_bar)
    
    # Add ground
    if render_ground:
        xyz_orig = None
        if ground_plane is not None:
            if render_body and len(body_mesh_seq) > 0:
                xyz_orig = body_mesh_seq[0].vertices[0, :]
            elif render_joints and joints_seq is not None:
                xyz_orig = joints_seq[0][0, :]
            elif points_seq is not None:
                xyz_orig = points_seq[0][0, :]
        
        mv.add_ground(
            ground_plane=ground_plane,
            xyz_orig=xyz_orig,
            color0=ground_color0,
            color1=ground_color1,
            alpha=ground_alpha,
        )
    
    # Set render settings
    mv.set_render_settings(
        out_path=out_path,
        wireframe=wireframe,
        RGBA=RGBA,
        single_frame=(render_points_static is not None or render_bodies_static is not None)
    )
    
    # Animate and save
    try:
        mv.animate(fps=fps, start=start, end=end, progress_bar=progress_bar)
    except RuntimeError as err:
        print(f"Could not render properly with the error: {str(err)}")
    
    # Clean up
    del mv


class Video:
    """
    Simple video class for converting frames to video.
    """
    
    def __init__(self, frames_dir, fps=20):
        self.frames_dir = frames_dir
        self.fps = fps
    
    def save(self, output_path):
        """Convert frames to video using ffmpeg."""
        try:
            # Try libopenh264 first (usually available)
            cmd = [
                'ffmpeg', '-y', '-framerate', str(self.fps),
                '-i', os.path.join(self.frames_dir, 'frame_%04d.png'),
                '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p',
                output_path
            ]
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"libopenh264 failed, trying mpeg4...")
                # Fallback to mpeg4
                cmd = [
                    'ffmpeg', '-y', '-framerate', str(self.fps),
                    '-i', os.path.join(self.frames_dir, 'frame_%04d.png'),
                    '-c:v', 'mpeg4', '-pix_fmt', 'yuv420p',
                    output_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"mpeg4 also failed, trying raw video...")
                    # Last resort: raw video
                    cmd = [
                        'ffmpeg', '-y', '-framerate', str(self.fps),
                        '-i', os.path.join(self.frames_dir, 'frame_%04d.png'),
                        '-c:v', 'rawvideo', '-pix_fmt', 'rgb24',
                        output_path.replace('.mp4', '.avi')
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        print(f"All video encoders failed. Frames saved in: {self.frames_dir}")
                        print(f"FFmpeg error: {result.stderr}")
                        return False
            
            print(f"Video saved successfully to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating video: {e}")
            print(f"Frames are available in: {self.frames_dir}")
            return False 