import numpy as np
from util.camera_pose_visualizer import CameraPoseVisualizer
from PIL import Image

if __name__ == '__main__':
    # argument : the minimum/maximum value of x, y, z
    visualizer = CameraPoseVisualizer([-20, 30], [-20, 30], [0, 50])
    # visualizer = CameraPoseVisualizer()
    P1 = np.eye(4)

    image = np.random.rand(100, 50)
    
    visualizer.image_plane(
    P1,
    image,
    focal_len_scaled=40,
    aspect_ratio=0.17
    )
    # argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
    visualizer.extrinsic2pyramid(P1, 'c', 10)
    
    # auto-fit to content
    visualizer.auto_bounds(margin=0.10, equal_aspect=True)
    
    visualizer.show()

    # Save screenshot
    visualizer.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    visualizer.fig.savefig(
        "cam_and_frame.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )
