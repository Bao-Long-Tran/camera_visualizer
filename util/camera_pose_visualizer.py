import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

class CameraPoseVisualizer:
    def __init__(self, xlim=None, ylim=None, zlim=None):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.plotly_data = None  # plotly data traces
        self.ax.set_aspect("auto")
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)
        if zlim is not None:
            self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        
        # store all scene points here
        self._scene_points = []
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color_map='red', focal_len_scaled=5, aspect_ratio=0.3, plotly_viz=False, legend_group='g', name='n', show_legend=True):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio,  focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio,   focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio,  focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        # register this camera for auto-fit
        self._add_scene_points(vertex_transformed[:, :3])
        
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]

        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)

        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

        if plotly_viz:
            color = sample_colorscale('rainbow', color_map)[0]
            self.plotly_data = self.draw_interactive(meshes=meshes, color=color, legend_group=legend_group, name=name, show_legend=show_legend)

    def draw_interactive(self, meshes, color, legend_group, name, show_legend):

        x = [polygon[0] for vertice in meshes for polygon in vertice]
        y = [polygon[1] for vertice in meshes for polygon in vertice]
        z = [polygon[2] for vertice in meshes for polygon in vertice]

        data = go.Mesh3d(x=x, y=y, z=z, opacity=1, color=color, showlegend=show_legend, legendgroup=legend_group, name=name)

        return data
    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        # plt.title('Extrinsic Parameters')
        plt.show()
        
    
    
    ## custom
    def image_plane(self,
                    extrinsic,
                    image,
                    focal_len_scaled=5,
                    aspect_ratio=0.3,
                    alpha=0.9,
                    cmap="gray"):
        """
        image:
            - grayscale: (H, W)
            - RGB:       (H, W, 3)
            - RGBA:      (H, W, 4)
        """

        image = np.asarray(image)
        H, W = image.shape[:2]

        plane_h = focal_len_scaled * aspect_ratio * 2
        plane_w = plane_h * (W / H)

        xs = np.linspace(-plane_w / 2, plane_w / 2, W)
        ys = np.linspace(plane_h / 2, -plane_h / 2, H)   # reverse to avoid upside-down

        X, Y = np.meshgrid(xs, ys)
        Z = np.ones_like(X) * focal_len_scaled

        points = np.stack(
            [X.reshape(-1), Y.reshape(-1), Z.reshape(-1), np.ones(H * W)],
            axis=1
        )

        points_world = points @ extrinsic.T

        Xw = points_world[:, 0].reshape(H, W)
        Yw = points_world[:, 1].reshape(H, W)
        Zw = points_world[:, 2].reshape(H, W)

        if image.ndim == 2:
            img = image.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            facecolors = plt.cm.get_cmap(cmap)(img)
            facecolors[..., -1] = alpha

        elif image.ndim == 3 and image.shape[2] == 3:
            facecolors = image.astype(np.float32)
            if facecolors.max() > 1.0:
                facecolors /= 255.0
            alpha_channel = np.ones((H, W, 1), dtype=np.float32) * alpha
            facecolors = np.concatenate([facecolors, alpha_channel], axis=2)

        elif image.ndim == 3 and image.shape[2] == 4:
            facecolors = image.astype(np.float32)
            if facecolors.max() > 1.0:
                facecolors /= 255.0
            facecolors[..., -1] = alpha

        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        # register 4 plane corners needed for auto-bounds
        corners = np.array([
            [Xw[0, 0],     Yw[0, 0],     Zw[0, 0]],
            [Xw[0, -1],    Yw[0, -1],    Zw[0, -1]],
            [Xw[-1, 0],    Yw[-1, 0],    Zw[-1, 0]],
            [Xw[-1, -1],   Yw[-1, -1],   Zw[-1, -1]],
        ])
        self._add_scene_points(corners)

        self.ax.plot_surface(
            Xw,
            Yw,
            Zw,
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            shade=False,
            linewidth=0,
            edgecolor="none",
            antialiased=False
        )
    
    # Add points
    def _add_scene_points(self, pts):
        """
        pts: array-like of shape (N, 3)
        """
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"pts must have shape (N, 3), got {pts.shape}")
        self._scene_points.append(pts)
    
    
    # Auto select content region. Rotate a bit too see all content
    def auto_bounds(self, margin=0.08, equal_aspect=True):
        """
        Automatically fit x/y/z limits tightly around all registered content.
        """
        if len(self._scene_points) == 0:
            return

        pts = np.concatenate(self._scene_points, axis=0)   # (N, 3)

        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)

        center = (mins + maxs) / 2.0
        size = maxs - mins

        # avoid zero-size axis
        size[size == 0] = 1e-6

        if equal_aspect:
            max_range = size.max() * (1.0 + margin)
            half = max_range / 2.0

            self.ax.set_xlim(center[0] - half, center[0] + half)
            self.ax.set_ylim(center[1] - half, center[1] + half)
            self.ax.set_zlim(center[2] - half, center[2] + half)

            # optional, keeps the 3D box visually balanced
            self.ax.set_box_aspect((1, 1, 1))
        else:
            pad = size * margin / 2.0
            self.ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
            self.ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
            self.ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
    
    
    # Action to select crop region for saving figure
    # Press c to start capture
    # Click two points: top-left and bottom-right
    def enable_interactive_crop(
            self,
            full_path="full_view.png",
            crop_path="cropped_view.png",
            dpi=300,
            trigger_key="c"
        ):
        """
        Enable interactive crop selection.

        Usage:
            visualizer.enable_interactive_crop()
            plt.show()

        In the plot window:
            1. Rotate the 3D plot
            2. Press 'c'
            3. Click two points: top-left and bottom-right
            4. Cropped image is saved
        """

        self._crop_full_path = full_path
        self._crop_path = crop_path
        self._crop_dpi = dpi
        self._crop_trigger_key = trigger_key
        self._crop_mode = False
        self._crop_points = []

        self.fig.canvas.mpl_connect("key_press_event", self._on_crop_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_crop_click)

        print(f"Interactive crop enabled. Press '{trigger_key}' then click two points.")


    def _on_crop_key(self, event):
        if event.key == self._crop_trigger_key:
            self._crop_mode = True
            self._crop_points = []
            print("Crop mode ON. Click two points: top-left and bottom-right.")


    def _on_crop_click(self, event):
        if not getattr(self, "_crop_mode", False):
            return

        if event.x is None or event.y is None:
            return

        # Convert mouse position from display pixels to figure-relative coordinates
        # Figure-relative coords: (0,0) bottom-left, (1,1) top-right
        fig_coord = self.fig.transFigure.inverted().transform((event.x, event.y))
        self._crop_points.append(fig_coord)

        print(f"Clicked point {len(self._crop_points)}:", fig_coord)

        if len(self._crop_points) == 2:
            self._crop_current_view()
            self._crop_mode = False
            print(f"Cropped image saved to: {self._crop_path}")


    def _crop_current_view(self):
        from PIL import Image

        # Save the current rotated view
        self.fig.savefig(
            self._crop_full_path,
            dpi=self._crop_dpi,
            bbox_inches=None,
            pad_inches=0
        )

        img = Image.open(self._crop_full_path)
        W, H = img.size

        (x1, y1), (x2, y2) = self._crop_points

        # Convert figure-relative coords to pixel crop box
        left = int(min(x1, x2) * W)
        right = int(max(x1, x2) * W)

        # Matplotlib figure coord: y=0 bottom
        # PIL image coord: y=0 top
        upper = int((1 - max(y1, y2)) * H)
        lower = int((1 - min(y1, y2)) * H)

        print("Crop box:", (left, upper, right, lower))
        # IMPORTANT: create cropped first
        cropped = img.crop((left, upper, right, lower))

        if self._crop_path.lower().endswith(".pdf"):
            cropped = cropped.convert("RGB")
            cropped.save(self._crop_path, "PDF", resolution=self._crop_dpi)
        else:
            cropped.save(self._crop_path)

        print(f"Saved full image to: {self._crop_full_path}")
        print(f"Saved cropped image to: {self._crop_path}")

        if getattr(self, "_crop_close_after_save", True):
            plt.close(self.fig)
