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
            shade=False
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
