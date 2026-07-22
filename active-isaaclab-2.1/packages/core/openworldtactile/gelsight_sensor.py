from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from matplotlib import pyplot as plt
from typing import TYPE_CHECKING

import cv2
import omni.kit.commands
import omni.usd
from isaacsim.core.prims import XFormPrim
from pxr import Sdf

from isaaclab.sensors import SensorBase, TiledCamera, TiledCameraCfg

from .gelsight_sensor_data import GelSightSensorData
from .simulation_approaches.gelsight_simulator import GelSightSimulator

# from torchvision.transforms import v2


# from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg
# from isaaclab.sensors.camera.camera import Camera
# from isaaclab.sensors.camera.camera_cfg import CameraCfg

if TYPE_CHECKING:
    from .gelsight_sensor_cfg import GelSightSensorCfg


class GelSightSensor(SensorBase):
    cfg: GelSightSensorCfg

    def __init__(self, cfg: GelSightSensorCfg, gelpad_obj=None):
        # initialize base class
        super().__init__(cfg)

        self._prim_view = None

        # sensor camera
        self.camera = None

        # object which represents the gelpad
        self.gelpad_obj = gelpad_obj

        self._indentation_depth: torch.tensor = None

        # simulation approaches for simulating GelSight sensor output
        self.optical_simulator: GelSightSimulator = None
        self.marker_motion_simulator: GelSightSimulator = None
        self.compute_indentation_depth_func = None

        # Create empty variables for storing output data
        self._data = GelSightSensorData()
        self._data.output = dict.fromkeys(self.cfg.data_types, None)

        # Flag to check that sensor is spawned.
        self._is_spawned = False

        # initialize classes for GelSight simulation approaches for simulating GelSight sensor output
        if self.cfg.optical_sim_cfg is not None:
            # initialize class we set in the cfg class of the sim approach
            self.optical_simulator = self.cfg.optical_sim_cfg.simulation_approach_class(
                sensor=self,
                cfg=self.cfg.optical_sim_cfg,
            )

        if self.cfg.marker_motion_sim_cfg is not None:
            if (self.optical_simulator is not None) and (
                self.cfg.optical_sim_cfg.simulation_approach_class
                == self.cfg.marker_motion_sim_cfg.simulation_approach_class
            ):
                # if same class for optical and marker sim, then use same obj
                self.marker_motion_simulator = self.optical_simulator
            else:
                self.marker_motion_simulator = self.cfg.marker_motion_sim_cfg.simulation_approach_class(
                    sensor=self, cfg=self.cfg.marker_motion_sim_cfg
                )

        self._set_debug_vis_flag = False
        self._debug_vis_is_initialized = False
        self._openworldtactile_value_texture = None
        self._openworldtactile_value_texture_shape = None
        self._openworldtactile_force_rgb_debug = None

    def __del__(self):
        """Unsubscribes from callbacks."""
        # unsubscribe callbacks
        super().__del__()

    # def __str__(self) -> str:
    #     """Returns: A string containing information about the instance."""
    #     # message for class
    #     return (
    #         f"Gelsight Mini @ '{self.cfg.prim_path}': \n"
    #         f"\tdata types   : {list(self._data.output)} \n"
    #         f"\tupdate period (s): {self.cfg.update_period}\n"
    #         f"\tframe        : {self.frame}\n"
    #         f"\camera resolution        : {self.camera_resolution}\n"
    #         f"\ttactile resolution        : {self.tactile_image_resolution}\n"
    #         f"\twith shadows        : {self._simulate_shadows}\n"
    #         # f"\tposition     : {self._data.position} \n"
    #         # f"\torientation  : {self._data.orientation} \n"
    #     )

    """
    Properties
    """

    @property
    def data(self) -> GelSightSensorData:
        """Data related to Camera sensor."""
        # update sensors if needed
        self._update_outdated_buffers()
        return self._data

    @property
    def frame(self) -> torch.tensor:
        """Frame number when the measurement took place."""
        return self._frame

    @property
    def tactile_image_shape(self) -> tuple[int, int, int]:
        """Shape of the simulated tactile RGB image, i.e. (channels, height, width)."""
        return (self.cfg.optical_sim_cfg.tactile_img_res[1], self.cfg.optical_sim_cfg.tactile_img_res[0], 3)

    @property
    def camera_resolution(self) -> tuple[int, int]:
        """The resolution (width x height) of the camera used by this sensor."""
        return self.cfg.sensor_camera_cfg.resolution[0], self.cfg.sensor_camera_cfg.resolution[1]  # type: ignore

    @property
    def indentation_depth(self):
        """How deep objects are inside the gel pad of the sensor.

        Units: [mm]
        """
        return self._indentation_depth

    @property
    def prim_view(self):
        return self._prim_view

    """
    Operations
    """

    # MARK: reset
    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        # note: cannot do smart indexing here since we do a for loop over data.
        if env_ids is None:
            env_ids = self._ALL_INDICES  # type: ignore

        # reset camera
        if self.camera is not None:
            self.camera.reset()

        # reset the buffer
        # self._data.position = None
        # self._data.orientation = None
        # self._data.image_resolution = self.image_resolution

        self._indentation_depth[env_ids] = 0

        # reset height map
        self._data.output["height_map"][env_ids] = 0
        # torch.zeros(
        #     (env_ids.size(), self.camera_cfg.height, self.camera_cfg.width),
        #     device=self.cfg.device
        # )

        # if self._interpolate_height_map:
        #     resized = F.resize(self._data.output["height_map"], (self.taxim.sensor_params.height,self.taxim.sensor_params.width))
        #     #TODO should I compute press depth after interpolation or before?
        #     self._data.output["height_map"] = resized

        if "camera_depth" in self._data.output:
            self._data.output["camera_depth"][env_ids] = 0

        if "tactile_force_field" in self._data.output:
            self._data.output["tactile_force_field"][env_ids] = 0

        # simulate optical/marker output, but without indentation
        if (self.optical_simulator is not None) and ("tactile_rgb" in self._data.output):
            self._data.output["tactile_rgb"][:] = self.optical_simulator.optical_simulation()
            self.optical_simulator.reset()

        if (self.marker_motion_simulator is not None) and ("marker_motion" in self._data.output):
            # height_map_shifted = self.taxim._get_shifted_height_map(self._indentation_depth, self._data.output["height_map"])
            self._data.output["marker_motion"][:] = self.marker_motion_simulator.marker_motion_simulation()
            # (yy_init_pos, xx_init_pos), i.e. along height x width of tactile img
            self._data.output["init_marker_pos"] = ([0], [0])

            self.marker_motion_simulator.reset()

        # Reset the frame count
        self._frame[env_ids] = 0

    ####
    # Implementation of abstract methods of base sensor class
    ####
    # MARK: _init_impl
    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers."""
        print(f"Initializing GelSight Sensor `{self.cfg.prim_path}`...")

        # Initialize parent class
        super()._initialize_impl()

        self._prim_view = XFormPrim(prim_paths_expr=self.cfg.prim_path, name=f"{self.cfg.prim_path}", usd=False)
        self._prim_view.initialize()
        # Check that sizes are correct
        if self._prim_view.count != self._num_envs:
            raise RuntimeError(
                f"Number of sensor prims in the view ({self._prim_view.count}) does not match"
                f" the number of environments ({self._num_envs})."
            )

        # set device, if specified (per default the same as the simulation)
        if self.cfg.device is not None:
            self._device = self.cfg.device

        # Create all env_ids buffer
        self._ALL_INDICES = torch.arange(self._num_envs, device=self._device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)

        if self.cfg.sensor_camera_cfg is not None:
            self.camera_cfg: TiledCameraCfg = TiledCameraCfg(
                prim_path=self.cfg.prim_path + self.cfg.sensor_camera_cfg.prim_path_appendix,
                update_period=self.cfg.sensor_camera_cfg.update_period,
                height=self.cfg.sensor_camera_cfg.resolution[1],
                width=self.cfg.sensor_camera_cfg.resolution[0],
                data_types=self.cfg.sensor_camera_cfg.data_types,
                update_latest_camera_pose=True,  # needed for FEM based marker sim
                spawn=None,  # use camera which is part of the GelSight Mini Asset
                # note: clipping range doesn't matter for existing camera prim -> only applied when camera is spawned # TODO fix?
                # spawn=sim_utils.PinholeCameraCfg(
                #    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
                # ),
                # depth_clipping_behavior="max", # doesn't work, cause "max" value is taking from spawn config, which we dont have
            )
            self.camera = TiledCamera(cfg=self.camera_cfg)

            # use normal camera
            # self.camera_cfg: CameraCfg = CameraCfg(
            #         prim_path= self.cfg.prim_path + self.cfg.sensor_camera_cfg.prim_path_appendix,
            #         update_period= self.cfg.sensor_camera_cfg.update_period,
            #         height= self.cfg.sensor_camera_cfg.resolution[1],
            #         width= self.cfg.sensor_camera_cfg.resolution[0],
            #         data_types= self.cfg.sensor_camera_cfg.data_types,
            #         spawn= None, # use camera which is part of the GelSight Mini Asset
            #         # spawn=sim_utils.PinholeCameraCfg(
            #         #    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            #         # ),
            #         #depth_clipping_behavior="max", # doesn't work, cause "max" value is taking from spawn config, which we dont have
            # )
            # self.camera = Camera(cfg=self.camera_cfg)

            # need to initialize the camera manually, since its not part of the scene cfg
            self.camera._initialize_impl()
            self.camera._is_initialized = True

        self._data.output["height_map"] = torch.zeros(
            (self._num_envs, self.camera_cfg.height, self.camera_cfg.width), device=self.cfg.device
        )

        # Check that sensor has been spawned
        # if sensor_prim_path is None:
        #     if not self._is_spawned:
        #         raise RuntimeError("Initializing the camera failed! Please provide a valid argument for `prim_path`.")
        #     sensor_prim_path = self.prim_path

        # initialize sim approaches
        if self.optical_simulator is not None:
            self.optical_simulator._initialize_impl()

        if self.marker_motion_simulator is not None:
            self.marker_motion_simulator._initialize_impl()

        # create buffers for output
        if "camera_depth" in self.cfg.data_types:
            self._data.output["camera_depth"] = torch.zeros(
                (self._num_envs, self.camera_resolution[1], self.camera_resolution[0], 1), device=self.cfg.device
            )
        if "camera_rgb" in self.cfg.data_types:
            self._data.output["camera_rgb"] = torch.zeros(
                (self._num_envs, self.camera_resolution[1], self.camera_resolution[0], 3), device=self.cfg.device
            )
        if "tactile_rgb" in self.cfg.data_types:
            self._data.output["tactile_rgb"] = torch.zeros(
                (
                    self._num_envs,
                    self.cfg.optical_sim_cfg.tactile_img_res[1],
                    self.cfg.optical_sim_cfg.tactile_img_res[0],
                    3,
                ),
                device=self.cfg.device,
            )
        if "tactile_force_field" in self.cfg.data_types:
            self._data.output["tactile_force_field"] = torch.zeros(
                (self._num_envs, self.camera_resolution[1], self.camera_resolution[0], 3), device=self.cfg.device
            )
        if "marker_motion" in self.cfg.data_types:
            # self._data.output["marker_motion"]= torch.zeros(
            #     (
            #         self._num_envs,
            #         self.cfg.marker_motion_sim_cfg.marker_params.num_markers_row,
            #         self.cfg.marker_motion_sim_cfg.marker_params.num_markers_col,
            #         2 # two, because each marker at (row,col) has position value (y,x)
            #     ),
            #     device=self.cfg.device
            # )
            self._data.output["marker_motion"] = torch.zeros(
                (
                    self._num_envs,
                    2,
                    self.cfg.marker_motion_sim_cfg.marker_params.num_markers,
                    2,  # two, because each marker at (row,col) has position value (y,x)
                ),
                device=self.cfg.device,
            )

        # set how the indentation depth should be computed
        if (self.cfg.compute_indentation_depth_class) == "optical_sim" and (self.optical_simulator is not None):
            self.compute_indentation_depth_func = self.optical_simulator.compute_indentation_depth
        elif (self.cfg.compute_indentation_depth_class == "marker_motion_sim") and (
            self.marker_motion_simulator is not None
        ):
            self.compute_indentation_depth_func = self.marker_motion_simulator.compute_indentation_depth
        else:
            self.compute_indentation_depth_func = None

        # Create all env_ids buffer
        self._ALL_INDICES = torch.arange(self._num_envs, device=self._device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        # reset internal buffers
        self.reset()

        # create debug visualization
        self._initialize_debug_vis(self._initialize_debug_vis_flag)

        # todo print init data
        # print(self)

    # MARK: _update_buffers_impl
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Updates the internal buffer with the latest data from the sensor.

        This function reads ...

        """
        # -- pose
        # self._data.position = self._sensor_prim.GetAttribute("xformOp:translate").Get()
        # self._data.orientation = self._sensor_prim.GetAttribute(
        #     "xformOp:rotation"
        # ).Get()

        self._frame[env_ids] += 1

        # -- update camera buffer
        if self.camera is not None:
            self.camera._timestamp = self._timestamp
            self.camera.update(dt=0, force_recompute=True)

        if self.compute_indentation_depth_func is not None:
            # -- height_map
            self._get_height_map()
            # -- pressing depth
            self._indentation_depth[:] = self.compute_indentation_depth_func()  # type: ignore #todo uncomment

        if "camera_depth" in self._data.output:
            self._get_camera_depth()

        if "camera_rgb" in self._data.output:
            self._data.output["camera_rgb"][:] = self.camera.data.output["rgb"]

        if (self.optical_simulator is not None) and ("tactile_rgb" in self.cfg.data_types):
            # self.optical_simulator.height_map = self._data.output["height_map"]
            self._data.output["tactile_rgb"][:] = self.optical_simulator.optical_simulation()

        if (self.marker_motion_simulator is not None) and ("marker_motion" in self.cfg.data_types):
            self._data.output["marker_motion"][:] = self.marker_motion_simulator.marker_motion_simulation()

    def _set_debug_vis_impl(self, debug_vis: bool):
        # we actually set the debug_vis in _initialize_impl, since we need the _prim_view, which
        # is only correctly initialized after _initialize_impl method (in ManagerBased workflow, in Direct workflow you can control it yourself)
        self._initialize_debug_vis_flag: bool = debug_vis

    def _initialize_debug_vis(self, debug_vis: bool):
        """Creates an USD attribute for the sensor assets, which can visualize the tactile image.

        Select the GelSight sensor case whose output you want to see in the Isaac Sim GUI,
        i.e. the `gelsight_mini_case` Xform (not the mesh!).
        Scroll down in the properties panel to "Raw Usd Properties" and click "Extra Properties".
        There is an attribute called "show_tactile_image".
        Toggle it on to show the sensor output in the GUI.

        If only optical simulation is used, then only an optical img is displayed.
        If only the marker simulatios is used, then only an image displaying the marker positions is displayed.
        If both, optical and marker simulation, are used, then the images are overlaid.

        > Method has to be called after the prim_view was initialized.
        """
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # need to create the attribute for the debug_vis here since it depends on self._prim_view
            for prim in self._prim_view.prims:
                # creates USD attributes for each data type, which can be found in the Isaac GUI under "Raw Usd Properties -> "Extra Properties"
                if "camera_depth" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_camera_depth", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)
                if "camera_rgb" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_camera_rgb", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)
                if "tactile_rgb" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_tactile_rgb", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)
                if "tactile_force_field" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_tactile_force_field", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)
                if "marker_motion" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_marker_motion", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)

            if not hasattr(self, "_windows"):
                # dict of windows that show the simulated tactile images, if the attribute of the sensor asset is turned on
                self._windows = {}
                self._img_providers = {}
                # todo check if there is a more efficient implementation than dict of dicts
                if "camera_depth" in self.cfg.data_types:
                    self._windows["camera_depth"] = {}
                    self._img_providers["camera_depth"] = {}
                if "camera_rgb" in self.cfg.data_types:
                    self._windows["camera_rgb"] = {}
                    self._img_providers["camera_rgb"] = {}
                if "tactile_rgb" in self.cfg.data_types and "tactile_force_field" in self.cfg.data_types:
                    self._windows["tactile_rgb_openworldtactile_rgb"] = {}
                    self._img_providers["tactile_rgb_openworldtactile_rgb"] = {}
                elif "tactile_force_field" in self.cfg.data_types:
                    self._windows["tactile_force_field"] = {}
                    self._img_providers["tactile_force_field"] = {}

            if "tactile_rgb" in self.cfg.data_types:
                self.optical_simulator._set_debug_vis_impl(debug_vis)

            if "marker_motion" in self.cfg.data_types:
                self.marker_motion_simulator._set_debug_vis_impl(debug_vis)

            self._debug_vis_is_initialized = True

    def _debug_vis_callback(self, event):
        if not self._debug_vis_is_initialized:
            return

        combined_tactile_debug = (
            "tactile_rgb" in self.cfg.data_types and "tactile_force_field" in self.cfg.data_types
        )

        # Update the GUI windows
        for i, prim in enumerate(
            self._prim_view.prims
        ):  # note: bad that we iterate through all prims multiple times (once per sim data type)
            if "camera_rgb" in self.cfg.data_types:
                show_img = prim.GetAttribute("debug_camera_rgb").Get()
                if show_img:
                    if str(i) not in self._windows["camera_rgb"]:
                        # create a window
                        window = omni.ui.Window(
                            self._prim_view.prim_paths[i] + "/camera_rgb",
                            height=self.cfg.sensor_camera_cfg.resolution[1],
                            width=self.cfg.sensor_camera_cfg.resolution[0],
                        )
                        self._windows["camera_rgb"][str(i)] = window
                        # create image provider
                        self._img_providers["camera_rgb"][
                            str(i)
                        ] = omni.ui.ByteImageProvider()  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    frame = self._data.output["camera_rgb"][i].cpu().numpy()

                    # update image of the window
                    frame = frame.astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)  # cv.COLOR_BGR2RGBA) COLOR_RGB2RGBA
                    height, width, channels = frame.shape
                    with self._windows["camera_rgb"][str(i)].frame:
                        # self._img_providers[str(i)].set_data_array(frame, [width, height, channels]) #method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        self._img_providers["camera_rgb"][str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )  # method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        omni.ui.ImageWithProvider(
                            self._img_providers["camera_rgb"][str(i)]
                        )  # , fill_policy=omni.ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT -> fill_policy by default: specifying the width and height of the item causes the image to be scaled to that size
                elif str(i) in self._windows["camera_rgb"]:
                    # remove window/img_provider from dictionary and destroy them
                    self._windows["camera_rgb"].pop(str(i)).destroy()
                    self._img_providers["camera_rgb"].pop(str(i)).destroy()

            if "camera_depth" in self.cfg.data_types:
                show_img = prim.GetAttribute("debug_camera_depth").Get()
                if show_img:
                    if str(i) not in self._windows["camera_depth"]:
                        # create a window
                        window = omni.ui.Window(
                            self._prim_view.prim_paths[i] + "/camera_depth",
                            height=self.cfg.sensor_camera_cfg.resolution[1],
                            width=self.cfg.sensor_camera_cfg.resolution[0],
                        )
                        self._windows["camera_depth"][str(i)] = window
                        # create image provider
                        self._img_providers["camera_depth"][
                            str(i)
                        ] = omni.ui.ByteImageProvider()  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    frame = self._data.output["camera_depth"][i].cpu().numpy()
                    # # image is channel first, convert to channel last
                    # frame = np.moveaxis(frame, 0, -1)
                    # convert to 3 channel image, to later turn it into 4 channel RGBA for Isaac Widget
                    frame = np.dstack((frame, frame, frame)).astype(np.uint8)
                    frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                    # update image of the window
                    frame = frame.astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)  # cv.COLOR_BGR2RGBA) COLOR_RGB2RGBA
                    height, width, channels = frame.shape
                    with self._windows["camera_depth"][str(i)].frame:
                        # self._img_providers[str(i)].set_data_array(frame, [width, height, channels]) #method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        self._img_providers["camera_depth"][str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )  # method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        omni.ui.ImageWithProvider(
                            self._img_providers["camera_depth"][str(i)]
                        )  # , fill_policy=omni.ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT -> fill_policy by default: specifying the width and height of the item causes the image to be scaled to that size
                elif str(i) in self._windows["camera_depth"]:
                    # remove window/img_provider from dictionary and destroy them
                    self._windows["camera_depth"].pop(str(i)).destroy()
                    self._img_providers["camera_depth"].pop(str(i)).destroy()

            if combined_tactile_debug:
                show_rgb = prim.GetAttribute("debug_tactile_rgb").Get()
                show_force_field = prim.GetAttribute("debug_tactile_force_field").Get()
                show_img = show_rgb or show_force_field
                if show_img:
                    rgb_frame = self._draw_tactile_rgb(self._data.output["tactile_rgb"][i].cpu().numpy())
                    sensor_force_field = self._data.output["tactile_force_field"][i].cpu().numpy()
                    openworldtactile_force_frame = self._draw_openworldtactile_sensor_force_field(sensor_force_field)
                    if openworldtactile_force_frame.shape[:2] != rgb_frame.shape[:2]:
                        openworldtactile_force_frame = cv2.resize(
                            openworldtactile_force_frame,
                            (rgb_frame.shape[1], rgb_frame.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    force_field = sensor_force_field
                    if force_field.shape[:2] != rgb_frame.shape[:2]:
                        force_field = self._resize_openworldtactile_force_field(
                            force_field,
                            target_height=rgb_frame.shape[0],
                            target_width=rgb_frame.shape[1],
                        )
                    gelsight_force_frame = self._draw_tactile_force_field(force_field)
                    openworldtactile_frame = self._draw_openworldtactile_rgb_from_force_field(force_field)
                    openworldtactile_force_frame = self._openworldtactile_force_rgb_debug
                    if openworldtactile_force_frame is None:
                        openworldtactile_force_frame = np.zeros_like(rgb_frame, dtype=np.uint8)
                    elif openworldtactile_force_frame.shape[:2] != rgb_frame.shape[:2]:
                        openworldtactile_force_frame = cv2.resize(openworldtactile_force_frame, (rgb_frame.shape[1], rgb_frame.shape[0]))

                    column_separator = np.full((rgb_frame.shape[0], 2, 3), 32, dtype=np.uint8)
                    blank_frame = np.zeros_like(rgb_frame, dtype=np.uint8)
                    top_row = np.concatenate(
                        (
                            rgb_frame,
                            column_separator,
                            gelsight_force_frame,
                            column_separator,
                            openworldtactile_force_frame,
                        ),
                        axis=1,
                    )
                    bottom_row = np.concatenate(
                        (
                            openworldtactile_frame,
                            column_separator,
                            openworldtactile_force_frame,
                            column_separator,
                            blank_frame,
                        ),
                        axis=1,
                    )
                    row_separator = np.full((2, top_row.shape[1], 3), 32, dtype=np.uint8)
                    frame = np.concatenate((top_row, row_separator, bottom_row), axis=0)

                    if str(i) not in self._windows["tactile_rgb_openworldtactile_rgb"]:
                        window = omni.ui.Window(
                            self._prim_view.prim_paths[i] + "/tactile_rgb_openworldtactile_rgb",
                            height=frame.shape[0],
                            width=frame.shape[1],
                        )
                        self._windows["tactile_rgb_openworldtactile_rgb"][str(i)] = window
                        self._img_providers["tactile_rgb_openworldtactile_rgb"][str(i)] = omni.ui.ByteImageProvider()

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    height, width, channels = frame.shape
                    with self._windows["tactile_rgb_openworldtactile_rgb"][str(i)].frame:
                        self._img_providers["tactile_rgb_openworldtactile_rgb"][str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )
                        omni.ui.ImageWithProvider(self._img_providers["tactile_rgb_openworldtactile_rgb"][str(i)])
                elif str(i) in self._windows["tactile_rgb_openworldtactile_rgb"]:
                    self._windows["tactile_rgb_openworldtactile_rgb"].pop(str(i)).destroy()
                    self._img_providers["tactile_rgb_openworldtactile_rgb"].pop(str(i)).destroy()

            elif "tactile_force_field" in self.cfg.data_types:
                show_img = prim.GetAttribute("debug_tactile_force_field").Get()
                if show_img:
                    if str(i) not in self._windows["tactile_force_field"]:
                        window = omni.ui.Window(
                            self._prim_view.prim_paths[i] + "/tactile_force_field",
                            height=self.cfg.sensor_camera_cfg.resolution[1],
                            width=self.cfg.sensor_camera_cfg.resolution[0],
                        )
                        self._windows["tactile_force_field"][str(i)] = window
                        self._img_providers["tactile_force_field"][str(i)] = omni.ui.ByteImageProvider()

                    frame = self._draw_tactile_force_field(self._data.output["tactile_force_field"][i].cpu().numpy())
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    height, width, channels = frame.shape
                    with self._windows["tactile_force_field"][str(i)].frame:
                        self._img_providers["tactile_force_field"][str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )
                        omni.ui.ImageWithProvider(self._img_providers["tactile_force_field"][str(i)])
                elif str(i) in self._windows["tactile_force_field"]:
                    self._windows["tactile_force_field"].pop(str(i)).destroy()
                    self._img_providers["tactile_force_field"].pop(str(i)).destroy()

        if "tactile_rgb" in self.cfg.data_types and not combined_tactile_debug:
            self.optical_simulator._debug_vis_callback(event)

        if "marker_motion" in self.cfg.data_types:
            self.marker_motion_simulator._debug_vis_callback(event)

    """
    Private Helper methods
    """
    # TODO implement
    # def _create_buffers(self):
    #     """Create buffers for storing data."""
    #     # create the data object
    #     # -- pose of the cameras
    #     self._data.pos_w = torch.zeros((self._prim_view.count, 3), device=self._device)
    #     self._data.quat_w_world = torch.zeros((self._prim_view.count, 4), device=self._device)
    #     # -- intrinsic matrix
    #     self._data.intrinsic_matrices = torch.zeros((self._prim_view.count, 3, 3), device=self._device)
    #     self._data.image_shape = self.image_shape
    #     # -- output data
    #     # lazy allocation of data dictionary
    #     # since the size of the output data is not known in advance, we leave it as None
    #     # the memory will be allocated when the buffer() function is called for the first time.
    #     self._data.output = TensorDict({}, batch_size=self._prim_view.count, device=self.device)
    #     self._data.info = [{name: None for name in self.cfg.data_types} for _ in range(self._prim_view.count)]

    # TODO implement properly
    # def _update_poses(self, env_ids: Sequence[int]):
    #     """Computes the pose of the camera in the world frame with ROS convention.

    #     This methods uses the ROS convention to resolve the input pose. In this convention,
    #     we assume that the camera front-axis is +Z-axis and up-axis is -Y-axis.

    #     Returns:
    #         A tuple of the position (in meters) and quaternion (w, x, y, z).
    #     """
    #     # check camera prim exists
    #     if len(self._sensor_prims) == 0:
    #         raise RuntimeError("Camera prim is None. Please call 'sim.play()' first.")

    #     # get the poses from the view
    #     poses, quat = self._prim_view.get_world_poses(env_ids)
    #     self._data.pos_w[env_ids] = poses
    #     self._data.quat_w_world[env_ids] = convert_orientation_convention(quat, origin="opengl", target="world")

    def _get_camera_depth(self):
        if self.camera is not None:
            depth_output = self.camera.data.output["depth"][
                :, :, :, 0
            ]  # tiled camera gives us data with shape (num_cameras, height, width, num_channels),
            # clip camera values that are = inf
            depth_output[torch.isinf(depth_output)] = self.cfg.sensor_camera_cfg.clipping_range[1]

            self._data.output["camera_depth"] = depth_output.reshape(
                (self._num_envs, 1, self.camera_resolution[1], self.camera_resolution[0])
            )  # add a channel to the depth image for debug_vis
            self._data.output["camera_depth"] *= 1000.0

            # normalize the depth image
            normalized = self._data.output["camera_depth"].view(self._data.output["camera_depth"].size(0), -1)
            normalized -= self.cfg.sensor_camera_cfg.clipping_range[0] * 1000
            normalized /= self.cfg.sensor_camera_cfg.clipping_range[1] * 1000
            normalized = (normalized * 255).type(dtype=torch.uint8)
            self._data.output["camera_depth"] = normalized.reshape(
                (self._num_envs, self.camera_resolution[1], self.camera_resolution[0], 1)
            )  # add a channel to the depth image for debug_vis

        return self._data.output["camera_depth"]

    def _get_height_map(self):
        if self.camera is not None:
            self._data.output["height_map"][:] = self.camera.data.output["depth"][
                :, :, :, 0
            ]  # tiled camera gives us data with shape (num_cameras, height, width, num_channels),
            # clip camera values that are = inf
            self._data.output["height_map"][torch.isinf(self._data.output["height_map"])] = (
                self.cfg.sensor_camera_cfg.clipping_range[1]
            )
            # default unit is meter -> convert to mm for optical sim
            self._data.output["height_map"] *= 1000

            return self._data.output["height_map"]
        else:
            # not setting camera cfg means "no need for camera"
            # e.g. use soft body deformation as height map? -> not implemented yet
            # or that we dont need a height map in general
            pass

    def _draw_tactile_rgb(self, tactile_rgb: np.ndarray) -> np.ndarray:
        frame = tactile_rgb.astype(np.float32)
        if float(np.max(frame)) <= 1.0:
            frame *= 255.0
        frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return frame.astype(np.uint8)

    def _resize_openworldtactile_force_field(
        self,
        force_field: np.ndarray,
        *,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        src_height, src_width = force_field.shape[:2]
        resized = cv2.resize(
            force_field.astype(np.float32),
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if src_width > 0:
            resized[..., 0] *= target_width / float(src_width)
        if src_height > 0:
            resized[..., 1] *= target_height / float(src_height)
        return resized

    def _get_openworldtactile_value_texture(self, height: int, width: int) -> np.ndarray:
        if self._openworldtactile_value_texture is not None and self._openworldtactile_value_texture_shape == (height, width):
            return self._openworldtactile_value_texture

        rng = np.random.default_rng(7)
        texture = np.full((height, width), 210.0, dtype=np.float32)
        texture += rng.normal(0.0, 36.0, size=(height, width)).astype(np.float32)
        texture = cv2.GaussianBlur(texture, (3, 3), 0)

        self._openworldtactile_value_texture = np.clip(texture, 0, 255).astype(np.uint8)
        self._openworldtactile_value_texture_shape = (height, width)
        return self._openworldtactile_value_texture

    def _draw_openworldtactile_rgb_from_force_field(self, force_field: np.ndarray) -> np.ndarray:
        force_field = np.nan_to_num(force_field.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        height, width = force_field.shape[:2]

        pressure = np.clip(force_field[..., 2], 0.0, None)
        max_pressure = float(np.percentile(pressure, 99.0))
        if max_pressure <= 1.0e-6:
            max_pressure = float(np.max(pressure))
        if max_pressure > 1.0e-6:
            pressure = np.clip(pressure / max_pressure, 0.0, 1.0)
            pressure = cv2.GaussianBlur(pressure, (5, 5), 0)
            pressure = np.clip(pressure, 0.0, 1.0)
        else:
            pressure = np.zeros((height, width), dtype=np.float32)

        displacement_x = force_field[..., 0]
        displacement_y = force_field[..., 1]
        lateral_magnitude = np.sqrt(displacement_x**2 + displacement_y**2)
        max_lateral = float(np.percentile(lateral_magnitude, 99.0))
        if max_lateral <= 1.0e-6:
            max_lateral = float(np.max(lateral_magnitude))
        if max_lateral > 1.0e-6:
            max_displacement_px = 8.0
            displacement_x = displacement_x / max_lateral * max_displacement_px * pressure
            displacement_y = displacement_y / max_lateral * max_displacement_px * pressure
        else:
            displacement_x = np.zeros((height, width), dtype=np.float32)
            displacement_y = np.zeros((height, width), dtype=np.float32)

        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        value = cv2.remap(
            self._get_openworldtactile_value_texture(height, width),
            grid_x - displacement_x.astype(np.float32),
            grid_y - displacement_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

        hsv = np.empty((height, width, 3), dtype=np.uint8)
        hsv[..., 0] = np.clip(120.0 * pressure, 0.0, 179.0).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = value
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _draw_tactile_force_field(self, force_field: np.ndarray) -> np.ndarray:
        force_x = force_field[..., 0]
        force_y = force_field[..., 1]
        magnitude = force_field[..., 2]

        height, width = magnitude.shape
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        step = max(8, min(height, width) // 24)

        for y in range(step // 2, height, step):
            for x in range(step // 2, width, step):
                cv2.circle(frame, (x, y), 1, (0, 180, 60), -1)

        max_magnitude = float(np.max(magnitude))
        if max_magnitude <= 1.0e-6:
            return frame

        magnitude = np.sqrt(force_x**2 + force_y**2)
        max_lateral = float(np.max(magnitude))
        if max_lateral <= 1.0e-6:
            return frame

        arrow_scale = 0.7 * step / max_lateral
        threshold = max_lateral * 0.05
        for y in range(step // 2, height, step):
            for x in range(step // 2, width, step):
                if magnitude[y, x] <= threshold:
                    continue
                end_x = int(np.clip(x + force_x[y, x] * arrow_scale, 0, width - 1))
                end_y = int(np.clip(y + force_y[y, x] * arrow_scale, 0, height - 1))
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), (255, 40, 40), 1, tipLength=0.25)

        return frame

    def _draw_openworldtactile_sensor_force_field(self, force_field: np.ndarray) -> np.ndarray:
        force_field = np.nan_to_num(force_field.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        force_x = force_field[..., 0]
        force_y = force_field[..., 1]
        pressure = np.clip(force_field[..., 2], 0.0, None)

        height, width = pressure.shape
        max_pressure = float(np.percentile(pressure, 99.0))
        if max_pressure <= 1.0e-6:
            max_pressure = float(np.max(pressure))

        if max_pressure > 1.0e-6:
            pressure_norm = np.clip(pressure / max_pressure, 0.0, 1.0)
            heat = (pressure_norm * 255.0).astype(np.uint8)
            frame = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        lateral_magnitude = np.sqrt(force_x**2 + force_y**2)
        max_lateral = float(np.percentile(lateral_magnitude, 99.0))
        if max_lateral <= 1.0e-6:
            max_lateral = float(np.max(lateral_magnitude))

        step = max(8, min(height, width) // 24)
        for y in range(step // 2, height, step):
            for x in range(step // 2, width, step):
                cv2.circle(frame, (x, y), 1, (0, 220, 120), -1)

        if max_lateral <= 1.0e-6:
            return frame

        arrow_scale = 0.7 * step / max_lateral
        threshold = max_lateral * 0.05
        for y in range(step // 2, height, step):
            for x in range(step // 2, width, step):
                if lateral_magnitude[y, x] <= threshold:
                    continue
                end_x = int(np.clip(x + force_x[y, x] * arrow_scale, 0, width - 1))
                end_y = int(np.clip(y + force_y[y, x] * arrow_scale, 0, height - 1))
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), (255, 255, 255), 1, tipLength=0.25)

        return frame

    def _show_height_map_inside_gui(self, index):
        plt.close()
        height_map = self._data.output["height_map"][index].cpu().numpy()
        np.save("height_map.npy", height_map)

        X = np.arange(0, height_map.shape[0])
        Y = np.arange(0, height_map.shape[1])
        X, Y = np.meshgrid(X, Y)
        Z = height_map
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, Z.T)
        # plt.show()
        print("saving height_map img")
        plt.savefig(f"height_map{index}.png")

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._prim_view = None

        self.camera._invalidate_initialize_callback(event)
        self.camera.__del__()

        if hasattr(self, "_windows"):
            self._windows = None
            self._img_providers = None
