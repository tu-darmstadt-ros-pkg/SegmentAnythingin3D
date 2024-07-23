# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Segment Anything in 3D Pipeline"""

from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText
from nerfstudio.utils import profiler

from sa3d.sa3d_datamanager import SA3DDataManagerConfig
from sa3d.sa3d import SA3DModelConfig
from sa3d.self_prompting.sam3d import SAM3DConfig, SAM3D

@dataclass
class SA3DPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""
    _target: Type = field(default_factory=lambda: SA3DPipeline)
    """target class to instantiate"""
    datamanager: SA3DDataManagerConfig = SA3DDataManagerConfig()
    # datamanager: SA3DDataManagerConfig = SA3DDataManagerConfig()
    """specifies the datamanager config"""
    model: SA3DModelConfig = SA3DModelConfig()
    """specifies the model config"""
    network: SAM3DConfig = SAM3DConfig()
    """specifies the segmentation model SAM3D config"""
    text_prompt: str = ""
    """text prompt"""

    

class SA3DPipeline(VanillaPipeline):
    """SA3D pipeline"""

    config: SA3DPipelineConfig

    def __init__(
        self,
        config: SA3DPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        self.init_prompt = None
        self.sam: SAM3D = config.network.setup(device=device)
        # viewer elements
        self.text_prompt_box = ViewerText(name="Text Prompt", default_value=self.config.text_prompt, cb_hook=self.text_prompt_callback)

    def text_prompt_callback(self, handle: ViewerText) -> None:
        """Callback for text prompt box, change prompt in config"""
        self.config.text_prompt = handle.value

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs mask inverse.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """

        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)
        if step != 0:
            prompt = None
        else:
            if self.config.text_prompt:
                prompt = self.config.text_prompt
            else:
                prompt = self.img2prompt(model_outputs['rgb'].cpu().numpy())
            self.init_prompt = prompt

        sam_outputs, loss_dict, metrics_dict = self.sam.get_outputs(model_outputs, init_prompt=prompt)
        model_outputs.update(sam_outputs)
        return model_outputs, loss_dict, metrics_dict
    
    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state, strict=False)

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
    
    def img2prompt(self, image: np.ndarray):
        """
        Display an image and let the user select points on it.
        Parameters:
        image (np.ndarray): An H x W x 3 array representing the image.
        Returns:
        np.ndarray: An n x 2 array representing the coordinates of the selected points.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be an H x W x 3 array.")

        plt.imshow(image)
        plt.title("Click on the image to select points, then press Enter")

        # Let user select points
        points = plt.ginput(n=-1, timeout=0)  # n=-1 means unlimited number of points, timeout=0 means wait indefinitely

        plt.close()

        # Convert list of tuples to np.ndarray
        points_array = np.array(points)

        return points_array
