import numpy as np
import cv2
from PIL import Image
import nndeploy.dag

# Canny edge detection node with numpy input/output
class CannyNumpy(nndeploy.dag.Node):
    """
    Canny edge detection node (numpy input/output)

    Input:
        - np.ndarray, image in BGR format

    Output:
        - np.ndarray, single-channel or multi-channel binary edge map
    """
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.image.CannyNumpy")
        super().set_desc("Canny edge detection (numpy input/output)")
        self.set_input_type(np.ndarray)
        self.set_output_type(np.ndarray)
        self.low_threshold = 100
        self.high_threshold = 200
        # New parameter: number of output channels, 1 for single-channel, 3 for RGB, 4 for RGBA
        self.return_num_channel = 3

    def run(self):
        input_edge = self.get_input(0)
        image = input_edge.get(self)
        if image is None:
            return nndeploy.base.Status.err("Input cannot be None")
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        # Check if multi-channel output is needed
        if self.return_num_channel > 1:
            edges = np.stack([edges]*self.return_num_channel, axis=2)
        output_edge = self.get_output(0)
        output_edge.set(edges)
        return nndeploy.base.Status.ok()

    def serialize(self):
        import json
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["low_threshold"] = self.low_threshold
        json_obj["high_threshold"] = self.high_threshold
        json_obj["return_num_channel"] = self.return_num_channel
        return json.dumps(json_obj)

    def deserialize(self, target: str):
        import json
        json_obj = json.loads(target)
        self.low_threshold = json_obj.get("low_threshold", 100)
        self.high_threshold = json_obj.get("high_threshold", 200)
        self.return_num_channel = json_obj.get("return_num_channel", 3)
        return super().deserialize(target)

class CannyNumpyCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = CannyNumpy(name, inputs, outputs)
        return self.node

canny_edge_numpy_creator = CannyNumpyCreator()
nndeploy.dag.register_node("nndeploy.image.CannyNumpy", canny_edge_numpy_creator)

# Canny edge detection node with PIL input/output
class CannyPIL(nndeploy.dag.Node):
    """
    Canny edge detection node (PIL input/output)

    Input:
        - PIL.Image.Image, image in RGB format

    Output:
        - PIL.Image.Image, single-channel or multi-channel binary edge map
    """
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.image.CannyPIL")
        super().set_desc("Canny edge detection (PIL input/output)")
        self.set_input_type(Image.Image)
        self.set_output_type(Image.Image)
        self.low_threshold = 100
        self.high_threshold = 200
        # New parameter: number of output channels, 1 for single-channel, 3 for RGB, 4 for RGBA
        self.return_num_channel = 3

    def run(self):
        input_edge = self.get_input(0)
        pil_image = input_edge.get(self)
        if pil_image is None:
            return nndeploy.base.Status.err("Input cannot be None")
        if not isinstance(pil_image, Image.Image):
            return nndeploy.base.Status.err("Input must be of type PIL.Image.Image")
        image = np.array(pil_image)
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        # Check if multi-channel output is needed
        if self.return_num_channel == 3:
            edges = np.stack([edges]*3, axis=2)
            pil_edges = Image.fromarray(edges, mode="RGB")
        elif self.return_num_channel == 4:
            edges = np.stack([edges]*4, axis=2)
            pil_edges = Image.fromarray(edges, mode="RGBA")
        else:
            pil_edges = Image.fromarray(edges)
        output_edge = self.get_output(0)
        output_edge.set(pil_edges)
        return nndeploy.base.Status.ok()

    def serialize(self):
        import json
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["low_threshold"] = self.low_threshold
        json_obj["high_threshold"] = self.high_threshold
        json_obj["return_num_channel"] = self.return_num_channel
        return json.dumps(json_obj)

    def deserialize(self, target: str):
        import json
        json_obj = json.loads(target)
        self.low_threshold = json_obj.get("low_threshold", 100)
        self.high_threshold = json_obj.get("high_threshold", 200)
        self.return_num_channel = json_obj.get("return_num_channel", 3)
        return super().deserialize(target)

class CannyPILCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = CannyPIL(name, inputs, outputs)
        return self.node

canny_edge_pil_creator = CannyPILCreator()
nndeploy.dag.register_node("nndeploy.image.CannyPIL", canny_edge_pil_creator)
