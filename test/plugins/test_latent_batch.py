from importlib.util import find_spec
import unittest


class LatentBatchTest(unittest.TestCase):
    @unittest.skipUnless(find_spec("torch"), "PyTorch disabled")
    def test_dynamic_inputs_concatenate_in_order(self):
        import torch
        import nndeploy.base as base
        import nndeploy.dag as dag
        from nndeploy.diffusion.latent import LatentBatch

        inputs = [dag.Edge(name) for name in ("first", "second", "third")]
        output = dag.Edge("batch")
        node = LatentBatch("batch_latents", inputs, [output])
        node.set_device_type(base.DeviceType(base.DeviceTypeCode.cpu))
        node.dtype = "float32"
        values = [torch.tensor([[[[x]]]], dtype=torch.float16) for x in (3, 1, 2)]
        for edge, value in zip(inputs, values):
            edge.set(value)
        self.assertTrue(node.run())
        torch.testing.assert_close(output.get(node), torch.cat(values).float())
