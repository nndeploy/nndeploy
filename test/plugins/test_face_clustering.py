from importlib.util import find_spec
import unittest

import numpy as np


class FaceClustering(unittest.TestCase):
    @unittest.skipUnless(find_spec("sklearn") and find_spec("cv2"), "face clustering dependencies disabled")
    def test_fewer_embeddings_than_requested_clusters(self):
        from nndeploy.face.deep_live_cam import find_cluster_centroids

        self.assertEqual(len(find_cluster_centroids([])), 0)
        np.testing.assert_allclose(find_cluster_centroids([[2, 4]]), [[2, 4]])
        samples = [[0, 1], [2, 3]]
        centers = find_cluster_centroids(samples)
        np.testing.assert_allclose(sorted(centers.tolist()), samples)
        np.testing.assert_allclose(find_cluster_centroids(samples, max_k=1), [[1, 2]])
