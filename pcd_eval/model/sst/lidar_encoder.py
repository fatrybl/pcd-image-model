from mmengine import Config
import torch
from torch import nn

from mmdet3d.registry import MODELS

from pcd_eval.model.sst.attention_pool import AttentionPool2d
from pcd_eval.model.sst.sst_encoder_config_v2 import model as sst_model_conf
import pcd_eval.model.sst.sstv2_base
import pcd_eval.model.sst.sst_input_layer_v2
import pcd_eval.model.sst.objects


def build_sst(config_path):
    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    model.init_weights()
    return model


class LidarEncoderSST(nn.Module):
    def __init__(self, sst_config_path, clip_embedding_dim=512):
        super().__init__()
        self._sst = build_sst(sst_config_path)
        self._pooler = AttentionPool2d(
            spacial_dim=sst_model_conf["backbone"]["output_shape"][0],
            embed_dim=clip_embedding_dim,
            num_heads=8,
            input_dim=sst_model_conf["backbone"]["conv_out_channel"],
        )

    def forward(self, point_cloud, no_pooling=False, return_attention=False):
        voxels_with_coords = self._sst.data_preprocessor.voxelize(point_cloud, data_samples=[])

        input_dict = {'voxels': voxels_with_coords}

        lidar_features = self._sst.extract_feat(input_dict)[0]  # bs, d, h, w
        pooled_feature, attn_weights = self._pooler(lidar_features, no_pooling, return_attention)
        return pooled_feature, attn_weights


if __name__ == "__main__":
    import torch
    from pathlib import Path

    cfg_path = Path(__file__).parent / "sst_encoder_config_v2.py"
    model = LidarEncoderSST(cfg_path)

    model.to("cuda")
    points = [torch.rand(100, 4).cuda() for _ in range(16)]

    out = model(points)