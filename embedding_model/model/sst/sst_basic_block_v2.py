import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from mmcv.cnn import build_norm_layer
from embedding_model.model.sst.sst_ops import flat2window_v2, window2flat_v2

class WindowAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout, batch_first=False, layer_id=None, layer_cfg=dict()):
        super().__init__()
        self.nhead = nhead

        if layer_cfg.get('cosine', False):
            from .cosine_msa import CosineMultiheadAttention
            tau_min = layer_cfg.get('tau_min', 0.01)
            self.self_attn = CosineMultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=False, tau_min=tau_min,
                cosine=True,
                non_shared_tau=layer_cfg.get('non_shared_tau', False)
            )
        elif layer_cfg.get('linear', False):
            raise NotImplementedError
            from mmdet3d.models.sst.linear_msa import LinearMultiheadAttention
            self.self_attn = LinearMultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=False
            )
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.exe_counter = 0

        self.layer_id = layer_id

    def forward(self, feat_2d, pos_dict, ind_dict, key_padding_dict):
        '''
        Args:

        Out:
            shifted_feat_dict: the same type as window_feat_dict
        '''

        out_feat_dict = {}

        feat_3d_dict = flat2window_v2(feat_2d, ind_dict)

        for name in feat_3d_dict:
            #  [n, num_token, embed_dim]
            pos = pos_dict[name]

            feat_3d = feat_3d_dict[name]
            feat_3d = feat_3d.permute(1, 0, 2)

            v = feat_3d

            if pos is not None:
                pos = pos.permute(1, 0, 2)
                assert pos.shape == feat_3d.shape, f'pos_shape: {pos.shape}, feat_shape:{feat_3d.shape}'
                q = k = feat_3d + pos
            else:
                q = k = feat_3d

            key_padding_mask = key_padding_dict[name]
            out_feat_3d, attn_map = self.self_attn(q, k, value=v, key_padding_mask=key_padding_mask)
            out_feat_dict[name] = out_feat_3d.permute(1, 0, 2)

        results = window2flat_v2(out_feat_dict, ind_dict)
        
        return results

class EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=False, layer_id=None, mlp_dropout=0, layer_cfg=dict()):
        super().__init__()
        assert not batch_first, 'Current version of PyTorch does not support batch_first in MultiheadAttention. After upgrading pytorch, do not forget to check the layout of MLP and layer norm to enable batch_first option.'
        self.batch_first = batch_first
        self.win_attn = WindowAttention(d_model, nhead, dropout, layer_id=layer_id, layer_cfg=layer_cfg)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        use_bn = layer_cfg.get('use_bn', False)
        if use_bn:
            self.norm1 = build_norm_layer(dict(type='naiveSyncBN1d', momentum=layer_cfg.get('mom', 0.1)), d_model)[1]
            self.norm2 = build_norm_layer(dict(type='naiveSyncBN1d', momentum=layer_cfg.get('mom', 0.1)), d_model)[1]
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(mlp_dropout)
        self.dropout2 = nn.Dropout(mlp_dropout)

        self.activation = _get_activation_fn(activation)
        self.post_norm = layer_cfg.get('post_norm', True)
        self.fp16_enabled=False

    def forward(
        self,
        src,
        pos_dict,
        ind_dict,
        key_padding_mask_dict,
        ):
        if self.post_norm:
            src2 = self.win_attn(src, pos_dict, ind_dict, key_padding_mask_dict) #[N, d_model]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        else:
            src2 = self.norm1(src)
            src2 = self.win_attn(src2, pos_dict, ind_dict, key_padding_mask_dict) #[N, d_model]
            src = src + self.dropout1(src2)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        return src

class BasicShiftBlockV2(nn.Module):
    ''' Consist of two encoder layer, shift and shift back.
    '''

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=False, block_id=-100, layer_cfg=dict()):
        super().__init__()

        encoder_1 = EncoderLayer(d_model, nhead, dim_feedforward, dropout,
            activation, batch_first, layer_id=block_id * 2 + 0, layer_cfg=layer_cfg)
        encoder_2 = EncoderLayer(d_model, nhead, dim_feedforward, dropout,
            activation, batch_first, layer_id=block_id * 2 + 1, layer_cfg=layer_cfg)
        # BasicShiftBlock(d_model[i], nhead[i], dim_feedforward[i], dropout, activation, batch_first=False)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])

    def forward(
        self,
        src,
        pos_dict_list,
        ind_dict_list,
        key_mask_dict_list,
        using_checkpoint=False,
        ):
        num_shifts = len(pos_dict_list)
        assert num_shifts in (1, 2)

        output = src
        for i in range(2):

            this_id = i % num_shifts
            pos_dict = pos_dict_list[this_id]
            ind_dict = ind_dict_list[this_id]
            key_mask_dict = key_mask_dict_list[this_id]

            layer = self.encoder_list[i]
            if using_checkpoint and self.training:
                output = checkpoint(layer, output, pos_dict, ind_dict, key_mask_dict)
            else:
                output = layer(output, pos_dict, ind_dict, key_mask_dict)

        return output
    

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.functional.relu
    if activation == "gelu":
        return nn.functional.gelu
    if activation == "glu":
        return nn.functional.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")