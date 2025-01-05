import math

import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer
from modules.basic_utils import load_json
import torch.nn.functional as F
from model.clip_model import load_clip


def load_scene(path):
    content = load_json(path)
    content = torch.Tensor(content['features'])
    return content


class Scene(nn.Module):
    def __init__(self, config: Config):
        super(Scene, self).__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(config.transformer_dropout)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, feature, scene):
        feature = feature.permute(0, 2, 1)
        feature = self.layer_norm1(feature)
        scene = self.layer_norm2(scene)
        q = self.q_proj(feature)
        k = self.k_proj(scene)
        v = self.v_proj(scene)

        if self.config.scene_type =='shared':
            k = k.t()
        else:
            k = k.permute(0, 2, 1)

        attention = q @ k
        attention = attention / math.sqrt(self.embed_dim)
        attention = F.softmax(attention, dim=1)
        out = attention @ v
        line_out = self.out_proj(out)
        out = out + self.dropout(line_out)
        return out


class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config

        self.clip = load_clip(config.clip_arch)

        # config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)

        self.shared = load_scene(config.scene_path).cuda()
        self.scene = Scene(config)
        self.tw = nn.Parameter(torch.Tensor([0.3, 0.7]))
        self.pool = nn.AdaptiveAvgPool1d(1)
        # self.threshold = nn.Parameter(torch.Tensor([config.threshold]))

    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)

        text_features = self.clip.encode_text(text_data)
        video_features = self.clip.encode_image(video_data)

        vf = video_features.reshape(batch_size, self.config.num_frames, -1).permute(0, 2, 1)
        if self.config.scene_type == 'shared':
            scene = self.shared
        elif self.config.scene_type == 'average':
            scene = self.pool(vf).squeeze()
            scene = scene.reshape(batch_size, 1, -1)
        elif self.config.scene_type == 'single':
            scene = self.pool(vf).squeeze()
            x = scene - vf[:, :, self.config.select_frame]
            # zeros = torch.zeros_like(scene)
            # t = torch.exp(self.threshold)
            # scene = torch.where(x > t, zeros, scene)
            # scene = torch.where(x < -t, zeros, scene)
            # scene = scene.reshape(batch_size, 1, -1)
            threshold_effect = 0.5 * (torch.tanh(-30 * (x - self.threshold)) + 1)
            scene = x * threshold_effect
            
            scene = scene.reshape(batch_size, 1, -1)


        video_scene = self.scene(vf, scene)
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)
        tw1 = torch.exp(self.tw[0]) / torch.sum(torch.exp(self.tw))
        tw2 = torch.exp(self.tw[1]) / torch.sum(torch.exp(self.tw))

        video_features = tw1 * video_scene + tw2 * video_features

        video_features_pooled = self.pool_frames(text_features, video_features)

        if return_all_frames:
            return text_features, video_features, video_features_pooled

        return text_features, video_features_pooled
