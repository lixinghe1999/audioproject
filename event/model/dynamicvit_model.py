from model.vit_model import AudioTransformerDiffPruning, VisionTransformerDiffPruning, PredictorLG,\
    batch_index_select, VisionTransformerTeacher
from torch.cuda.amp import autocast
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class AVnet_Dynamic(nn.Module):
    def __init__(self, scale='base', distill=False, num_classes=309, \
                 pruning_loc=[3, 6, 9], token_ratio=[0.7, 0.7**2, 0.7**3], pretrained=True):
        super(AVnet_Dynamic, self).__init__()
        if scale == 'base':
            config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                          pruning_loc=pruning_loc, token_ratio=token_ratio)
            embed_dim = 768
        else:
            config = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                                pruning_loc=pruning_loc, token_ratio=token_ratio)
            embed_dim = 384
        self.audio = AudioTransformerDiffPruning(config, imagenet_pretrain=pretrained)
        self.image = VisionTransformerDiffPruning(**config)
        if pretrained:
            self.image.load_state_dict(torch.load('assets/deit_base_patch16_224.pth')['model'], strict=False)

        self.num_patches = self.audio.num_patches + 14 * 14

        self.head = nn.Sequential(nn.LayerNorm(embed_dim * 2), nn.Linear(embed_dim * 2, 309))
        if len(pruning_loc) > 0:
            predictor_list = [PredictorLG(embed_dim) for _ in range(len(pruning_loc))]
            self.score_predictor = nn.ModuleList(predictor_list)

        self.distill = distill

        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio
    def output(self, audio, image):
        audio = self.audio.norm(audio)
        image = self.image.norm(image)
        features = torch.cat([audio[:, 1:], image[:, 1:]], dim=1)
        x = torch.cat([audio[:, 0], image[:, 0]], dim=1)
        x = torch.flatten(x, start_dim=1)
        x = self.head(x)
        return x, features
    @autocast()
    def forward(self, audio, image):
        B, audio = self.audio.preprocess(audio.unsqueeze(1))
        B, image = self.image.preprocess(image)

        p_count = 0
        out_pred_prob = []
        early_output = []
        prev_decision = torch.ones(B, self.num_patches, 1, dtype=audio.dtype, device=audio.device)
        policy = torch.ones(B, self.num_patches + 2, 1, dtype=audio.dtype, device=audio.device)
        t_stamp = []
        t_start = time.time()
        for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
            if i in self.pruning_loc:
                spatial_x = torch.cat([audio[:, 1:], image[:, 1:]], dim=1)
                token_len_audio = audio.shape[1] - 1
                pred_score = self.score_predictor[p_count](spatial_x, prev_decision).reshape(B, -1, 2)
                if self.training:
                    hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                    out_pred_prob.append(hard_keep_decision.reshape(B, self.num_patches))
                    decision_audio = hard_keep_decision[:, :token_len_audio]
                    decision_image = hard_keep_decision[:, token_len_audio:]

                    cls_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                    policy_a = torch.cat([cls_policy, decision_audio], dim=1)
                    audio = blk_a(audio, policy=policy_a)

                    cls_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                    policy_i = torch.cat([cls_policy, decision_image], dim=1)
                    image = blk_i(image, policy=policy_i)
                    prev_decision = hard_keep_decision
                    policy = torch.cat([policy_a, policy_i], dim=1)
                    early_output.append(self.output(audio, image)[0])
                else:
                    score = pred_score[:, :, 0]
                    num_keep_node = int(self.num_patches * self.token_ratio[p_count])
                    keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]
                    prev_decision = batch_index_select(prev_decision, keep_policy)
                    keep_audio = keep_policy[keep_policy < token_len_audio].unsqueeze(0)
                    keep_image = keep_policy[keep_policy >= token_len_audio].unsqueeze(0) - token_len_audio

                    cls_policy = torch.zeros(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
                    now_policy = torch.cat([cls_policy, keep_audio + 1], dim=1)
                    audio = batch_index_select(audio, now_policy)
                    audio = blk_a(audio)

                    cls_policy = torch.zeros(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
                    now_policy = torch.cat([cls_policy, keep_image + 1], dim=1)
                    image = batch_index_select(image, now_policy)
                    image = blk_i(image)

                p_count += 1
            else:
                if self.training:
                    policy_a = policy[:, :self.audio.num_patches+1]
                    policy_i = policy[:, self.audio.num_patches + 1:]
                    audio = blk_a(audio, policy=policy_a)
                    image = blk_i(image, policy=policy_i)
                else:
                    audio = blk_a(audio)
                    image = blk_i(image)
            t_stamp.append(time.time() - t_start)
        x, features = self.output(audio, image)
        if self.training:
            if self.distill:
                return x, features, prev_decision.detach(), out_pred_prob, early_output
            else:
                return x, out_pred_prob
        else:
            if self.distill:
                return x, features
            else:
                ratio = audio.shape[1] / (audio.shape[1] + image.shape[1])
                return x, t_stamp, ratio
if __name__ == "__main__":
    device = 'cpu'
    base_rate = 0.5
    pruning_loc = [3, 6, 9]
    token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]
    config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                  pruning_loc=pruning_loc, token_ratio=token_ratio)

    model_image = VisionTransformerDiffPruning(**config).to(device)

    model_audio = AudioTransformerDiffPruning(config, imagenet_pretrain=True).to(device)

    model_teacher = AVnet_Dynamic(pruning_loc=()).to(device)

    multi_modal_model = AVnet_Dynamic().to(device)

    model_image.eval()
    model_audio.eval()
    model_teacher.eval()
    multi_modal_model.eval()

    audio = torch.zeros(1, 384, 128).to(device)
    image = torch.zeros(1, 3, 224, 224).to(device)

    num_iterations = 10
    with torch.no_grad():
        for i in range(num_iterations):
            if i == 1:
                t_start = time.time()
            multi_modal_model(audio, image)
        print((time.time() - t_start) / (num_iterations - 1))

    with torch.no_grad():
        for i in range(num_iterations):
            if i == 1:
                t_start = time.time()
            model_image(image)
        print((time.time() - t_start) / (num_iterations - 1))

    with torch.no_grad():
        for i in range(num_iterations):
            if i == 1:
                t_start = time.time()
            model_audio(audio)
        print((time.time() - t_start) / (num_iterations - 1))

    with torch.no_grad():
        for i in range(num_iterations):
            if i == 1:
                t_start = time.time()
            model_teacher(audio, image)
        print((time.time() - t_start) / (num_iterations-1))
