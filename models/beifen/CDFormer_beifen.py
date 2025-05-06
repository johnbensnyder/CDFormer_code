import copy
import math
import torch
import random
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, is_dist_avail_and_initialized, inverse_sigmoid,
                       autocast)
"""
手动切换dino还是resnet
"""
# from .backbone import build_backbone
from .models.dino_backbone import build_backbone
from .models.matcher import build_matcher
from .models.deformable_transformer import build_deforamble_transformer
from .models.deformable_transformer import DeformableTransformerDecoderLayer, DeformableTransformerDecoder
from .models.position_encoding import TaskPositionalEncoding, QueryEncoding


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0)
        self.scale_factor = 10

    def forward(self, x):
        # 如果输入有批次维度 (batch_size, num_samples, num_features)
        if x.dim() == 3:
            batch_size, num_samples, num_features = x.shape
            output = []

            for i in range(batch_size):
                # 对每个样本 (num_samples, num_features) 进行处理
                x_single = x[i]  # 取出单个样本，形状为 (num_samples, num_features)
                x_norm = torch.norm(x_single, p=2, dim=1).unsqueeze(1).expand_as(x)
                x_normalized = x.div(x_norm + 0.00001)
                if not self.class_wise_learnable_norm:
                    L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
                    self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
                cos_dist = self.L(x_normalized)
                scores = self.scale_factor * cos_dist
                output.append(scores)
            # 将每个样本的结果堆叠回一起，形成批次的输出
            output = torch.stack(output)
            return output

        # 如果输入没有批次维度
        else:
            x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
            x_normalized = x.div(x_norm + 0.00001)
            if not self.class_wise_learnable_norm:
                L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
                self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
            cos_dist = self.L(x_normalized)
            scores = self.scale_factor * cos_dist
            return scores

        # x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        # x_normalized = x.div(x_norm + 0.00001)
        # if not self.class_wise_learnable_norm:
        #     L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
        #     self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        # cos_dist = self.L(x_normalized)
        # scores = self.scale_factor * cos_dist
        # return scores


class CDFormer(nn.Module):
    def __init__(self, args, backbone, transformer, num_classes, num_queries, num_feature_levels, aux_loss=True, with_box_refine=False):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = args.hidden_dim
        self.num_feature_levels = num_feature_levels

        self.transformer = transformer
        self.task_positional_encoding = TaskPositionalEncoding(self.hidden_dim, dropout=0., max_len=self.args.episode_size)
        self.class_embed = nn.Linear(self.hidden_dim, self.args.episode_size)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        if args.category_codes_cls_loss:
            if num_feature_levels == 1:
                if args.fewshot_finetune == True and args.feature_space_enable == True:
                    if args.base_dataset_name == 'coco':
                        listdinear_out_dim = self.num_classes + 80
                    elif args.base_dataset_name == 'voc':
                        listdinear_out_dim = self.num_classes + 20
                else:
                    listdinear_out_dim = self.num_classes
                self.category_codes_cls = distLinear(self.hidden_dim, listdinear_out_dim)
                if args.multi_category_loss and args.all_distLinear:
                    self.category_codes_cls1 = distLinear(self.hidden_dim, listdinear_out_dim)
                    self.category_codes_cls2 = distLinear(self.hidden_dim, listdinear_out_dim)
                    self.category_codes_cls3 = distLinear(self.hidden_dim, listdinear_out_dim)
                    self.category_codes_cls4 = distLinear(self.hidden_dim, listdinear_out_dim)
                    self.category_codes_cls5 = distLinear(self.hidden_dim, listdinear_out_dim)
                # self.category_codes_cls = distLinear(self.hidden_dim, self.num_classes)
                # if args.multi_category_loss and args.all_distLinear:
                #     self.category_codes_cls1 = distLinear(self.hidden_dim, self.num_classes)
                #     self.category_codes_cls2 = distLinear(self.hidden_dim, self.num_classes)
                #     self.category_codes_cls3 = distLinear(self.hidden_dim, self.num_classes)
                #     self.category_codes_cls4 = distLinear(self.hidden_dim, self.num_classes)
                #     self.category_codes_cls5 = distLinear(self.hidden_dim, self.num_classes)
            elif num_feature_levels > 1:
                category_codes_cls_list = []
                for _ in range(self.num_feature_levels):
                    category_codes_cls_list.append(distLinear(self.hidden_dim, self.num_classes))
                self.category_codes_cls = nn.ModuleList(category_codes_cls_list)
            else:
                raise RuntimeError

        # self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim * 2)
        queryencoding = QueryEncoding(self.hidden_dim, dropout=0., max_len=self.num_queries)
        qe = queryencoding()
        self.query_embed = torch.cat([qe, qe], dim=1)

        if self.num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])
        self.backbone = backbone
        self.with_box_refine = with_box_refine
        self.aux_loss = aux_loss

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(1) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        assert args.hidden_dim == self.hidden_dim
        decoder_layer = DeformableTransformerDecoderLayer(args.hidden_dim,
                                                          args.dim_feedforward,
                                                          args.dropout,
                                                          'relu',
                                                          args.num_feature_levels,
                                                          args.nheads,
                                                          args.dec_n_points)

        self.meta_decoder = DeformableTransformerDecoder(decoder_layer,
                                                         args.dec_layers,
                                                         return_intermediate=True)

        num_pred = self.meta_decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.meta_decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])

    def forward(self, samples, targets=None, supp_samples=None, supp_class_ids=None, supp_targets=None, category_codes=None):

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        batchsize = samples.tensors.shape[0]
        # logging.info(f"Rank {torch.distributed.get_rank()}, Batch size: {batchsize}")
        device = samples.tensors.device

        # During training, category_codes are generated from sampled (supp_samples, supp_class_ids, supp_targets)
        if self.training:
            assert supp_samples is not None
            assert supp_class_ids is not None
            assert supp_targets is not None
            # During training stage: we don't have to cover all categories, so there is only 1 episode
            num_support = supp_class_ids.shape[0]
            support_batchsize = self.args.episode_size
            assert num_support == (self.args.episode_size * self.args.episode_num)
            num_episode = self.args.episode_num
            category_codes = self.compute_category_codes(supp_samples, supp_targets, supp_class_ids)
        # During inference, category_codes should be provided and ready to use for all activated categories
        else:
            assert category_codes is not None
            assert supp_class_ids is not None
            # During inference stage: there are multiple episodes to cover all categories, including both base and novel
            num_support = supp_class_ids.shape[0]
            support_batchsize = self.args.episode_size
            num_episode = math.ceil(num_support / support_batchsize)

        """
        如果用resnet这里不用太关心
        但是用dinov2,这个backbone就是由dino+position_embedding组成的(后者有v2不可训练的sine和v3可训练的位置编码)
        我们需要只冻结dino:self.backbone的DinoVisionTransformer = self.backbone[0]
        这里相当于三重保险:初始化时置为false、with torch.no_grad、eval模式
        """
        # for param in self.backbone.parameters():
        #     if param.requires_grad:
        #         print("Some parameters are not frozen.")
        #         break
        #     else:
        #         print("All parameters are frozen.")
        if self.args.backbone == 'dinov2':
            if not self.args.VPT_enable:
                # 注释掉，这里不保存计算图的操作放在joiner的forward里
                # with torch.no_grad():
                if self.backbone[0].training: self.backbone[0].eval()
                # 启用自动混合精度（Automatic Mixed Precision，AMP）。目的：在保持模型精度的同时提高计算效率，尤其是在支持的GPU上。效果：某些操作会自动使用较低的精度（如float16）来加速计算。
                with autocast(enabled=True): features, pos = self.backbone(samples)
            else:
                if self.backbone[0].training: self.backbone[0].eval()
                self.backbone[0].prompt_dropout.train()
                with autocast(enabled=True): features, pos = self.backbone(samples, self.args.VPT_enable)
        else:
            features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = self.query_embed.to(device)

        # To store predictions for each episode
        meta_outputs_classes = []
        meta_outputs_coords = []
        meta_support_class_ids = []

        count_last = 0
        for i in range(num_episode):

            if self.num_feature_levels == 1:
                if (support_batchsize * (i + 1)) <= num_support:
                    # cc = [c[(support_batchsize * i): (support_batchsize * (i + 1)), :].unsqueeze(0).expand(batchsize, -1, -1) for c in category_codes]
                    episode_class_ids = supp_class_ids[(support_batchsize * i): (support_batchsize * (i + 1))]

                    cc2 = []

                    for c in category_codes:
                        count_now = count_last
                        cc1_tmp = []
                        for index, tmp in enumerate(supp_class_ids[i*support_batchsize: (i+1)*support_batchsize]):
                            if tmp != 100:
                                cc1_tmp.append(c[count_now])
                                count_now = count_now + 1
                        cc1 = torch.stack(cc1_tmp, dim=0)
                        cc1 = cc1.unsqueeze(0).expand(batchsize, -1, -1)
                        cc2.append(cc1)
                    
                    for index, tmp in enumerate(supp_class_ids[i*support_batchsize: (i+1)*support_batchsize]):
                            if tmp != 100:
                                count_last += 1
                    
                    # 比较验证,debug用
                    # if len(cc) != len(cc2):
                    #     are_equal = False
                    # else:
                    #     are_equal = all(torch.equal(t1, t2) for t1, t2 in zip(cc, cc2))
                    # print(are_equal)
                # 注意：代码这个else【应该】是会报错的，因为tsp仍为[BZ,5,256]
                # 比如support只有三个了，那么做CAM时，q为[4,441,256],KV为[4,3,256],而tsp为[4,5,256]就会有问题
                else:
                    cc = [c[-support_batchsize:, :].unsqueeze(0).expand(batchsize, -1, -1) for c in category_codes]
                    episode_class_ids = supp_class_ids[-support_batchsize:]
            elif self.num_feature_levels == 4:
                raise NotImplementedError
            else:
                raise NotImplementedError

            """
            expand(batchsize, -1, -1) 扩展张量的形状为 (batchsize, self.args.episode_size, self.hidden_dim)，相当于复制了 batchsize 份。
            """
            _, init_reference, _, encoder_outputs = \
                self.transformer(srcs, masks, pos, query_embeds, cc2,
                                 self.task_positional_encoding(torch.zeros(self.args.episode_size, self.hidden_dim, device=device)).unsqueeze(0).expand(batchsize, -1, -1), 
                                 episode_class_ids)

            (memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten, tgt) = encoder_outputs

            # Category-agnostic transformer decoder
            hs, inter_references = self.meta_decoder(
                tgt,
                init_reference,
                memory,
                spatial_shapes,
                level_start_index,
                valid_ratios,
                query_embed,
                mask_flatten,
            )

            # Final FFN to predict confidence scores and boxes coordinates
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference.reshape(batchsize, self.num_queries, 2)
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.class_embed[lvl](hs[lvl])
                tmp = self.bbox_embed[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class.view(batchsize, self.num_queries, self.args.episode_size))
                outputs_coords.append(outputs_coord.view(batchsize, self.num_queries, 4))

            meta_outputs_classes.append(torch.stack(outputs_classes))
            meta_outputs_coords.append(torch.stack(outputs_coords))
            meta_support_class_ids.append(episode_class_ids)

        # Calculate targets for the constructed meta-tasks
        # meta_targets are computed based on original targets and the sampled support images.
        meta_targets = []
        for b in range(batchsize):
            for episode_class_ids in meta_support_class_ids:
                meta_target = dict()
                # 这里不需要额外的代码进行过滤，用原本的代码即可，因为100的class不会有的
                target_indexes = [i for i, x in enumerate(targets[b]['labels'].tolist()) if x in episode_class_ids]
                meta_target['boxes'] = targets[b]['boxes'][target_indexes]
                meta_target['labels'] = targets[b]['labels'][target_indexes]
                meta_target['area'] = targets[b]['area'][target_indexes]
                meta_target['iscrowd'] = targets[b]['iscrowd'][target_indexes]
                meta_target['image_id'] = targets[b]['image_id']
                meta_target['size'] = targets[b]['size']
                meta_target['orig_size'] = targets[b]['orig_size']
                meta_targets.append(meta_target)

        # Create tensors for final outputs
        # default logits are -inf (default confidence scores are 0.00 after sigmoid)
        final_meta_outputs_classes = torch.ones(hs.shape[0], batchsize, num_episode, self.num_queries, self.num_classes, device=device) * (-999999.99)
        final_meta_outputs_coords = torch.zeros(hs.shape[0], batchsize, num_episode, self.num_queries, 4, device=device)
        # 计算loss时用的预测:这里固定为21
        # loss_meta_outputs_classes = torch.ones(hs.shape[0], batchsize, num_episode, self.num_queries, self.num_classes, device=device) * (-999999.99)
        loss_meta_outputs_classes = torch.ones(hs.shape[0], batchsize, num_episode, self.num_queries, 91, device=device) * (-999999.99)
        # loss_meta_outputs_coords = torch.zeros(hs.shape[0], batchsize, num_episode, self.num_queries, 4, device=device)
        # 计算loss时用的支持id(其中class为100被随机替换)
        loss_updated_class_id = []
        # Fill in predicted logits into corresponding positions
        class_ids_already_filled_in = []
        """
        此处数组的作用：因为我们要计算背景类的loss，所以我们要将背景类随机分配给除了support中非背景的类别
        因为实际上是没有背景类的，所以只能给其他类别来计算loss，但二分匹配和postprocess不能用
        """
        """
        那么问题来了，这个loss计算实际和预测并不是用的一个结果
        那么即使对于训练集中只有[1, 2]这两个类的情况
        我们loss_meta_outputs_classes是不是还能用21类的输出，因为我们实际上监督的是五类输出头
        """
        # voc_base1_class_ids = [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20]
        # voc_base1_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        """
        因为finetune的数据集的class是从0开始的,所以这里加上0
        我认为即使基类数据集是从1开始的,也不影响finetune数据集从0开始,因为此网络除了类别损失的linear和类别有关,其余本质上是和类别无关的
        """
        # voc_base1_class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        coco_class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
                39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
                57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 
                75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
        for episode_index, (pred_classes, pred_coords, class_ids) in enumerate(zip(meta_outputs_classes, meta_outputs_coords, meta_support_class_ids)):
            ###############################################################################################################
            # if self.training:
            updated_class_id = []
            for class_index, class_id in enumerate(class_ids):
                if class_id == 100:
                    # 从 voc_base1_class_ids 中选一个不在 updated_class_ids 和 class_ids 中的值
                    available_ids = [i for i in coco_class_ids if i not in updated_class_id and i not in class_ids]
                    if available_ids:
                        new_class_id = torch.tensor(random.choice(available_ids), device= class_id.device)
                        updated_class_id.append(new_class_id)
                    else:
                        raise ValueError("No available IDs to replace 100")
                else:
                    updated_class_id.append(class_id)
            tmp_updated_class_id =  torch.stack(updated_class_id, dim=0)
            loss_updated_class_id.append(tmp_updated_class_id)
            # 这里提取的是含id 100的预测,只在训练时用得到，用于loss计算，bbox的预测不用改变
            # for class_index, class_id in enumerate(updated_class_id):
            #     # if self.training:
            #     loss_meta_outputs_classes[:, :, episode_index, :, class_id] = pred_classes[:, :, :, class_index]
            ###############################################################################################################
            """
            思路：二分匹配时用不含id 100的预测，计算loss时用含id 100的预测，因为这样能够监督背景类，后处理时用不含id 100的预测
            也就是说只在训练时用含id 100的预测
            """
            # 这里提取的是不含id 100的预测，用于推理后处理以及二分匹配
            for class_index, class_id in enumerate(class_ids):
                if class_id == 100:
                    continue
                # During inference, we need to ignore the classes that already have predictions
                # During training, the same category might appear over different episodes, so no need to filter
                """这里后面应该要再改"""
                if self.training or (class_id.item() not in class_ids_already_filled_in):
                    class_ids_already_filled_in.append(class_id.item())
                    final_meta_outputs_classes[:, :, episode_index, :, class_id] = pred_classes[:, :, :, class_index]
                    loss_meta_outputs_classes[:, :, episode_index, :, class_id] = pred_classes[:, :, :, class_index]
                    final_meta_outputs_coords[:, :, episode_index, :, :] = pred_coords[:, :, :, :]
        # Pretend we have a batchsize of (batchsize x num_support), and produce final predictions
        final_meta_outputs_classes = final_meta_outputs_classes.view(hs.shape[0], batchsize * num_episode, self.num_queries, self.num_classes)
        final_meta_outputs_coords = final_meta_outputs_coords.view(hs.shape[0], batchsize * num_episode, self.num_queries, 4)
        # 类别loss计算时用:这里固定为21
        # loss_meta_outputs_classes = loss_meta_outputs_classes.view(hs.shape[0], batchsize * num_episode, self.num_queries, self.num_classes)
        loss_meta_outputs_classes = loss_meta_outputs_classes.view(hs.shape[0], batchsize * num_episode, self.num_queries, 91)

        out = dict()

        out['pred_logits'] = final_meta_outputs_classes[-1]
        out['pred_boxes'] = final_meta_outputs_coords[-1]
        out['activated_class_ids'] = torch.stack(meta_support_class_ids).unsqueeze(0).expand(batchsize, -1, -1).reshape(batchsize * num_episode, -1)
        out['meta_targets'] = meta_targets  # Add meta_targets into outputs for optimization

        out['batchsize'] = batchsize
        out['num_episode'] = num_episode
        out['num_queries'] = self.num_queries
        out['num_classes'] = self.num_classes

        """
        ！！！！！！！！！！！！！！！！！！！！！！消融实验注释！！！！！！！！！！！！！！！！！！！！！！
        ！！！！！！！！！！！！！！！！！！！！！！消融实验注释！！！！！！！！！！！！！！！！！！！！！！
        ！！！！！！！！！！！！！！！！！！！！！！消融实验注释！！！！！！！！！！！！！！！！！！！！！！
        """
        # # 类别loss计算时用
        # out['loss_pred_logits'] = loss_meta_outputs_classes[-1]
        # out['loss_updated_class_id'] = torch.stack(loss_updated_class_id).unsqueeze(0).expand(batchsize, -1, -1).reshape(batchsize * num_episode, -1)
        
        # 类别loss计算时用
        out['loss_pred_logits'] = loss_meta_outputs_classes[-1]
        out['loss_updated_class_id'] = torch.stack(loss_updated_class_id).unsqueeze(0).expand(batchsize, -1, -1).reshape(batchsize * num_episode, -1)

        if self.args.category_codes_cls_loss:
            if self.num_feature_levels == 1:
                # out['category_codes_cls_logits'] = self.category_codes_cls(category_codes)
                # out['category_codes_cls_targets'] = supp_class_ids
                # TODO: category_codes_cls_loss @ every encoder layer! THIS IS ONLY TRIAL!
                # out['category_codes_cls_logits'] = self.category_codes_cls(torch.cat(category_codes, dim=0))
                # out['category_codes_cls_targets'] = supp_class_ids.repeat(self.args.dec_layers)
                """
                这里也需要改,假如25个提取中有5个背景类,那么类别编码就20个,但supp_class_ids会有25个,需要去掉
                """
                if self.args.multi_category_loss:
                    if self.args.all_distLinear:
                        output = []
                        output.append(self.category_codes_cls(category_codes[0]))
                        output.append(self.category_codes_cls1(category_codes[1]))
                        output.append(self.category_codes_cls2(category_codes[2]))
                        output.append(self.category_codes_cls3(category_codes[3]))
                        output.append(self.category_codes_cls4(category_codes[4]))
                        output.append(self.category_codes_cls5(category_codes[5]))
                        # 将每个样本的结果堆叠回一起，形成批次的输出
                        # out['category_codes_cls_logits'] = torch.stack(output)
                        out['category_codes_cls_logits'] = torch.cat(output, dim=0)
                        out['category_codes_cls_targets'] = supp_class_ids[supp_class_ids != 100].repeat(self.args.dec_layers)
                    else:
                        # 假如有22个非背景class，那么torch.cat(category_codes, dim=0)为（22*6,256），并不是（6，22，256）
                        # 下面repeat后也一样，是(22*6)，不是(6,22)
                        out['category_codes_cls_logits'] = self.category_codes_cls(torch.cat(category_codes, dim=0))
                        out['category_codes_cls_targets'] = supp_class_ids[supp_class_ids != 100].repeat(self.args.dec_layers)
                else:
                    out['category_codes_cls_logits'] = self.category_codes_cls(category_codes[0])
                    # out['category_codes_cls_targets'] = supp_class_ids
                    out['category_codes_cls_targets'] = supp_class_ids[supp_class_ids != 100]
            elif self.num_feature_levels == 4:
                raise NotImplementedError
            else:
                raise NotImplementedError

        """
        ！！！！！！！！！！！！！！！！！！！！！！消融实验注释！！！！！！！！！！！！！！！！！！！！！！
        ！！！！！！！！！！！！！！！！！！！！！！消融实验注释！！！！！！！！！！！！！！！！！！！！！！
        ！！！！！！！！！！！！！！！！！！！！！！消融实验注释！！！！！！！！！！！！！！！！！！！！！！
        """
        # if self.aux_loss:
        #     """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
        #     # 这里也要修改，放入类别loss
        #     out['aux_outputs'] = self._set_aux_loss(final_meta_outputs_classes, final_meta_outputs_coords, loss_meta_outputs_classes)
        #     for aux_output in out['aux_outputs']:
        #         aux_output['activated_class_ids'] = torch.stack(meta_support_class_ids).unsqueeze(0).expand(batchsize, -1, -1).reshape(batchsize * num_episode, -1)
        #         aux_output['loss_updated_class_id'] = torch.stack(loss_updated_class_id).unsqueeze(0).expand(batchsize, -1, -1).reshape(batchsize * num_episode, -1)
        
        if self.aux_loss:
            # 这里也要修改，放入类别loss
            out['aux_outputs'] = self._set_aux_loss(final_meta_outputs_classes, final_meta_outputs_coords, loss_meta_outputs_classes)
            for aux_output in out['aux_outputs']:
                aux_output['activated_class_ids'] = torch.stack(meta_support_class_ids).unsqueeze(0).expand(batchsize, -1, -1).reshape(batchsize * num_episode, -1)
                aux_output['loss_updated_class_id'] = torch.stack(loss_updated_class_id).unsqueeze(0).expand(batchsize, -1, -1).reshape(batchsize * num_episode, -1)
        
        
        return out

    def compute_category_codes(self, supp_samples, supp_targets, supp_class_ids):
        # 这里不能再用supp_samples的shape[0]，要用supp_class_ids的，因为前者shape[0]是动态的
        # num_supp = supp_samples.tensors.shape[0]
        num_supp = supp_class_ids.shape[0]

        if self.num_feature_levels == 1:
            if self.args.backbone == 'dinov2':
                if not self.args.VPT_enable:
                    if self.backbone[0].training: self.backbone[0].eval()
                    with autocast(enabled=True):
                        features, pos = self.backbone.forward_supp_branch(supp_samples, return_interm_layers=False)
                else:
                    if self.backbone[0].training: self.backbone[0].eval()
                    self.backbone[0].prompt_dropout.train()
                    with autocast(enabled=True):
                        features, pos = self.backbone.forward_supp_branch(supp_samples, return_interm_layers=False, VPT_enable=True)
            else:
                features, pos = self.backbone.forward_supp_branch(supp_samples, return_interm_layers=False)
            srcs = []
            masks = []
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src))
                masks.append(mask)
                assert mask is not None

            boxes = [box_ops.box_cxcywh_to_xyxy(t['boxes']) for t in supp_targets]
            # and from relative [0, 1] to absolute [0, height] coordinates
            img_sizes = torch.stack([t["size"] for t in supp_targets], dim=0)
            img_h, img_w = img_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            for b in range(supp_samples.tensors.shape[0]):
                boxes[b] *= scale_fct[b]

            query_embeds = self.query_embed.to(src.device)
            """
            expand(num_supp, -1, -1) 扩展张量的形状为 (num_supp, self.args.episode_size, self.hidden_dim)，相当于复制了 num_supp 份。
            """
            tsp = self.task_positional_encoding(torch.zeros(self.args.episode_size, self.hidden_dim, device=src.device)).unsqueeze(0).expand(num_supp, -1, -1)
            # 因为tsp是不可训练的，也即为定值，所以我们就不在这里拓展了，直接将[5,256]送到SingleHeadSiameseAttention进行拓展即可
            # tsp = self.task_positional_encoding(torch.zeros(self.args.episode_size, self.hidden_dim, device=src.device))
            category_codes_list = list()

            count = 0
            for i in range(num_supp // self.args.episode_size):
                src = []
                mask = []
                pos_tmp = []
                box = []
                tsp_tmp = []
                supp_class_id = []
                for index, tmp in enumerate(supp_class_ids[i*self.args.episode_size: (i+1)*self.args.episode_size]):
                    if tmp != 100:
                        # src.append(srcs[0][i*self.args.episode_size + index: index + i*self.args.episode_size + 1])
                        # mask.append(masks[0][i*self.args.episode_size + index: index + i*self.args.episode_size + 1])
                        # pos_tmp.append(pos[0][i*self.args.episode_size + index: index + i*self.args.episode_size + 1])
                        # # box不能给一个范围提取，否则就是嵌套列表
                        # box.append(boxes[i*self.args.episode_size + index])
                        src.append(srcs[0][count])
                        mask.append(masks[0][count])
                        pos_tmp.append(pos[0][count])
                        # box不能给一个范围提取，否则就是嵌套列表
                        box.append(boxes[count])
                        tsp_tmp.append(tsp[i*self.args.episode_size + index])
                        count = count + 1
                
                # 因为forward_supp_branch解析的格式需要是lvl,eposide_size,C,H,W
                # 上面是eposide_size个C,H,W
                # 这里会变成eposide_size,C,H,W
                src_cat = torch.stack(src, dim=0)
                mask_cat = torch.stack(mask, dim=0)
                pos_cat = torch.stack(pos_tmp, dim=0)
                tsp_in = torch.stack(tsp_tmp, dim=0)

                # 这里会变成lvl,eposide_size,C,H,W(当然,lvl就一层)
                src_in = []
                mask_in = []
                pos_in = []
                src_in.append(src_cat)
                mask_in.append(mask_cat)
                pos_in.append(pos_cat)
                supp_class_id = supp_class_ids[i*self.args.episode_size: (i+1)*self.args.episode_size]

                category_codes_list.append(
                    self.transformer.forward_supp_branch(src_in,
                                                         mask_in,
                                                         pos_in,
                                                         query_embeds,
                                                         tsp_in,
                                                         box,
                                                         supp_class_id)
                )

                # category_codes_list.append(
                #     self.transformer.forward_supp_branch([srcs[0][i*self.args.episode_size: (i+1)*self.args.episode_size]],
                #                                          [masks[0][i*self.args.episode_size: (i+1)*self.args.episode_size]],
                #                                          [pos[0][i*self.args.episode_size: (i+1)*self.args.episode_size]],
                #                                          query_embeds,
                #                                          tsp[i*self.args.episode_size: (i+1)*self.args.episode_size],
                #                                          boxes[i*self.args.episode_size: (i+1)*self.args.episode_size],
                #                                          supp_class_id)
                # )
            if count != supp_samples.tensors.shape[0]:
                raise ValueError
            else:
                count = 0
            final_category_codes_list = []
            for i in range(self.args.enc_layers):
                final_category_codes_list.append(
                    torch.cat([ccl[i] for ccl in category_codes_list], dim=0)
                )
            # 在适当的地方调用，通常是在内存压力较大的部分之后
            # del category_codes_list, features, pos, srcs, masks
            # torch.cuda.empty_cache()
            
            return final_category_codes_list

        elif self.num_feature_levels == 4:
            raise NotImplementedError
        else:
            raise NotImplementedError

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, loss_outputs_class):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'loss_pred_logits': c} for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], loss_outputs_class[:-1])]


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class SetCriterion(nn.Module):
    """ 
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, args, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # 不使用原来的，使用更新后的
        # assert 'pred_logits' in outputs
        # src_logits = outputs['pred_logits']
        assert 'loss_pred_logits' in outputs
        src_logits = outputs['loss_pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]

        # ################### Only Produce Loss for Activated Categories ###################
        # activated_class_ids = outputs['activated_class_ids']   # (bs, num_support)
        activated_class_ids = outputs['loss_updated_class_id']   # (bs, num_support)
        activated_class_ids = activated_class_ids.unsqueeze(1).repeat(1, target_classes_onehot.shape[1], 1)
        loss_ce = sigmoid_focal_loss(src_logits.gather(2, activated_class_ids),
                                     target_classes_onehot.gather(2, activated_class_ids),
                                     num_boxes,
                                     alpha=self.focal_alpha,
                                     gamma=2)

        loss_ce = loss_ce * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = dict()
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_category_codes_cls(self, outputs, targets, indices, num_boxes):
        logits = outputs['category_codes_cls_logits']
        targets = outputs['category_codes_cls_targets']
        losses = {
            "loss_category_codes_cls": F.cross_entropy(logits, targets)
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'category_codes_cls': self.loss_category_codes_cls,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
        """
        # Since we are doing meta-learning over our constructed meta-tasks, the targets for these meta-tasks are
        # stored in outputs['meta_targets']. We dont use original targets.
        targets = outputs['meta_targets']

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'category_codes_cls':
                        # meta-attention cls loss not for aux_outputs
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        batchsize = outputs['batchsize']
        num_episode = outputs['num_episode']
        num_queries = outputs['num_queries']
        num_classes = outputs['num_classes']

        out_logits = out_logits.view(batchsize, num_episode * num_queries, num_classes)
        out_bbox = out_bbox.view(batchsize, num_episode * num_queries, 4)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    if args.dataset_file in ['coco', 'coco_base']:
        num_classes = 91
    elif args.dataset_file in ['voc', 'voc_base1', 'voc_base2', 'voc_base3']:
        num_classes = 21
    # UODD:3 + 1 = 4
    # 这里应该是不用+1的，因为uodd的id是从0开始的三类，012
    # 但是这里加了也没影响，因为不影响匹配和self.category_codes_cls = distLinear(self.hidden_dim, self.num_classes)计算
    # 而voc就必须得加，因为id从1开始
    elif args.dataset_file in ['uodd']:
        num_classes = 4
    # FISH:1 + 1 = 2（这个得加，因为类别id从1开始，为1类）
    elif args.dataset_file in ['deepfish']:
        num_classes = 2
    # NEU-DET:1 + 6 = 7（这个得加，因为类别id从1开始，为1类）
    elif args.dataset_file in ['neu']:
        num_classes = 7
    # clipart1k:1 + 20 = 21（这个得加，因为类别id从1开始，为1类）
    elif args.dataset_file in ['clipart']:
        num_classes = 21
    # artaxor:1 + 7 = 8（这个得加，因为类别id从1开始，为1类）
    elif args.dataset_file in ['artaxor']:
        num_classes = 8
    # dior:1 + 20 = 21（这个得加，因为类别id从1开始，为1类）
    elif args.dataset_file in ['dior']:
        num_classes = 21
    # dior:1 + 7 = 8（这个可以不加，因为类别id从0开始，为7类）
    elif args.dataset_file in ['dior']:
        num_classes = 21
    else:
        raise ValueError('Unknown args.dataset_file!')

    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    model = CDFormer(
        args,
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
    )

    matcher = build_matcher(args)

    weight_dict = dict()
    weight_dict['loss_ce'] = args.cls_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.category_codes_cls_loss:
        weight_dict["loss_category_codes_cls"] = args.category_codes_cls_loss_coef

    losses = ['labels', 'boxes', 'cardinality']

    if args.category_codes_cls_loss:
        losses += ["category_codes_cls"]

    criterion = SetCriterion(args, num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)

    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

