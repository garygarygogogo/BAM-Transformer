import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

import numpy as np
import random
import time
import cv2

import model.resnet as models
import model.vgg as vgg_models
from model.loss import WeightedDiceLoss
from model.cyc_transformer import CyCTransformer
from model.ops.modules import MSDeformAttn
from model.ASPP import ASPP
from model.PSPNet import OneModel as PSPNet
from util.util import get_train_val_set


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def get_gram_matrix(fea):
    b, c, h, w = fea.shape        
    fea = fea.reshape(b, c, h*w)    # C*N
    fea_T = fea.permute(0, 2, 1)    # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)    # C*C
    return gram


class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.dataset = args.data_set
        # self.criterion = WeightedDiceLoss()
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        trans_multi_lvl=1
        self.trans_multi_lvl = trans_multi_lvl

        self.print_freq = args.print_freq/2

        self.pretrained = True
        self.classes = 2
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60
        
        assert self.layers in [50, 101, 152]
    
        PSPNet_ = PSPNet(args)
        backbone_str = 'vgg' if args.vgg else 'resnet'+str(args.layers)
        weight_path = 'initmodel/PSPNet/{}/split{}/{}/best.pth'.format(args.data_set, args.split, backbone_str)               
        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
        try: 
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:                   # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4

        # Base Learner
        self.learner_base = nn.Sequential(PSPNet_.ppm, PSPNet_.cls)

        # Meta Learner
        reduce_dim = 256
        self.low_fea_id = args.low_fea[-1]
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512           
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        mask_add_num = 1
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.supp_merge_feat = nn.Sequential(
            nn.Conv2d(reduce_dim*2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            )

        self.transformer = CyCTransformer(embed_dims=reduce_dim, shot=self.shot, num_points=9)
        self.merge_multi_lvl_reduce = nn.Sequential(
            nn.Conv2d(reduce_dim*self.trans_multi_lvl, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            )
        self.merge_multi_lvl_sum = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
                )
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1))

        # Gram and Meta
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.gram_merge.weight))

        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.cls_merge.weight))
        
        
        qry_dim_scalar = 1
        self.pred_supp_qry_proj = nn.Sequential(
                nn.Conv2d(reduce_dim*qry_dim_scalar, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )
        scalar = 2
        self.supp_init_merge = nn.Sequential(
                nn.Conv2d(reduce_dim*scalar, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )
        self.supp_beta_conv = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
        self.supp_cls = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
            )

        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

        self.sigmoid = nn.Sigmoid()

    def get_optim(self, model, args, LR):
        if args.shot > 1:
            optimizer = torch.optim.SGD(
                [     
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.init_merge.parameters()},
                {'params': model.supp_merge_feat.parameters()},
                {'params': model.merge_multi_lvl_reduce.parameters()},
                {'params': model.merge_multi_lvl_sum.parameters()},
                {'params': model.cls_meta.parameters()},
                {'params': model.gram_merge.parameters()},
                {'params': model.cls_merge.parameters()},
                {'params': model.kshot_rw.parameters()},
                {'params': model.pred_supp_qry_proj.parameters()},
                {'params': model.supp_init_merge.parameters()},
                {'params': model.supp_beta_conv.parameters()},
                {'params': model.supp_cls.parameters()},
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
            
        else:
            optimizer = torch.optim.SGD(
                [     
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.init_merge.parameters()},
                {'params': model.supp_merge_feat.parameters()},
                {'params': model.merge_multi_lvl_reduce.parameters()},
                {'params': model.merge_multi_lvl_sum.parameters()},        
                {'params': model.cls_meta.parameters()},
                {'params': model.gram_merge.parameters()},
                {'params': model.cls_merge.parameters()}, 
                {'params': model.pred_supp_qry_proj.parameters()},
                {'params': model.supp_init_merge.parameters()},
                {'params': model.supp_beta_conv.parameters()},
                {'params': model.supp_cls.parameters()},
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        transformer_param_dicts = [
            {   
                "params": [p for n, p in model.named_parameters() if "transformer" in n and "bias" not in n and p.requires_grad],
                "lr": 1e-4,
                "weight_decay": 1e-2,
            },
            {
                "params": [p for n, p in model.named_parameters() if "transformer" in n and "bias" in n and p.requires_grad],
                "lr": 1e-4,
                "weight_decay": 0,
            }    
        ]            
        transformer_optimizer = torch.optim.AdamW(transformer_param_dicts, lr=1e-4, weight_decay=1e-4)
        return optimizer, transformer_optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.learner_base.parameters():
            param.requires_grad = False


    # que_img, sup_img, sup_mask, que_mask(meta), que_mask(base), cat_idx(meta)
    def forward(self, x, s_x, s_y, y_m, y_b, cat_idx=None, padding_mask=None, s_padding_mask=None):
        x_size = x.size()
        img_size = x.size()[-2:]
        bs = x_size[0]
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)
        mid_query_feat = query_feat.clone()
        
        fts_size = query_feat.shape[-2:]
        supp_mask = F.interpolate((s_y==1).view(-1, *img_size).float().unsqueeze(1), size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)    

        # Support Feature
        supp_pro_list = []
        final_supp_list = []
        mask_list = []
        supp_feat_list = [] 
        supp_feat_cat_list = []
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_feat_cat_list.append(supp_feat)
            supp_pro = Weighted_GAP(supp_feat, mask)
            supp_pro_list.append(supp_pro)
            supp_feat_list.append(eval('supp_feat_' + self.low_fea_id))
        
        # K-Shot Reweighting
        que_gram = get_gram_matrix(eval('query_feat_' + self.low_fea_id)) # [bs, C, C] in (0,1)
        norm_max = torch.ones_like(que_gram).norm(dim=(1,2))
        est_val_list = []
        for supp_item in supp_feat_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1,2))/norm_max).reshape(bs,1,1,1)) # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            weight = weight.gather(1, idx2)
            weight_soft = torch.softmax(weight, 1)
            for i in range(self.shot):
                supp_feat = torch.cat(supp_feat_cat_list, dim=0)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1,True) # [bs, 1, 1, 1]            

        # Prior Similarity Mask
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s               
            tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1) 
            tmp_supp = tmp_supp.permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].reshape(bsize, sp_sz*sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1)
        corr_query_mask = (weight_soft * corr_query_mask).sum(1,True)

        # Support Prototype
        supp_pro = torch.cat(supp_pro_list, 2)  # [bs, 256, shot, 1]
        supp_pro = (weight_soft.permute(0,2,1,3) * supp_pro).sum(2,True)
        
        if self.shot > 1:
            multi_supp_pp = Weighted_GAP(supp_feat, supp_mask) # [bs*shot, c, 1, 1]
        else:
            multi_supp_pp = supp_pro

        # Tile & Cat
        concat_feat = supp_pro.expand_as(query_feat)
        merge_feat = torch.cat([query_feat, concat_feat, corr_query_mask], 1)   # 256+256+1
        merge_feat = self.init_merge(merge_feat)
               
        # Transformer
        to_merge_fts = [supp_feat, concat_feat]
        aug_supp_feat = torch.cat(to_merge_fts, dim=1)
        aug_supp_feat = self.supp_merge_feat(aug_supp_feat)

        query_feat_list = self.transformer(merge_feat, padding_mask.float(), aug_supp_feat, s_y.clone().float(), s_padding_mask.float())
        fused_query_feat = []
        for lvl, qry_feat in enumerate(query_feat_list):
            if lvl == 0:
                fused_query_feat.append(qry_feat)
            else:
                fused_query_feat.append(F.interpolate(qry_feat, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True))
        fused_query_feat = torch.cat(fused_query_feat, dim=1)
        fused_query_feat = self.merge_multi_lvl_reduce(fused_query_feat)
        fused_query_feat = self.merge_multi_lvl_sum(fused_query_feat)+fused_query_feat
            
        # Base and Meta
        base_out = self.learner_base(query_feat_4)

        meta_out = self.cls_meta(fused_query_feat)
        
        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # Classifier Ensemble
        meta_map_bg = meta_out_soft[:,0:1,:,:]                           # [bs, 1, 60, 60]
        meta_map_fg = meta_out_soft[:,1:,:,:]                            # [bs, 1, 60, 60]
        if self.training and self.cls_type == 'Base':
            c_id_array = torch.arange(self.base_classes+1, device='cuda')
            base_map_list = []
            for b_id in range(bs):
                c_id = cat_idx[0][b_id] + 1
                c_mask = (c_id_array!=0)&(c_id_array!=c_id)
                base_map_list.append(base_out_soft[b_id,c_mask,:,:].unsqueeze(0).sum(1,True))
            base_map = torch.cat(base_map_list,0)
            # <alternative implementation>
            # gather_id = (cat_idx[0]+1).reshape(bs,1,1,1).expand_as(base_out_soft[:,0:1,:,:]).cuda()
            # fg_map = base_out_soft.gather(1,gather_id)
            # base_map = base_out_soft[:,1:,:,:].sum(1,True) - fg_map            
        else:
            base_map = base_out_soft[:,1:,:,:].sum(1,True)

        est_map = est_val.expand_as(meta_map_fg)

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg,est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg,est_map], dim=1))

        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)                     # [bs, 1, 60, 60]

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        # Output Part
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
            final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

        # Loss
        if self.training:
            # prepare inputs for aux loss
            qry_mask = F.interpolate((y_m==1).float().unsqueeze(1), size=(fused_query_feat.size(2), fused_query_feat.size(3)), mode='bilinear', align_corners=True) # 'nearest')
            qry_proj_feat = self.pred_supp_qry_proj(fused_query_feat)+mid_query_feat
            qry_pp = Weighted_GAP(qry_proj_feat, qry_mask)
            qry_pp = qry_pp
            qry_pp = qry_pp.expand(-1, -1, supp_feat.size(2), supp_feat.size(3)) # default
            temp_supp_feat = supp_feat.view(bs, self.shot, -1, supp_feat.size(2), supp_feat.size(3))
            supp_out_list = []
            for st_id in range(self.shot):
                supp_merge_bin = torch.cat([temp_supp_feat[:, st_id, ...], qry_pp], dim=1)
                merge_supp_feat = self.supp_init_merge(supp_merge_bin)
                merge_supp_feat = self.supp_beta_conv(merge_supp_feat) + merge_supp_feat
                supp_out = self.supp_cls(merge_supp_feat)
                supp_out_list.append(supp_out)
            
            main_loss = self.criterion(final_out, y_m.long())
            out_list = []
            for lvl, query_feat in enumerate(query_feat_list):
                inter_out = self.cls_meta[lvl](query_feat)
                out_list.append(F.interpolate(inter_out, size=(h, w), mode='bilinear', align_corners=True))

            aux_loss1 = torch.zeros_like(main_loss)
            for st_id, supp_out in enumerate(supp_out_list):
                supp_out = F.interpolate(supp_out, size=(h, w), mode='bilinear', align_corners=True)
                supp_loss = self.criterion(supp_out, s_y[:, st_id, ...].long())
                aux_loss1 += supp_loss/self.shot
                          
            # aux_loss1 = self.criterion(meta_out, y_m.long())
            aux_loss2 = self.criterion(base_out, y_b.long())
            return final_out.max(1)[1], main_loss, aux_loss1, aux_loss2
        else:
            return final_out, meta_out, base_out

