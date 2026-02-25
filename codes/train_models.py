import os
import random
from torch.utils.data.dataloader import DataLoader
import tqdm
import test as test
from loss import EntailmentConeLoss, HyperbolicHardNegativeAlignmentLoss
from torch.nn.modules.loss import CrossEntropyLoss
import torch.multiprocessing
import numpy as np
import json
from clip import clip

def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(model, dataset, config)
    test_stats = test.test(dataset, evaluator, all_logits, all_attr_gt, all_obj_gt, all_pair_gt, config)
    result = ""
    key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]
    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)
    model.train()
    return loss_avg, test_stats

def save_checkpoint(state, save_path, epoch, best=False):
    filename = os.path.join(save_path, f"epoch_resume.pt")
    torch.save(state, filename)

def c2c_vanilla(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset, scaler):
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    print("Pre-extracting coarse text Euclidean features using frozen CLIP...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frozen_clip, _ = clip.load(config.backbone, device=device)
    frozen_clip.eval()
    
    coarse_v_feats_dict, coarse_o_feats_dict = {}, {}
    with torch.no_grad():
        for cv in set(train_dataset.verb_hierarchy.values()):
            tokens = clip.tokenize(f"an action of {cv}").cuda()
            coarse_v_feats_dict[cv] = frozen_clip.encode_text(tokens).float()
        for co in set(train_dataset.obj_hierarchy.values()):
            tokens = clip.tokenize(f"a photo of a {co}").cuda()
            coarse_o_feats_dict[co] = frozen_clip.encode_text(tokens).float()
            
    del frozen_clip
    print("Pre-extraction complete.")

    model.train()
    best_loss = 1e5
    best_metric = 0
    Loss_fn = CrossEntropyLoss()
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'w')

    attr2idx, obj2idx = train_dataset.attr2idx, train_dataset.obj2idx
    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj]) for attr, obj in train_dataset.train_pairs]).cuda()

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(total=len(train_dataloader), desc="epoch % 3d" % (i + 1))
        epoch_train_losses, epoch_com_losses = [], []
        epoch_oo_losses, epoch_vv_losses = [], []
        epoch_ent_losses, epoch_ali_losses = [], []

        print(f"Current_lr:{optimizer.param_groups[-1]['lr']}")
        actual_model = model.module if hasattr(model, 'module') else model

        for bid, batch in enumerate(train_dataloader):
            batch_img = batch[0].cuda()
            batch_verb, batch_obj, batch_target = batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            batch_coarse_verb, batch_coarse_obj = batch[4], batch[5]

            # 1. AMP 半精度跑 Backbone
            with torch.cuda.amp.autocast(enabled=True):
                verb_logits_hyp, obj_logits_hyp, v_feat_hyp, o_feat_hyp, verb_text_hyp, obj_text_hyp, _curv = model(batch_img)
                
                coarse_v_raw = torch.cat([coarse_v_feats_dict[cv] for cv in batch_coarse_verb], dim=0).cuda()
                coarse_o_raw = torch.cat([coarse_o_feats_dict[co] for co in batch_coarse_obj], dim=0).cuda()
                
                coarse_v_eucl = actual_model.c2c_text_v(coarse_v_raw)
                coarse_o_eucl = actual_model.c2c_text_o(coarse_o_raw)
                
            # 2. 强制 FP32 计算双曲法则
            with torch.cuda.amp.autocast(enabled=False):
                _curv_fp32 = _curv.float()
                t_alpha_fp32 = actual_model.textual_alpha.exp().float()
                
                coarse_v_eucl_norm = torch.nan_to_num(coarse_v_eucl.float() / (coarse_v_eucl.float().norm(dim=-1, keepdim=True) + 1e-5))
                coarse_o_eucl_norm = torch.nan_to_num(coarse_o_eucl.float() / (coarse_o_eucl.float().norm(dim=-1, keepdim=True) + 1e-5))
                
                coarse_v_tangent = actual_model.hyp_proj_v_text(coarse_v_eucl_norm)
                coarse_o_tangent = actual_model.hyp_proj_o_text(coarse_o_eucl_norm)
                
                import models.vlm_models.lorentz as L
                coarse_v_hyp = L.exp_map0(coarse_v_tangent * t_alpha_fp32, _curv_fp32)
                coarse_o_hyp = L.exp_map0(coarse_o_tangent * t_alpha_fp32, _curv_fp32)
                
                # ====== 动态读取配置参数 ======
                w_att_obj = getattr(config, 'att_obj_w', 0.2)
                w_entail = getattr(config, 'lambda_entail', getattr(config, 'w_entail', 0.1))
                w_align = getattr(config, 'lambda_align', getattr(config, 'w_align', 1.0))
                align_margin = getattr(config, 'align_margin', 0.2)
                entail_margin = getattr(config, 'entail_margin', 0.01)
                hyp_temp = getattr(config, 'hyp_temp', 20.0)

                entail_loss_fn = EntailmentConeLoss(margin=entail_margin)
                align_loss_fn = HyperbolicHardNegativeAlignmentLoss(margin=align_margin)
                
                # ==============================================================
                # 【核心逻辑】：纯双曲分数的组合！
                # 动词双曲分数 + 物品双曲分数 = 组合动作双曲分数
                # ==============================================================
                train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]
                pred_com_train = verb_logits_hyp[:, train_v_inds] + obj_logits_hyp[:, train_o_inds]
                
                # 双曲空间的 CrossEntropy
                loss_com = Loss_fn(pred_com_train * hyp_temp, batch_target)
                loss_verb_hyp = Loss_fn(verb_logits_hyp * hyp_temp, batch_verb)
                loss_obj_hyp = Loss_fn(obj_logits_hyp * hyp_temp, batch_obj)
                
                # 判别对齐
                loss_align_v = align_loss_fn(verb_logits_hyp, batch_verb)
                loss_align_o = align_loss_fn(obj_logits_hyp, batch_obj)
                loss_alignment = loss_align_v + loss_align_o
                
                # 层次蕴含
                batch_fine_v_hyp = verb_text_hyp[batch_verb].float()
                batch_fine_o_hyp = obj_text_hyp[batch_obj].float()
                loss_ent_v = entail_loss_fn(batch_fine_v_hyp, coarse_v_hyp, _curv_fp32)
                loss_ent_o = entail_loss_fn(batch_fine_o_hyp, coarse_o_hyp, _curv_fp32)
                loss_ent_vid_v = entail_loss_fn(v_feat_hyp.float(), batch_fine_v_hyp, _curv_fp32)
                loss_ent_vid_o = entail_loss_fn(o_feat_hyp.float(), batch_fine_o_hyp, _curv_fp32)
                loss_entailment = loss_ent_v + loss_ent_o + loss_ent_vid_v + loss_ent_vid_o
                
                # 【统一流形总损失】：完全抛弃欧式特征计算，全盘双曲化！
                loss = loss_com + w_att_obj * (loss_verb_hyp + loss_obj_hyp) + w_entail * loss_entailment + w_align * loss_alignment
                
                loss = torch.nan_to_num(loss) / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)
                # 删除了 clip_grad_norm_ 保障安全
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            epoch_com_losses.append(loss_com.item())
            # 记录的也是纯双曲的基础 Loss
            epoch_vv_losses.append(loss_verb_hyp.item())
            epoch_oo_losses.append(loss_obj_hyp.item())
            epoch_ent_losses.append(loss_entailment.item())
            epoch_ali_losses.append(loss_alignment.item())

            current_loss = np.nanmean(epoch_train_losses[-50:])
            progress_bar.set_postfix({"train loss": current_loss})
            progress_bar.update()

        lr_scheduler.step()
        progress_bar.close()
        
        log_training.write(f"\nepoch {i + 1} train loss {np.nanmean(epoch_train_losses)}\n")
        log_training.write(f"epoch {i + 1} com loss {np.nanmean(epoch_com_losses)}\n")

        if (i + 1) % config.save_every_n == 0:
            save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': lr_scheduler.state_dict(), 'scaler': scaler.state_dict()}, config.save_path, i)
            
        if i % config.eval_every_n == 0 or i + 1 == config.epochs or i >= config.val_epochs_ts:
            loss_avg, val_result = evaluate(model, val_dataset, config)
            if loss_avg.cpu().float() < best_loss:
                best_loss = loss_avg.cpu().float()
                torch.save(model.state_dict(), os.path.join(config.save_path, f"best.pt"))
                evaluate(model, test_dataset, config)
        log_training.flush()