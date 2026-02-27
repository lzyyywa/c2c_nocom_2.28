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

    use_frozen_coarse = False
    print("Forcing coarse parents to be computed from mean of fine children (log-map mean) to preserve hyperbolic hierarchy.")

    model.train()
    best_loss = 1e5
    best_metric = 0
    Loss_fn = CrossEntropyLoss()
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'w')

    attr2idx, obj2idx = train_dataset.attr2idx, train_dataset.obj2idx
    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj]) for attr, obj in train_dataset.train_pairs]).cuda()

    print("Building Semantic Hard Negative Masks and Global Tree Indices...")
    idx2attr = {v: k for k, v in attr2idx.items()}
    num_verbs = len(idx2attr)

    train_verbs = set([a for (a, _) in train_dataset.train_pairs])
    train_objs = set([o for (_, o) in train_dataset.train_pairs])
    is_train_verb = torch.tensor([idx2attr[i] in train_verbs for i in range(num_verbs)], dtype=torch.bool)

    verb_hard_mask_matrix = torch.zeros((num_verbs, num_verbs), dtype=torch.bool)
    for i in range(num_verbs):
        if not is_train_verb[i]:
            continue
        coarse_i = train_dataset.verb_hierarchy[idx2attr[i]]
        for j in range(num_verbs):
            if i != j and is_train_verb[j] and train_dataset.verb_hierarchy[idx2attr[j]] == coarse_i:
                verb_hard_mask_matrix[i, j] = True
        if not verb_hard_mask_matrix[i].any():
            verb_hard_mask_matrix[i] = is_train_verb
            verb_hard_mask_matrix[i, i] = False

    idx2obj = {v: k for k, v in obj2idx.items()}
    num_objs = len(idx2obj)
    is_train_obj = torch.tensor([idx2obj[i] in train_objs for i in range(num_objs)], dtype=torch.bool)
    obj_hard_mask_matrix = torch.zeros((num_objs, num_objs), dtype=torch.bool)
    for i in range(num_objs):
        if not is_train_obj[i]:
            continue
        coarse_i = train_dataset.obj_hierarchy[idx2obj[i]]
        for j in range(num_objs):
            if i != j and is_train_obj[j] and train_dataset.obj_hierarchy[idx2obj[j]] == coarse_i:
                obj_hard_mask_matrix[i, j] = True
        if not obj_hard_mask_matrix[i].any():
            obj_hard_mask_matrix[i] = is_train_obj
            obj_hard_mask_matrix[i, i] = False

    unique_coarse_verbs = sorted(set(train_dataset.verb_hierarchy.values()))
    fine2coarse_v_idx = torch.tensor([unique_coarse_verbs.index(train_dataset.verb_hierarchy[idx2attr[i]]) for i in range(num_verbs)]).cuda()
    coarse_v_counts = torch.bincount(fine2coarse_v_idx, minlength=len(unique_coarse_verbs)).float().clamp_min(1.0)

    unique_coarse_objs = sorted(set(train_dataset.obj_hierarchy.values()))
    fine2coarse_o_idx = torch.tensor([unique_coarse_objs.index(train_dataset.obj_hierarchy[idx2obj[i]]) for i in range(num_objs)]).cuda()
    coarse_o_counts = torch.bincount(fine2coarse_o_idx, minlength=len(unique_coarse_objs)).float().clamp_min(1.0)
    print("Masks and Tree Built Successfully.")

    mask_device = train_pairs.device
    verb_hard_mask_matrix = verb_hard_mask_matrix.to(mask_device)
    obj_hard_mask_matrix = obj_hard_mask_matrix.to(mask_device)

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

            with torch.cuda.amp.autocast(enabled=True):
                verb_logits_hyp, obj_logits_hyp, v_feat_hyp, o_feat_hyp, verb_text_hyp, obj_text_hyp, _curv = model(batch_img)

            with torch.cuda.amp.autocast(enabled=False):
                _curv_fp32 = _curv.float()
                import models.vlm_models.lorentz as L

                v_child_scaled_tangent = L.log_map0(verb_text_hyp.float(), _curv_fp32)
                o_child_scaled_tangent = L.log_map0(obj_text_hyp.float(), _curv_fp32)

                all_coarse_v_scaled = torch.zeros(
                    (len(unique_coarse_verbs), v_child_scaled_tangent.shape[-1]),
                    device=v_child_scaled_tangent.device,
                    dtype=v_child_scaled_tangent.dtype,
                )
                all_coarse_o_scaled = torch.zeros(
                    (len(unique_coarse_objs), o_child_scaled_tangent.shape[-1]),
                    device=o_child_scaled_tangent.device,
                    dtype=o_child_scaled_tangent.dtype,
                )

                all_coarse_v_scaled.index_add_(0, fine2coarse_v_idx, v_child_scaled_tangent)
                all_coarse_o_scaled.index_add_(0, fine2coarse_o_idx, o_child_scaled_tangent)

                all_coarse_v_scaled = all_coarse_v_scaled / coarse_v_counts.to(all_coarse_v_scaled.device).unsqueeze(1)
                all_coarse_o_scaled = all_coarse_o_scaled / coarse_o_counts.to(all_coarse_o_scaled.device).unsqueeze(1)

                all_coarse_v_hyp = L.exp_map0(all_coarse_v_scaled, _curv_fp32)
                all_coarse_o_hyp = L.exp_map0(all_coarse_o_scaled, _curv_fp32)

                w_att_obj = getattr(config, 'att_obj_w', 0.2)
                w_entail = getattr(config, 'lambda_entail', getattr(config, 'w_entail', 0.1))
                w_align = getattr(config, 'lambda_align', getattr(config, 'w_align', 1.0))
                align_margin = getattr(config, 'align_margin', 0.2)
                entail_margin = getattr(config, 'entail_margin', 0.01)

                entail_loss_fn = EntailmentConeLoss(margin=entail_margin)
                align_loss_fn = HyperbolicHardNegativeAlignmentLoss(margin=align_margin)

                # ==============================================================
                # 【终极修正：强化温度截断】
                # 防止在训练期间，双曲空间的负距离导致的 logit 被指数级放大，
                # 将最大值牢牢锁在 4.6052 以内 (即 scale 最大为 100)
                # ==============================================================
                actual_model.logit_scale_v.data = torch.clamp(actual_model.logit_scale_v.data, max=4.6052)
                actual_model.logit_scale_o.data = torch.clamp(actual_model.logit_scale_o.data, max=4.6052)

                scale_v = actual_model.logit_scale_v.exp()
                scale_o = actual_model.logit_scale_o.exp()

                verb_logits_scaled = verb_logits_hyp * scale_v
                obj_logits_scaled = obj_logits_hyp * scale_o

                train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]
                pred_com_train = verb_logits_scaled[:, train_v_inds] + obj_logits_scaled[:, train_o_inds]

                loss_com = Loss_fn(pred_com_train, batch_target)
                loss_verb_hyp = Loss_fn(verb_logits_scaled, batch_verb)
                loss_obj_hyp = Loss_fn(obj_logits_scaled, batch_obj)

                batch_v_mask = verb_hard_mask_matrix[batch_verb]
                batch_o_mask = obj_hard_mask_matrix[batch_obj]

                loss_align_v = align_loss_fn(verb_logits_hyp, batch_verb, hard_mask=batch_v_mask)
                loss_align_o = align_loss_fn(obj_logits_hyp, batch_obj, hard_mask=batch_o_mask)
                loss_alignment = loss_align_v + loss_align_o

                loss_ent_v = entail_loss_fn(verb_text_hyp.float(), all_coarse_v_hyp[fine2coarse_v_idx], _curv_fp32)
                loss_ent_o = entail_loss_fn(obj_text_hyp.float(), all_coarse_o_hyp[fine2coarse_o_idx], _curv_fp32)

                batch_fine_v_hyp = verb_text_hyp[batch_verb].float()
                batch_fine_o_hyp = obj_text_hyp[batch_obj].float()
                loss_ent_vid_v = entail_loss_fn(v_feat_hyp.float(), batch_fine_v_hyp, _curv_fp32)
                loss_ent_vid_o = entail_loss_fn(o_feat_hyp.float(), batch_fine_o_hyp, _curv_fp32)

                loss_entailment = loss_ent_v + loss_ent_o + loss_ent_vid_v + loss_ent_vid_o

                loss = loss_com + w_att_obj * (loss_verb_hyp + loss_obj_hyp) + w_entail * loss_entailment + w_align * loss_alignment

                loss = torch.nan_to_num(loss) / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            epoch_com_losses.append(loss_com.item())
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