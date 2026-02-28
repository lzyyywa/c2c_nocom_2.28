import torch

def get_optimizer_vm(cfg,model):
    comp_param=[]
    video_en_param=[]
    for name, param in model.named_parameters():
        if 'video_encoder' in name:
            video_en_param.append(param)
        else:
            comp_param.append(param)
    optimizer = torch.optim.Adam([
        {'params': comp_param, 'lr': cfg.com_lr,'weight_decay': cfg.com_wd},
        {'params': video_en_param, 'lr': cfg.ve_lr,'weight_decay': cfg.ve_wd}],
        lr=cfg.ve_lr, eps=1e-8,weight_decay=cfg.ve_wd)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    return optimizer

def get_optimizer_vlm(cfg,model):
    vision_no_wd=[]
    vision_with_wd=[]

    prompt_param = []
    c2c_with_wd=[]
    
    # -------------------------------------------------------------
    # 新增：双曲专有参数组
    # -------------------------------------------------------------
    hyp_proj_params = []   # 双曲投影基底矩阵 (需要 weight decay)
    hyp_scale_params = []  # 曲率、Alpha、温度等标量参数 (不需要 weight decay)

    for name, param in model.named_parameters():
        if 'video_encoder' in name:
            if 'temporal_embedding' in name or 'ln_post' in name:
                vision_no_wd.append(param)
            elif 'Adapter' in name or 'clip_proj' in name:
                vision_with_wd.append(param)
                
    if cfg.text_encoding_manner=='composition':
        for name, param in model.named_parameters():
            if 'dfsp' in name:
                c2c_with_wd.append(param)
        optimizer = torch.optim.AdamW([
            {'params': model.prompt_learner.parameters(), 'lr': cfg.text_lr, 'weight_decay': cfg.text_wd},
            {'params': vision_with_wd, 'lr': cfg.visual_lr, 'weight_decay': cfg.visual_wd},
            {'params': vision_no_wd, 'lr': cfg.visual_lr, 'weight_decay': 0.0}, ],
            betas=(0.9, 0.999), lr=cfg.visual_lr, eps=1e-8,
            weight_decay=cfg.visual_wd)  # Params used from paper, the lr is
            
    elif cfg.text_encoding_manner=='component':
        for name, param in model.verb_prompt_learner.named_parameters():
            prompt_param.append(param)
        for name, param in model.obj_prompt_learner.named_parameters():
            if 'token_embedding' not in name:
                prompt_param.append(param)
                
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # 避免重复添加视觉和 prompt 参数
            if 'video_encoder' in name or 'prompt_learner' in name:
                continue

            # =========================================================
            # 精准分类：双曲参数 vs 欧式网络参数
            # =========================================================
            if 'hyp_proj' in name:
                hyp_proj_params.append(param)
            elif 'curv' in name or 'alpha' in name or 'logit_scale' in name:
                hyp_scale_params.append(param)
            elif 'c2c' in name:
                c2c_with_wd.append(param)
                
        # 核心：设定一个极小的双曲学习率 (建议为主学习率的 1/100)
        hyp_lr = cfg.visual_lr * 0.01
        
        optimizer = torch.optim.AdamW([
            {'params': prompt_param, 'lr': cfg.text_lr, 'weight_decay': cfg.text_wd},
            {'params': vision_with_wd, 'lr': cfg.visual_lr, 'weight_decay': cfg.visual_wd},
            {'params': vision_no_wd, 'lr': cfg.visual_lr, 'weight_decay': 0.0},

            {'params': c2c_with_wd, 'lr': cfg.visual_lr, 'weight_decay': cfg.visual_wd},
            
            # --- 引入极低学习率的双曲参数组 ---
            {'params': hyp_proj_params, 'lr': hyp_lr, 'weight_decay': cfg.visual_wd},
            {'params': hyp_scale_params, 'lr': hyp_lr, 'weight_decay': 0.0},
        ],
            betas=(0.9, 0.999), lr=cfg.visual_lr, eps=1e-8,
            weight_decay=cfg.visual_wd)  # Params used from paper, the lr is
    else:
        raise NotImplementedError
    return optimizer



def get_optimizer(cfg,model):
    if cfg.framework=='vm':
        return get_optimizer_vm(cfg,model)
    elif cfg.framework=='vlm':
        return get_optimizer_vlm(cfg,model)