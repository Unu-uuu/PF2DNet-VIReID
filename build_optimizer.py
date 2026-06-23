import torch 
import torch.optim as optim

def build_optim(net, opt, lr):
    # 处理 DataParallel 包装的模型
    if hasattr(net, 'module'):
        model = net.module
    else:
        model = net
    
    
    adaptive_keywords = [
        'base_mix', 'adapt_scale', 'adapt_bias',  # AdaptiveSpectrumFilter
        # 'size_scale', 'fusion_weight', 'mix_weight',  # 其他自适应参数
        'size_scale', 'fusion_weight', 'mix_ratio',  # 其他自适应参数（mix_ratio 在 FreqEnhancedNAFBlock 中）
        'alpha',  # FrequencyDenoisedCompensation 中的 alpha
        # 注意：'delta' 和 'sin' 是你原来就有的，保持原来的学习率
    ]
    
    shallow_optimizer = None  # 默认值
    
    if opt == 'ADM':
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            
            lr_temp = lr * 0.1
            weight_decay = 1e-4
            
            # 偏置项
            if "bias" in key:
                lr_temp = lr_temp * 2
                weight_decay = 0.0
            
            # bottleneck 和 classifier 用完整学习率
            if "bottleneck" in key or "classifier" in key:
                lr_temp = lr
            
            # ========== 新增：自适应参数用更小的学习率 ==========
            if any(kw in key for kw in adaptive_keywords):
                lr_temp = lr * 0.01  # 1/100 的学习率，非常保守
                weight_decay = 1e-5  # 更小的 weight_decay
            params += [{"params": [value], "lr": lr_temp, "weight_decay": weight_decay}]

        optimizer = optim.Adam(
            params,        
            betas=(0.9, 0.999),  
            eps=1e-3,
        )
        
    elif opt == 'SGD':
        # 收集不同类型的参数
        bottleneck_params = list(model.bottleneck.parameters())
        classifier_params = list(model.classifier.parameters())
        
        ignored_ids = list(map(id, bottleneck_params)) + list(map(id, classifier_params))
        
        # 分离自适应参数和普通基础参数
        adaptive_params = []
        base_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in ignored_ids:
                continue  # 跳过 bottleneck 和 classifier
            
            if any(kw in name for kw in adaptive_keywords):
                adaptive_params.append(param)
            else:
                base_params.append(param)
        
        
        # 构建参数组
        param_groups = [
            {'params': base_params, 'lr': 0.1 * lr},
            {'params': bottleneck_params, 'lr': lr},
            {'params': classifier_params, 'lr': lr},
        ]
        
        # 如果有自适应参数，添加到参数组
        if len(adaptive_params) > 0:
            param_groups.append({
                'params': adaptive_params, 
                'lr': 0.01 * lr,  # 自适应参数用 1/100 学习率
                'weight_decay': 1e-5
            })
        
        deep_optimizer = optim.SGD(
            param_groups,
            weight_decay=5e-4, 
            momentum=0.9, 
            nesterov=True
        )

        # shallow_optimizer 只优化 base_params（保持你原来的逻辑）
        shallow_optimizer = optim.SGD(
            [{'params': base_params, 'lr': 0.1 * lr}],
            weight_decay=5e-4, 
            momentum=0.9, 
            nesterov=True
        )
        
        
        return deep_optimizer, shallow_optimizer

    elif opt == 'ADM_ORI':
        ignored_params = list(map(id, model.bottleneck.parameters())) \
                         + list(map(id, model.classifier.parameters()))
    
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optimizer = optim.Adam(
            [
                {'params': base_params, 'lr': 0.1 * lr, "weight_decay": 0.00004},
                {'params': model.bottleneck.parameters(), 'lr': lr, "weight_decay": 0.0},
                {'params': model.classifier.parameters(), 'lr': lr, "weight_decay": 0.0}
            ],        
            betas=(0.9, 0.999),  
            eps=1e-8,
        )
    
    return optimizer, shallow_optimizer