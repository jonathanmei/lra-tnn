optimizer = {
    "adam": "torch.optim.Adam",
    "adamw": "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd": "torch.optim.SGD",
    "lamb": "src.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant": "transformers.get_constant_schedule",
    "plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step": "torch.optim.lr_scheduler.StepLR",
    "multistep": "torch.optim.lr_scheduler.MultiStepLR",
    "cosine": "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup": "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
    "timm_cosine": "src.utils.optim.schedulers.TimmCosineLRScheduler",
}

model = {
    "transformer": "src.models.baselines.transformer.Transformer",
    "model": "src.models.sequence.SequenceModel",
}

layer = {
    "id": "src.models.sequence.base.SequenceIdentity",
    "glu": "src.models.sequence.glu.GLU",
    "mha": "src.models.sequence.mha.MultiheadAttention",
    # tnn_draft
    "tno": "src.models.sequence.tnn_draft.tno.TNO",
    "tno2d": "src.models.sequence.tnn_draft.tno2d.TNO",
    "skitno": "src.models.sequence.tnn_draft.skitno.SKITNO",
    "skitno2d": "src.models.sequence.tnn_draft.skitno2d.SKITNO2d",
    "tno_fd": "src.models.sequence.tnn_draft.tno_fd.TNO",
    "tno_fd2d": "src.models.sequence.tnn_draft.tno_fd2d.TNO",
    # tnn
    "gtu": "src.models.sequence.tnn.gtu.Gtu",
    "gtu2d": "src.models.sequence.tnn.gtu2d.Gtu2d",
}

callbacks = {
    "timer": "src.callbacks.timer.Timer",
    "params": "src.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint": "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping": "pytorch_lightning.callbacks.EarlyStopping",
    "swa": "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary": "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar": "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_learning": "src.callbacks.progressive_learning.ProgressiveLearning",
}

layer_decay = {
    "convnext_timm_tiny": "src.models.baselines.convnext_timm.get_num_layer_for_convnext_tiny",
}

model_state_hook = {
    "convnext_timm_tiny_2d_to_3d": "src.models.baselines.convnext_timm.convnext_timm_tiny_2d_to_3d",
    "convnext_timm_tiny_s4nd_2d_to_3d": "src.models.baselines.convnext_timm.convnext_timm_tiny_s4nd_2d_to_3d",
}
