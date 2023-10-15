import os

from fairseq import checkpoint_utils


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    if not hasattr(config, 'hubert_model_path'):
        hubert_model_path = "assets/hubert/hubert_base.pt"
    else:
        hubert_model_path = config.hubert_model_path

    print(f"Loading {hubert_model_path}")
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_model_path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
