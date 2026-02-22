from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AutonLab/MOMENT-1-large",
    local_dir="/moment/MOMENT-1-large",
    local_dir_use_symlinks=False,
)
