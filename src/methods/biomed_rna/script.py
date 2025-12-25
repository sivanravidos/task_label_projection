import os
import subprocess
import anndata as ad
import pandas as pd
import numpy as np
import sys
import torch
import gc
from omegaconf import OmegaConf
from pathlib import Path
import bmfm_targets
from hydra import compose, initialize_config_dir
from scipy import sparse
from sklearn.model_selection import train_test_split
from bmfm_targets.tasks.scbert.scbert_main import main


def refactor_adata(adata):
    """
    Process adata file and refactor to format used by biomed-rna:
    - Move counts to X
    - Move gene names to var_name
    - Save as sparse 
    """
    processed_adata = adata.copy()
    if "counts" in processed_adata.layers:
        processed_adata.X = processed_adata.layers["counts"].copy()
    else:
        raise ValueError("layers['counts'] not found — cannot set X")
    if "feature_name" in processed_adata.var.columns:
        if processed_adata.var.shape[0] != processed_adata.X.shape[1]:
            raise ValueError(
                f"var has {processed_adata.var.shape[0]} rows but X has {processed_adata.X.shape[1]} columns"
            )
        processed_adata.var_names = processed_adata.var["feature_name"].astype(str)
        processed_adata.var_names_make_unique()
    else:
        raise ValueError("var['feature_name'] not found — cannot set var_names")

    if not sparse.issparse(processed_adata.X):
        processed_adata.X = sparse.csr_matrix(processed_adata.X)

    return processed_adata



## VIASH START
par = {
    'input_train': 'resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad',
    'input_test': 'resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad',
    'output': 'output.h5ad',
}
meta = {
    'name': 'biomed_rna'
}
## VIASH END


#verify home
print(f"HOME: {os.environ.get('HOME')}")
print(f"USER: {os.environ.get('USER')}")

# verify gpu
print(f"torch.cuda.is_available():{torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")


# get config files folder path
package_location = Path(bmfm_targets.__file__).parent
print(f"Package location: {package_location}")
BIO_MED_ROOT = str(package_location.parent / "run")


print("Load input and test data", flush=True)
input_train = ad.read_h5ad(par['input_train'])
input_test = ad.read_h5ad(par['input_test'])

print(f"input train shape: {input_train.shape}")
print(f"input test shape: {input_test.shape}")

print("Splitting train into train/dev", flush=True)
train_idx, dev_idx = train_test_split(
    np.arange(len(input_train)),
    test_size=0.1,
    shuffle=True,
    random_state=42
)
input_train.obs['split'] = 'train'
input_train.obs.loc[input_train.obs.index[dev_idx], 'split'] = 'dev'
# print("\ntrain file split counts:")
# print(input_train.obs["split"].value_counts())
# print(input_train.obs.columns)
# print("\train file cell_type counts:")
# print(input_train.obs["label"].value_counts())

print("Processing train data, saving to input_train_processed.h5ad", flush=True)
processed_train = refactor_adata(input_train)
processed_train.write_h5ad("input_train_processed.h5ad")

print("Processing test data, saving to input_test_processed.h5ad", flush=True)
processed_test = refactor_adata(input_test)
processed_test.write_h5ad("input_test_processed.h5ad")

print("Check contents of workspace folder")
for item in Path("/workspace").iterdir():
    print(item)


print("Train biomed-rna model", flush=True)

with initialize_config_dir(
    version_base="1.2",
    config_dir=BIO_MED_ROOT,
):
    cfg = compose(
        config_name="finetune",
        overrides=[
            "task=train",
            "label_column_name=label",
            "split_column_name=split",
            "input_file=input_train_processed.h5ad",
            "working_dir=.",
            # "++data_module.rda_transform=auto_align",
            # "data_module.log_normalize_transform=false",
            "data_module.max_length=16",
            "data_module.batch_size=16",
            "data_module.num_workers=8",
            "data_module.collation_strategy=multitask",
            "trainer.losses.0.name=focal",
            "checkpoint=ibm-research/biomed.rna.bert.110m.mlm.rda.v1",
            "max_epochs=1",
            # "task.max_steps=2",
            "accelerator=gpu",
            "task.precision=16-mixed",
            "+track_clearml.project_name=bmfm_targets/open_problems",
            "+track_clearml.task_name=label_projection_immune_cell_atlas_test",
            "+checkpoints_every_n_train_steps=1"
        ],
    )

print("--- train config ---")
print(OmegaConf.to_yaml(cfg, resolve=True))

main(cfg)
print("Created checkpoints: ")
for item in Path("/workspace").iterdir():
    if item.is_file() and item.suffix == ".ckpt":
        print(item)

print("Using model to generate prediction on test data", flush=True)

with initialize_config_dir(
    version_base="1.2",
    config_dir=BIO_MED_ROOT,
):
    cfg = compose(
        config_name="predict",
        overrides=[
            "target_column_name=label",
            f"input_file=input_test_processed.h5ad",
            "working_dir=.",
            # "++data_module.rda_transform=auto_align",
            # "data_module.log_normalize_transform=false",
            "data_module.max_length=16",
            "data_module.batch_size=16",
            "data_module.num_workers=8",
           # "+data_module.limit_dataset_samples=100",
            "data_module.collation_strategy=multitask",
            "checkpoint=/workspace/last.ckpt",
            "accelerator=gpu",
            "task.precision=16-mixed",
            "+track_clearml.project_name=bmfm_targets/open_problems",
            "+track_clearml.task_name=label_projection_immune_cell_atlas_test",
        ],
    )

print("--- test config ---")
print(OmegaConf.to_yaml(cfg, resolve=True))
main(cfg)


predictions = pd.read_csv("predictions.csv")
# predictions.csv columns are "Unamed: 0, label
predictions = predictions.rename(columns={"Unnamed: 0": "cell_id", "label": "label_pred"}).set_index("cell_id")
output_adata = ad.AnnData(
    obs=predictions,
    var=input_test.var[[]],
        uns={
            "dataset_id": input_test.uns["dataset_id"],
            "normalization_id": input_test.uns["normalization_id"],
            "method_id": meta["name"],
        },
    )
print(f"Writing predictions to {par['output']}", flush=True)
output_adata.write_h5ad(par["output"])

print("BIOMED-RNA done, cleaning up", flush=True) 
torch.cuda.empty_cache()
gc.collect()
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()

print("Finished clean up,  exiting", flush=True) # to speed up exit
os._exit(0)
