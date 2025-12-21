import os
import subprocess
import anndata as ad
import pandas as pd
import numpy as np
import sys
from pathlib import Path


from hydra import initialize, compose, initialize_config_dir

from scipy import sparse
from sklearn.model_selection import train_test_split

from bmfm_targets.tasks.scbert.scbert_main import main

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

# verify gpu
import torch
print(f"torch.cuda.is_available():{torch.cuda.is_available()}")
print(f"torch.backends.mps.is_available():{torch.backends.mps.is_available()}")



import bmfm_targets

# Find the installed package location
package_location = Path(bmfm_targets.__file__).parent
print(f"Package location: {package_location}")
print(f"Package parent: {package_location.parent}")

# Check parent directory
run_in_parent = package_location.parent / "run"
print(f"run folder in parent: {run_in_parent.exists()}")

BIO_MED_ROOT = str(package_location.parent / "run")

print("Load input data", flush=True)
input_train = ad.read_h5ad(par['input_train'])
input_test = ad.read_h5ad(par['input_test'])
print(input_train.obs.columns)
print(input_test.obs.columns)

print("Splitting train into train/dev", flush=True)
train_idx, dev_idx = train_test_split(
    np.arange(len(input_train)),
    test_size=0.1,
    shuffle=True,
    random_state=42
)
input_train.obs['split'] = 'train'
input_train.obs.loc[input_train.obs.index[dev_idx], 'split'] = 'dev'
input_test.obs['split'] = 'test'

if "counts" in input_train.layers:
    print("Moving layers['counts'] → .X", flush=True)
    input_train.X = input_train.layers["counts"].copy()
if "counts" in input_test.layers:
    print("Moving layers['counts'] → .X", flush=True)
    input_test.X = input_test.layers["counts"].copy()

if "feature_name" in input_train.var.columns:
    if input_train.var.shape[0] != input_train.X.shape[1]:
        raise ValueError(
            f"var has {input_train.var.shape[0]} rows but X has {input_train.X.shape[1]} columns"
        )

    input_train.var_names = input_train.var["feature_name"].astype(str)
    input_train.var_names_make_unique()
else:
    raise ValueError("var['feature_name'] not found — cannot set var_names")

if "feature_name" in input_test.var.columns:
    if input_test.var.shape[0] != input_test.X.shape[1]:
        raise ValueError(
            f"var has {input_test.var.shape[0]} rows but X has {input_test.X.shape[1]} columns"
        )

    input_test.var_names = input_test.var["feature_name"].astype(str)
    input_test.var_names_make_unique()
else:
    raise ValueError("var['feature_name'] not found — cannot set var_names")

print("\nSplit counts:")
print(input_train.obs["split"].value_counts())
print(input_train.obs.columns)
print("\cell_type counts:")
print(input_train.obs["label"].value_counts())
print("Writing unified datasets to datasets_unified.h5ad", flush=True)
if not sparse.issparse(input_train.X):
    input_train.X = sparse.csr_matrix(input_train.X)
input_train.write_h5ad("dataset_unified.h5ad", compression="gzip")

print("Check contents of workspace folder")
for item in Path("/workspace").iterdir():
    print(item)

print("Create bmfm-rna model", flush=True)

print("Train model", flush=True)


with initialize_config_dir(
    version_base="1.2",
    config_dir=BIO_MED_ROOT,
):
    cfg = compose(
        config_name="finetune",
        overrides=[
            "label_column_name=label",
            "split_column_name=split",
            f"input_file=dataset_unified.h5ad",
            "working_dir=.",
            # "++data_module.rda_transform=auto_align",
            # "data_module.log_normalize_transform=false",
            "data_module.max_length=16",
            "checkpoint=ibm-research/biomed.rna.bert.110m.mlm.rda.v1",
            #"trainer.losses[0].name=focal",
            "data_module.batch_size=1",
            "data_module.num_workers=4",
            "max_epochs=1",
            "accelerator=cpu",
            "+track_clearml.project_name=bmfm_targets/open_problems",
            "+track_clearml.task_name=label_projection_immune_cell_atlas_test",

        ],
    )

main(cfg)
print("Check contents of workspace folder")
for item in Path("/workspace").iterdir():
    print(item)

print("Make predictions", flush=True)

with initialize_config_dir(
    version_base="1.2",
    config_dir=BIO_MED_ROOT,
):
    cfg = compose(
        config_name="predict",
        overrides=[
            "label_column_name=label",
            "split_column_name=split",
            f"input_file=unified_dataset.h5ad",
            "working_dir=.",
            # "++data_module.rda_transform=auto_align",
            # "data_module.log_normalize_transform=false",
            "data_module.max_length=16",
            "checkpoint=ibm-research/biomed.rna.bert.110m.mlm.rda.v1",
            "accelerator=cpu"

        ],
    )

main(cfg)

# preds = 

# print("Create SCANVI model and train it on fully labelled reference dataset", flush=True)
# sca.models.SCVI.setup_anndata(
#     adata, 
#     batch_key="batch", 
#     labels_key="label",
#     layer="counts"
# )

# vae = sca.models.SCVI(
#     adata,
#     n_layers=2,
#     encode_covariates=True,
#     deeply_inject_covariates=False,
#     use_layer_norm="both",
#     use_batch_norm="none",
# )

# print("Create the SCANVI model instance with ZINB loss", flush=True)
# scanvae = sca.models.SCANVI.from_scvi_model(vae, unlabeled_category = "Unknown")

# print("Train SCANVI model", flush=True)
# scanvae.train()

# print("Make predictions", flush=True)
# preds = scanvae.predict(adata)

# print("Store outputs", flush=True)
# output = ad.AnnData(
#     obs=pd.DataFrame(
#         {"label_pred": preds[adata.obs['is_test'].values]},
#         index=input_test.obs.index,
#     ),
#     var=input_test.var[[]],
#     uns={
#         "dataset_id": input_test.uns["dataset_id"],
#         "normalization_id": input_test.uns["normalization_id"],
#         "method_id": meta["name"],
#     },
# )

# print("Write output to file", flush=True)
# output.write_h5ad(par["output"], compression="gzip")
