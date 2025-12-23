import os
import subprocess
import anndata as ad
import pandas as pd
import numpy as np
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from scipy import sparse
from sklearn.model_selection import train_test_split

from bmfm_targets.tasks.scbert.scbert_main import main


def refactor_adata(adata):

    if "counts" in adata.layers:
        print("Moving layers['counts'] → .X", flush=True)
        adata.X = adata.layers["counts"].copy()
    if "feature_name" in adata.var.columns:
        if adata.var.shape[0] != adata.X.shape[1]:
            raise ValueError(
                f"var has {adata.var.shape[0]} rows but X has {adata.X.shape[1]} columns"
            )

        adata.var_names = adata.var["feature_name"].astype(str)
        adata.var_names_make_unique()
    else:
        raise ValueError("var['feature_name'] not found — cannot set var_names")


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
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")


import bmfm_targets

# Find the installed package location
package_location = Path(bmfm_targets.__file__).parent
print(f"Package location: {package_location}")
print(f"Package parent: {package_location.parent}")

# run_in_parent = package_location.parent / "run"
# print(f"run folder in parent: {run_in_parent.exists()}")

BIO_MED_ROOT = str(package_location.parent / "run")

print("Load input data", flush=True)
input_train = ad.read_h5ad(par['input_train'])
input_test = ad.read_h5ad(par['input_test'])
# print(input_train.obs.columns)
# print(input_test.obs.columns)

print("Splitting train into train/dev", flush=True)
train_idx, dev_idx = train_test_split(
    np.arange(len(input_train)),
    test_size=0.1,
    shuffle=True,
    random_state=42
)
input_train.obs['split'] = 'train'
input_train.obs.loc[input_train.obs.index[dev_idx], 'split'] = 'dev'


refactor_adata(input_train)
refactor_adata(input_test)

print("\ntrain file split counts:")
print(input_train.obs["split"].value_counts())
print(input_train.obs.columns)
print("\train file cell_type counts:")
print(input_train.obs["label"].value_counts())

print("Writing processed train to input_train_processed.h5ad", flush=True)
if not sparse.issparse(input_train.X):
    input_train.X = sparse.csr_matrix(input_train.X)
input_train.write_h5ad("input_train_processed.h5ad", compression="gzip")

print("Writing processed test to input_test_processed.h5ad", flush=True)
if not sparse.issparse(input_test.X):
    input_test.X = sparse.csr_matrix(input_test.X)
input_test.write_h5ad("input_test_processed.h5ad", compression="gzip")

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
            "task=train",
            "label_column_name=label",
            "split_column_name=split",
            "input_file=input_train_processed.h5ad",
            "working_dir=.",
            # "++data_module.rda_transform=auto_align",
            # "data_module.log_normalize_transform=false",
            "data_module.max_length=16",
            "data_module.batch_size=1",
            "data_module.num_workers=4",
            "data_module.collation_strategy=multitask",
            "trainer.losses.0.name=focal",
            "checkpoint=ibm-research/biomed.rna.bert.110m.mlm.rda.v1",
            "max_epochs=1",
            "accelerator=cpu",
            "task.precision=32",
            "+track_clearml.project_name=bmfm_targets/open_problems",
            "+track_clearml.task_name=label_projection_immune_cell_atlas_test",
            "+checkpoints_every_n_train_steps=null"


        ],
    )

main(cfg)
# print("Check contents of workspace folder")
# for item in Path("/workspace").iterdir():
#     print(item)

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
            "data_module.batch_size=1",
            "data_module.num_workers=4",
            "data_module.collation_strategy=multitask",
            "checkpoint=/workspace/last.ckpt",
            "accelerator=cpu",
            "task.precision=32",
            "+track_clearml.project_name=bmfm_targets/open_problems",
            "+track_clearml.task_name=label_projection_immune_cell_atlas_test",
        ],
    )

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

output_adata.write_h5ad(par["output"], compression="gzip")

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


