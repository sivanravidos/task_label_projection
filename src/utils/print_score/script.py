import anndata as ad

## VIASH START
par = {
    'input_score': 'resources_test/task_label_projection/cxg_immune_cell_atlas/score.h5ad',
}
meta = {
    'name': 'print_score'
}
## VIASH END
adata = ad.read_h5ad(par['input_score'])
print(f"dataset_id: {adata.uns['dataset_id']}")
print(f"method_id: {adata.uns['method_id']}")
print(f"normalization_id: {adata.uns['normalization_id']}")

for m, v in zip(adata.uns["metric_ids"], adata.uns["metric_values"]):
    print(f"{m}: {v:.3f}")
