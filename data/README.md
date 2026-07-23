# Datasets

## Bundled example

`W-I-S1.zip` contains the ICEWS-WIKI (split S1) dataset. Unzip it before running:

```bash
unzip data/W-I-S1.zip -d data/
```

## Full dataset suite

The six datasets (ICEWS-WIKI S1/S2, ICEWS-YAGO S1/S2, DrugBank-WIKI, UMLS-WIKI, OpenAlex-DBpedia, Crunchbase-DBpedia) follow the DKGF-Bench release. Download links will be provided here; in the meantime, please contact the authors for access.

## File format

Each dataset folder contains:

| File | Description |
| ---- | ----------- |
| `triples_1` | Relation triples of KG1 (GKG side), tab-separated IDs `h\tr\tt` |
| `triples_2` | Relation triples of KG2 (DKG side) |
| `ent_ids_1` / `ent_ids_2` | Entity ID mappings (`id\tname`) of KG1 / KG2 |
| `rel_ids_1` / `rel_ids_2` | Relation ID mappings of KG1 / KG2 |
| `sup_triples_1_fusion` | Fusion supervision triples used for training |
| `ref_triples_1_fusion` | Ground-truth fused triples (positives) reserved for testing |
| `false_ref_triples_1_fusion` | Hard negative triples for testing |
| `ref_relative_ent_ids_2` | Ground-truth relevant entities in KG2 |
| `sup_relative_ent_ids_2` / `sum_relative_ent_ids_2` | Relevant-entity supervision / full summary lists |
| `time_id` | Time IDs (for temporal KGs) |

## Long-tail buckets

Degree-bucketed test files can be regenerated from the raw data:

```bash
python tools/compute_node_degrees.py --base_dir data --datasets W-I-S1
python tools/extract_long_tail.py --base_dir data --datasets W-I-S1 \
    --output_dir outputs/long_tail_data --extract_by_degree
```

This produces, per dataset, `long_tail_deg_{0..4}_{pos,neg}.txt`, `long_tail_all_{pos,neg}.txt`, and `non_long_tail_{pos,neg}.txt` under the output directory, where an entity is long-tail if `d_min = min(deg(h), deg(t)) <= 4`.
