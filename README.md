# CellPFP prediction

Run the pretrained CellPFP ensemble for Biological Process (`bp`), Molecular Function (`mf`), or Cellular Component (`cc`). Each score is the arithmetic mean of the sigmoid probabilities from the three checkpoints for that ontology.

## Setup

Use Python 3.8 and install the pinned dependencies.

```bash
python -m pip install -r requirements.txt
```

## Weights and features

Download the `data/` and `models/` folders using extraction code **`ayhb`**. [Baidu Netdisk download](https://pan.baidu.com/s/12wpzP4U_w-nw2OJgbz4ZLA?pwd=ayhb)

After downloading, place both folders in the repository root alongside `predict.py`. Keep the folder names and their internal structure so the examples can use `--data-dir data --models-dir models`.

Each ontology uses three inference checkpoints, `<ontology>_0.pt`, `<ontology>_1.pt`, and `<ontology>_2.pt`. They contain the trained parameters, GO term order, and inference settings.

The feature directory passed to `--data-dir` has this layout.

```text
data/
  atgo_interpro_features.pt
  atgo_interpro_row_mapping.json
  global_node_mapping.pkl
  cell_name_mapping.txt
  protein_cell_expression.txt
  protein_degree_in_cells.txt
  156_dgl_graphs_string_weighted/
    <cell_name>.dgl
  esm2_embeddings/
    <gene_symbol>.pt
```


## Run prediction

List one human gene symbol or `GENE_HUMAN` identifier per line. `example_proteins.txt` contains two proteins with precomputed features. From the release directory, run the following command.

```bash
python predict.py -d mf --proteins example_proteins.txt --data-dir data --models-dir models --output outputs/mf.tsv
```

The output columns are `protein_id`, `ontology`, `go_id`, and `score`. By default, the script writes the 20 highest scores for each protein. Use `--top-k 0` for every GO term in the checkpoint vocabulary. The same input file works for `-d bp` and `-d cc` with the corresponding weights.

To use a separate directory of compatible precomputed ESM2 tensors, pass `--embeddings-dir`. For proteins with an embedding and zero available PPI contexts, the trained model uses its sequence branch. The entry point accepts IDs with precomputed features. Starting from raw FASTA requires a separate ESM2 feature extraction step consistent with the training inputs.


## Full test-set evaluation

```bash
python predict.py -d mf --evaluate --data-dir data --models-dir models
```

Use `-d bp`, `-d mf`, or `-d cc` to select the ontology. The evaluator reads `data/<ontology>_test2_pid_list.txt`, `data/<ontology>_test2_go.txt`, and `data/go.obo`. It evaluates every GO term in the checkpoint vocabulary. The original evaluation protocol propagates predictions and labels to GO ancestors, excludes the namespace root, and scans thresholds from 0.00 to 1.00 in increments of 0.01.

The terminal prints Fmax and AUPR. Keep the default batch size of 8 for comparison with the training runs. Sampled neighborhoods and hardware can affect exact scores.

