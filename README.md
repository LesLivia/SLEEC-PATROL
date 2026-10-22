# SLEEC to Marabou Verification Tool

This project provides a tool for verifying SLEEC (social, legal, ethical, empathetic, and cultural) requirements against Neural Network models.

## Environment Setup

To recreate the Conda environment required for this project, follow these steps:

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository-url>
    cd SLEEC2Ver
    ```

2.  **Create the Conda environment** using the provided `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment**:
    ```bash
    conda activate ai-sleec-ver
    ```

4.  **Optional extras** (install inside the activated environment):
    - CBM training (`autonomous_driving/CBM/rss`):
      ```bash
      pip install -r requirements-cbm.txt
      ```
    - The LEGOS web server (`legos_sleec/analyzer/routes.py`):
      ```bash
      pip install -r requirements-legos.txt
      ```

## Running the Web App

The web app in `webapp/` is the tool's frontend. It takes a SLEEC specification (`.sleec`) and a neural network (`.onnx`), turns the rules into Marabou queries, and reports for each rule whether the network complies with it.

1.  **Start the server** from the repository root, with the environment activated:
    ```bash
    python -m uvicorn webapp.app:app --port 8000
    ```
    Marabou has no Windows build, so on Windows run this inside WSL.

2.  **Open** http://localhost:8000 in a browser.

**Inputs.** The network's inputs must be the specification's measures, in the order they are declared, and its outputs must be the specification's output events. The specification language is defined in `legos_sleec/sleec-grammar.txt`.

**Example setups.** Each folder in `webapp/examples/` that holds exactly one `.sleec` file and one `.onnx` file is offered as a ready-made setup. Adding such a folder adds a setup.

## Training the Highway Policy

The highway behaviour planner is a PPO policy trained in highway-env. The environment is chosen by `ENV_ID` at the top of each script (`highway-fast-v0` by default). Run the commands from the repository root.

1.  **Train the policy**:
    ```bash
    python autonomous_driving/highway/train_highway_ppo.py
    ```
    The models are saved in `autonomous_driving/highway/saved_models/<ENV_ID>/`. `best_model.zip`, the better of the last and the best policy on 50 fixed episodes, is the one the export uses.

2.  **Export the actor to ONNX** for verification:
    ```bash
    python autonomous_driving/highway/export_to_onnx.py
    ```
    This writes `verifier/resources/models/ppo_<ENV_ID>.onnx`.

## Training the SDD-OIA CBM

The SDD-OIA concept bottleneck model is trained on image features that are extracted once from the rsbench SDD-OIA images. Run every command from `autonomous_driving/CBM/rss`, with the CBM extras installed (step 4 of Environment Setup).

1.  **Download the data**: `sdd-oia.zip` from the rsbench dataset record on Zenodo (https://zenodo.org/records/11612556).

2.  **Unzip it into `data/`**, which gives `data/mini_boia_out/` (the `train`, `val`, `test` and `ood` image folders, and `scenes/`):
    ```bash
    unzip sdd-oia.zip -d data/
    ```

3.  **Extract the features** with each backbone. The first run of a backbone downloads its weights. A finished feature folder is never overwritten; to redo an extraction, delete its folder first.
    ```bash
    python -m utils.extract_features_sdd_oia --model vit_large_patch16_dinov3.lvd1689m --out data/sddoia_dinov3_l_384 --height 384 --width 464
    python -m utils.extract_features_sdd_oia --model vit_pe_spatial_large_patch14_448.fb --out data/sddoia_pe_spatial_l_378 --height 378 --width 462
    ```

4.  **Combine the two feature sets** into one folder:
    ```bash
    python -m utils.combine_features_sdd_oia --inputs sddoia_dinov3_l_384 sddoia_pe_spatial_l_378 --out sddoia_dinov3_l_pe_spatial
    ```

5.  **Train the model** on the combined features:
    ```bash
    python main.py --dataset presddoia --model presddoiacbm --task sddoia --c_sup 1 --seed 42 --n_epochs 200 --features sddoia_dinov3_l_pe_spatial --concept_hidden 4096 --lr 0.0001 --head_lr 0.001 --c_weight sqrt --checkout
    ```
    The best model is saved as `best_model_presddoia_presddoiacbm_42_partial_sup_sddoia_dinov3_l_pe_spatial_h4096_cw_sqrt.pth`, and, with `--checkout`, also as `data/ckpts/presddoia-presddoiacbm-dis-42-sddoia_dinov3_l_pe_spatial_h4096_cw_sqrt-end.pt`.

6.  **Export the action layer to ONNX** for verification:
    ```bash
    python -m utils.export_head_sdd_oia --checkpoint data/ckpts/presddoia-presddoiacbm-dis-42-sddoia_dinov3_l_pe_spatial_h4096_cw_sqrt-end.pt
    ```
    This writes `verifier/resources/models/sddoia_cbm_head.onnx`.

## SDD-OIA Verification Notes

- **No yellow-light concept.** SDD-OIA renders yellow traffic lights, but the rsbench generator labels them `red_light` (`deal_with_tl` in `rssgen/rssgen/sddoia/sddoia.py` of [rsbench](https://github.com/unitn-sml/rsbench-code)), and their action label is stop, as for red. We keep this definition (no relabelling, no retraining), so `redLight` in `resources/requirements/sddoia_behaviourplanner.sleec` means "red or yellow light ahead". AutoCar R14 (yellow light) is therefore not adapted: its main part would repeat R13, and its "unless the yellow follows a red" part needs the previous light, which the single-frame model does not see.
- **No relation block.** The head's inputs are the network's concept predictions, and the only fact about them is the [0, 1] box. The generator's concept relations hold for the ground truth, not for the predictions: e.g. `follow` and `car` never hold together in a scene, yet both are predicted above 0.5 in 104 of the 1,464 test frames.
