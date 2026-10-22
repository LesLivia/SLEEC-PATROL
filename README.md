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

## SDD-OIA Verification Notes

- **No yellow-light concept.** SDD-OIA renders yellow traffic lights, but the rsbench generator labels them `red_light` (`deal_with_tl` in `rssgen/rssgen/sddoia/sddoia.py` of [rsbench](https://github.com/unitn-sml/rsbench-code)), and their action label is stop, as for red. We keep this definition (no relabelling, no retraining), so `redLight` in `resources/requirements/sddoia_behaviourplanner.sleec` means "red or yellow light ahead". AutoCar R14 (yellow light) is therefore not adapted: its main part would repeat R13, and its "unless the yellow follows a red" part needs the previous light, which the single-frame model does not see.
- **No relation block.** The head's inputs are the network's concept predictions, and the only fact about them is the [0, 1] box. The generator's concept relations hold for the ground truth, not for the predictions: e.g. `follow` and `car` never hold together in a scene, yet both are predicted above 0.5 in 104 of the 1,464 test frames.
