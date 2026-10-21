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

## Running the Frontend Demo (Dressing Robot)

The `sleec2ver_frontend.py` provides a graphical user interface to load requirements, select NN models, and perform verification.

To run the dressing robot demo:

1.  **Launch the frontend**:
    ```bash
    python sleec2ver_frontend.py
    ```

2.  **Load the Dressing Robot requirements**:
    - In the **Requirements** dropdown menu at the top, select `dressassist_nn.sleec`.
    - The SLEEC specification will appear in the left editor pane.

3.  **Select the Model**:
    - In the **Model** dropdown menu, select `carebot_decision_model_v2.onnx`.

4.  **Perform Verification**:
    - Click the **VERIFY** button (in green) to check the loaded requirements against the selected NN model.
    - The results will be displayed in the **Output** panel and the right editor pane.

5.  **SMT Checks (Optional)**:
    - You can also perform standard SLEEC SMT checks (redundancy, conflicts, concern, purpose, situational conflict) by clicking the respective buttons.

## SDD-OIA Scene Rendering Setup

SDD-OIA scenes are rendered with the rsbench generator in `autonomous_driving/CBM/rssgen`. It runs inside Blender 2.91, which has its own Python 3.7, separate from the Conda environment.

1.  **Install the system packages** (virtual display and the libraries Blender needs):
    ```bash
    sudo apt install -y xvfb libgl1 libgl1-mesa-dri libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 xz-utils wget
    ```

2.  **Install Blender 2.91.2**:
    ```bash
    mkdir -p ~/tools && cd ~/tools
    wget https://download.blender.org/release/Blender2.91/blender-2.91.2-linux64.tar.xz
    tar -xf blender-2.91.2-linux64.tar.xz && rm blender-2.91.2-linux64.tar.xz
    BLENDER=~/tools/blender-2.91.2-linux64/blender
    $BLENDER --version
    ```

3.  **Install the generator's packages into Blender's Python**:
    ```bash
    BPY=~/tools/blender-2.91.2-linux64/2.91/python/bin/python3.7m
    $BPY -m ensurepip
    $BPY -m pip install --upgrade pip
    $BPY -m pip install sympy==1.10.1 pyyaml
    ```

4.  **Check that rendering works** (from the repository root):
    ```bash
    cd autonomous_driving/CBM/rssgen/boia_config
    xvfb-run -a $BLENDER -b base_scene_boia.blend -E BLENDER_EEVEE -o /tmp/eevee_test_#### -F PNG -f 1
    ```
    This writes `/tmp/eevee_test_0001.png`. On a machine with a display, `xvfb-run -a` can be left out.

5.  **Add the traffic-light models**, which rsbench does not distribute: `TL_G.blend`, `TL_Y.blend` and `TL_R.blend` in `autonomous_driving/CBM/rssgen/boia_config/shapes/`, each holding one object named `TLG`, `TLY` or `TLR` with the lit lamp's colour in its own materials, saved with Blender 2.91.
