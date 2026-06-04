# SLEEC to Marabou Verification Tool

This project provides a tool for verifying SLEEC (social, legal, ethical, empathetic, and cultural) requirements against Neural Network models.

## Environment Setup

To recreate the Conda environment required for this project, follow these steps:

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository-url>
    cd ai-sleec-ver
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
    - In the **Model** dropdown menu, select `carebot_decision_model.onnx` or `carebot_decision_model_v2.onnx`.

4.  **Perform Verification**:
    - Click the **VERIFY** button (in green) to check the loaded requirements against the selected NN model.
    - The results will be displayed in the **Output** panel and the right editor pane.

5.  **SMT Checks (Optional)**:
    - You can also perform standard SLEEC SMT checks (redundancy, conflicts, concern, purpose, situational conflict) by clicking the respective buttons.
