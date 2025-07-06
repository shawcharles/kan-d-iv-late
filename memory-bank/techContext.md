# Technical Context: KAN-D-IV-LATE Project

## Core Technologies

-   **Python**: The primary programming language for all simulation and estimation code.
-   **PyTorch**: The deep learning framework used for implementing the Kolmogorov-Arnold Networks (KANs). It is leveraged for its automatic differentiation and GPU acceleration capabilities.
-   **Jupyter Notebook**: The environment used for interactive development, experimentation, and visualization. The main performance investigation is structured within `kan_performance_investigation.ipynb`.

## Key Python Libraries

-   **`efficient-kan`**: The specific KAN implementation used in this project. It is chosen over the original `kan` library for its improved computational performance. The project's `kan_utils.py` module is built around this library.
-   **`scikit-learn`**: Used for standard machine learning components and utilities.
    -   `RandomForestClassifier`: Serves as the baseline model for comparison against KANs.
    -   `KFold`: Used for the cross-fitting procedure.
    -   `StandardScaler`: Applied to features before they are passed to the KAN models to improve training stability.
-   **`numpy`**: The fundamental library for numerical operations and data manipulation.
-   **`pandas`**: Used for data management, especially for handling the simulation and empirical datasets.
-   **`statsmodels`**: Used for statistical tests and multiple testing corrections (`multipletests`).
-   **`matplotlib` & `seaborn`**: The primary libraries for data visualization, used to create the plots in the paper and the analysis plots in the Jupyter workbook.
-   **`tqdm`**: Used to display progress bars during long-running simulations and bootstrap procedures, providing a better user experience.

## Development and Execution Environment

-   **Local Development**: The code is structured as a standard Python project with modules in the `code/` directory.
-   **GPU Acceleration**: The code is written to automatically leverage a CUDA-enabled GPU if available (`DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`). This is critical for making the KAN training computationally feasible.
-   **Google Colab**: The intended execution environment for the `kan_performance_investigation.ipynb` workbook, specifically to take advantage of the available TPU/GPU resources to accelerate the hyperparameter search and bootstrap analysis.

## Tool Usage Patterns

-   **Standardized Functions**: The `kan_utils.py` script acts as a central utility module. It contains standardized functions for training KANs (`train_standardized_kan`), performing cross-fitting, and running bootstrap inference. This promotes code reuse and ensures consistency between the simulation and empirical parts of the project.
-   **Global Hyperparameters**: Key hyperparameters (e.g., `KAN_STEPS`, `K_FOLDS`) are defined globally at the top of `kan_utils.py`. The investigation workbook temporarily overrides these settings to perform the grid search, but this centralized configuration provides a clear baseline.
