# Project Brief: KAN-D-IV-LATE Paper Enhancement

## Core Goal

The primary objective is to revise and enhance the research paper, "Rethinking Distributional IVs: KAN-Powered D-IV-LATE & Model Choice," to prepare it for a successful submission to a peer-reviewed academic journal.

## Primary Challenge

A significant discrepancy exists between the paper's current narrative and recent empirical findings. The paper posits the novel Kolmogorov-Arnold Network (KAN) based estimator as a superior alternative to standard methods. However, initial simulation results from the `STRATEGIC_IMPLEMENTATION_PLAN.md` indicate that the KAN estimator unexpectedly underperforms a Random Forest baseline, and its asymptotic confidence intervals have poor coverage.

## Secondary Challenge

The project operates under computational constraints. The initial full simulation run took over 13 hours. While some budget for cloud computing (Google Colab with TPU) is available, the overall research strategy must remain computationally efficient, favouring targeted experiments over exhaustive, long-running computations.

## Immediate Objective

The immediate task is to execute the Jupyter workbook located at `kan-d-iv-late-main/workbook/kan_performance_investigation.ipynb`. This workbook is designed to systematically and efficiently investigate the KAN estimator's performance by:

1.  **Hyperparameter Tuning:** Conducting a grid search to determine if a different set of hyperparameters can improve the KAN estimator's RMSE.
2.  **Bootstrap Analysis:** Running a limited bootstrap simulation with the best-performing hyperparameters to assess the reliability and coverage of its confidence intervals.

## Ultimate Goal

The ultimate goal is to use the evidence from the investigation to make a final decision on the paper's core narrative. This will lead to one of two outcomes:

1.  **Scenario A (KAN is Salvageable):** If the KAN estimator's performance can be improved to a competitive level, the project will proceed with one final, large-scale computation to generate robust results for the paper.
2.  **Scenario B (KAN is Not Superior):** If the KAN estimator remains inferior, the paper's focus will pivot to a "caveat emptor" narrative, highlighting the challenges of applying new, complex models in causal inference.

The final step will be to revise the paper's text, figures, and conclusions to reflect the chosen, evidence-based narrative.
