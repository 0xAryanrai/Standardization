# Standardization
A deep-dive into feature standardization using the Titanic dataset — comparing StandardScaler, MinMaxScaler, and RobustScaler, exposing the data leakage mistake most tutorials miss, and proving scaling's real impact with before/after model accuracy.

#  Standardization on Titanic: Why Scaling Quietly Breaks (or Fixes) Your Model

A deep-dive notebook on feature scaling and standardization, using the Titanic dataset as a running example. Instead of just showing `StandardScaler()` in one line, this notebook explains **why** scaling matters, **which** scaler to use and when, and the **#1 mistake** most tutorials skip: data leakage from scaling before a train/test split.

[![Kaggle](https://img.shields.io/badge/Kaggle-View%20Notebook-20BEFF?logo=kaggle)](#) <!-- replace # with your notebook link -->
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](#)

---

##  What this notebook covers

- Why `Age` and `Fare` sitting on very different scales hurts distance-based models
- A visual and numerical comparison of **StandardScaler**, **MinMaxScaler**, and **RobustScaler**
- The most common scaling mistake: fitting a scaler before splitting train/test data (data leakage)
- The correct, production-safe approach using `Pipeline`
- Real model accuracy, with vs. without scaling, across Logistic Regression, KNN, and SVM
- Why tree-based models (Random Forest) don't need scaling at all

##  Why this matters

Most tutorials treat standardization as a single line of boilerplate code before modeling. This notebook treats it as a decision that affects model performance, generalization, and correctness — backed by visuals and real before/after accuracy numbers rather than just stating "scaling is important."

##  Dataset

This notebook uses the [Titanic dataset](https://www.kaggle.com/c/titanic) (`train.csv`), specifically the `Age`, `Fare`, `Pclass`, `Sex`, `SibSp`, `Parch`, and `Embarked` features, with `Survived` as the target.

##  Tech stack

- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Pipeline`, `LogisticRegression`, `KNeighborsClassifier`, `SVC`, `RandomForestClassifier`)

##  Notebook structure

| Section | What it shows |
|---|---|
| 1. Introduction | Why this notebook exists and what makes it different |
| 2. Data loading & light cleaning | Minimal preprocessing — not an EDA showcase |
| 3. The scale mismatch problem | Visualizing `Age` vs `Fare` distributions |
| 4. Scaler comparison | StandardScaler vs MinMaxScaler vs RobustScaler, visually |
| 5. The data leakage mistake | Wrong way vs right way to scale relative to train/test split |
| 6. Pipelines | The production-safe pattern for scaling |
| 7. Model impact study | Real accuracy comparison, scaled vs unscaled |
| 8. When scaling doesn't matter | Random Forest as a counterexample |
| 9. Key takeaways | Summary table and rules of thumb |

##  How to run

1. Open the notebook on Kaggle (link above) and click **Copy & Edit**, or
2. Clone this repo and run locally:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
jupyter notebook standardization-titanic.ipynb
```

You'll need the Titanic `train.csv` file, available from the [Kaggle competition page](https://www.kaggle.com/c/titanic/data).

##  Key takeaways from the notebook

1. `Age` and `Fare` sit on very different scales — this directly affects distance-based models like KNN and SVM.
2. **Always split before scaling.** Fitting a scaler on the full dataset leaks test set information into training.
3. Use `Pipeline` to enforce correct scaling automatically, especially under cross-validation.
4. `RobustScaler` handles `Fare`'s outliers better than `StandardScaler` or `MinMaxScaler`.
5. Tree-based models (Random Forest, XGBoost) are scale-invariant — don't bother scaling for them.

##  Contributing

Found an issue or have a suggestion? Open an issue or PR — feedback is welcome.

##  Support

If this notebook or repo helped you understand standardization better, consider:
- Starring this repo
- Upvoting the notebook on Kaggle

##  License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
