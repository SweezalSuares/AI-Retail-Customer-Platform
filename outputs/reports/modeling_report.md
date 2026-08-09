# Modelling Report — Customer Segment Classification

## Problem framing
Predict `customer_segment` (Occasional, Regular, Loyal, High_Value) from
demographic and behavioural features using a supervised multi-class classifier.

## Models compared
| model              |   accuracy |   precision_macro |   recall_macro |   f1_macro |
|:-------------------|-----------:|------------------:|---------------:|-----------:|
| GradientBoosting   |     0.7617 |            0.7753 |         0.7537 |     0.7635 |
| RandomForest       |     0.7575 |            0.7759 |         0.745  |     0.7584 |
| LogisticRegression |     0.7087 |            0.7132 |         0.6869 |     0.697  |

## Best model: GradientBoosting

### Classification report (test set)
```
              precision    recall  f1-score   support

  Occasional       0.77      0.84      0.81      4422
     Regular       0.65      0.60      0.63      2691
       Loyal       0.84      0.79      0.82      1832
  High_Value       0.83      0.78      0.80      1055

    accuracy                           0.76     10000
   macro avg       0.78      0.75      0.76     10000
weighted avg       0.76      0.76      0.76     10000

```

## Interpretation
- The best-performing model was selected using macro-averaged F1 score, which
  treats all four segments equally regardless of class imbalance (Occasional
  customers are the majority class in this dataset).
- Feature importance (see outputs/figures/10_feature_importance.png) highlights
  which behavioural signals most strongly separate segments — typically spend,
  purchase frequency, and engagement-derived features dominate over raw
  demographics like age.
- The confusion matrix shows where the model confuses adjacent segments (e.g.
  Regular vs Loyal), which is expected since segment boundaries are behavioural
  gradients rather than hard cutoffs.

## Business use
This model powers the "Predict My Segment" tool in the Streamlit app, allowing
a business user to enter a customer's behavioural profile and get an instant
segment prediction plus the confidence per class — useful for real-time
personalization and targeted retention campaigns.
