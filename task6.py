    print(results.round(3).to_string(index=False))

    best_name = results.iloc[0]["model"]
    best_pred = predictions[best_name]
    best_model = models[best_name]

    pred_df = pd.DataFrame({
        "actual_score": y_test.reset_index(drop=True),
        "predicted_score": np.round(best_pred, 2),
    })
    pred_df.to_csv(OUT / "test_predictions.csv", index=False)
    results.to_csv(OUT / "model_comparison.csv", index=False)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=pred_df["actual_score"], y=pred_df["predicted_score"])
    low = min(pred_df.min().min(), 0)
    high = max(pred_df.max().max(), 100)
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title(f"Actual vs Predicted ({best_name})")
    plt.tight_layout()
    plt.savefig(OUT / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Feature importance/coefficient overview
    if best_name == "Random Forest":
        importance = pd.Series(best_model.feature_importances_, index=features)
    else:
        estimator = best_model.named_steps["model"]
        importance = pd.Series(np.abs(estimator.coef_), index=features)
    importance.sort_values().plot(kind="barh", figsize=(8, 5))
    plt.title(f"Feature Importance / Coefficient Magnitude ({best_name})")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close()

    return results, best_name, pred_df


def write_summary(df, results, best_name):
    best = results.iloc[0]
    report_lines = [
        "# Final Data Science Capstone Report",
        "",
        "## Executive Summary",
        "This capstone demonstrates an end-to-end data science workflow for predicting student final exam scores.",
        "The dataset is synthetic and was created for educational demonstration, not from actual student records.",
        "",
        "## Dataset",
        f"- Records after cleaning: {len(df)}",
        "- Features: study hours, attendance, assignment average, previous exam score, extracurricular participation",
        "- Target: final exam score",
        "",
        "## Workflow",
        "1. Generated a reproducible sample dataset.",
        "2. Inspected data types, missing values, and summary statistics.",
        "3. Removed duplicates and filled missing numeric values with medians.",
        "4. Explored relationships using plots and correlations.",
        "5. Split data into training and test subsets.",
        "6. Trained Linear Regression, Ridge Regression, and Random Forest models.",
        "7. Evaluated models using MAE, RMSE, and R-squared.",
        "8. Saved predictions, model metrics, and charts.",
        "",
        "## Model Comparison",
        results.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        "## Best Model",
        f"- Selected model based on lowest test RMSE: {best_name}",
        f"- MAE: {best['MAE']:.3f}",
        f"- RMSE: {best['RMSE']:.3f}",
        f"- R-squared: {best['R2']:.3f}",
        "",
        "## Interpretation and Recommendations",
        "- Treat model results as a learning demonstration because the dataset is simulated.",
        "- In a real educational setting, validate the model on real, consented, appropriately anonymized data.",
        "- Use predictions as supportive information, not as the sole basis for decisions about students.",
        "- Further work could include cross-validation, tuning, fairness evaluation, and comparison with additional models.",
        "",
        "## Outputs",
        "- cleaned_dataset.csv",
        "- model_comparison.csv",
        "- test_predictions.csv",
        "- final_score_distribution.png",
        "- study_hours_vs_score.png",
        "- correlation_heatmap.png",
        "- actual_vs_predicted.png",
        "- feature_importance.png",
        "",
        "## Limitations",
        "The synthetic data encodes an artificial relationship between inputs and target. Performance metrics therefore "
        "do not establish real-world predictive ability. This is a portfolio demonstration of workflow and tools.",
    ]
    (OUT / "capstone_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main():
    raw = create_dataset()
    explore_data(raw)
    cleaned = clean_data(raw)
    cleaned.to_csv(OUT / "cleaned_dataset.csv", index=False)
    visualize(cleaned)
    results, best_name, _ = train_and_evaluate(cleaned)
    write_summary(cleaned, results, best_name)
    print(f"\nCapstone completed. Outputs are in: {OUT.resolve()}")


if __name__ == "__main__":
    main()
