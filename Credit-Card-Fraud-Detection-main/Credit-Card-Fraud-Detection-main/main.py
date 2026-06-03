"""
Command Line Interface (CLI) Orchestrator for the Credit Card Fraud Detection Pipeline.
Built using Click to provide clean, structured commands for all operational stages.
"""

import sys
import os
from typing import Dict, Any, Optional
import click
import pandas as pd

from src.config import load_config, PipelineConfig
from src.exceptions import FraudPipelineError
from src.utils import ensure_directories, save_model, setup_logger
from src.data_loader import load_data
from src.preprocessing import split_data, get_preprocessor, impute_missing_values, handle_outliers
from src.train import train_baseline, train_champion, evaluate_with_cross_val_score
from src.xgboost_model import train_xgboost
from src.evaluate import evaluate_pipeline
from src.predict import FraudPredictor
from src.visualize import (
    plot_correlation_matrix,
    plot_confusion_matrix,
    plot_evaluation_curves,
    plot_feature_importance,
    plot_fraud_distribution
)

# Initialize structured logger
logger = setup_logger()


@click.group()
@click.option(
    "--config-path", 
    type=click.Path(exists=True), 
    default="config/config.yaml",
    help="Path override to YAML configuration file."
)
@click.pass_context
def cli(ctx: click.Context, config_path: str) -> None:
    """💳 Credit Card Fraud Detection Production MLOps Suite."""
    try:
        ctx.ensure_object(dict)
        config: PipelineConfig = load_config(config_path)
        ctx.obj["config"] = config
        ensure_directories(config)
    except Exception as e:
        click.secho(f"Configuration Initialization Error: {str(e)}", fg="red", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--quick-mode/--no-quick-mode",
    default=True,
    help="If quick-mode, trains RF directly without randomized hyperparameter optimization."
)
@click.option(
    "--model-type",
    type=click.Choice(["random_forest", "xgboost", "both"], case_sensitive=False),
    default="both",
    help="Which champion model(s) to train: random_forest, xgboost, or both."
)
@click.pass_context
def train(ctx: click.Context, quick_mode: bool, model_type: str) -> None:
    """Trains and serializes both baseline and champion models."""
    config: PipelineConfig = ctx.obj["config"]
    
    logger.info("=======================================================================")
    logger.info("STARTING MODEL TRAINING STAGE")
    logger.info("=======================================================================")
    
    try:
        # 1. Ingest Data
        df = load_data(config.data_path)
        
        # 2. Basic cleaning & winsorization
        df = impute_missing_values(df)
        df = handle_outliers(
            df, 
            columns=config.data.outlier_capping_columns,
            threshold_percentile=config.data.winsorize_quantile
        )
        
        # 3. Generate Exploratory Plots
        plot_correlation_matrix(df, output_dir=config.directories.visuals_dir)
        plot_fraud_distribution(df, output_dir=config.directories.visuals_dir)
        
        # 4. Stratified Split
        X_train, X_test, y_train, y_test = split_data(
            df,
            target_column=config.data.target_column,
            test_size=config.data.test_size,
            random_state=config.project.seed,
            stratify=config.data.stratify
        )
        
        # 5. Pipeline Preprocessing Transformer
        preprocessor = get_preprocessor(scaling_cols=config.data.scaling_columns)
        
        # 6. Train Baseline Model (Logistic Regression)
        baseline_pipeline = train_baseline(X_train, y_train, preprocessor, config=config)
        
        # Assess baseline cross-validation score
        cv_folds = config.models.champion.tuning.get("cv_folds", 5) if config.models.champion.tuning else 5
        evaluate_with_cross_val_score(
            baseline_pipeline, 
            X_train, 
            y_train, 
            cv=cv_folds, 
            scoring="f1",
            random_state=config.project.seed
        )
        
        # Serialize baseline model with basic metadata
        baseline_model_path = os.path.join(
            config.directories.models_dir, 
            config.models.baseline.name,
            f"{config.models.baseline.name}.joblib"
        )
        save_model(
            baseline_pipeline, 
            baseline_model_path, 
            metadata={
                "model_name": config.models.baseline.name,
                "type": config.models.baseline.type,
                "hyperparameters": config.models.baseline.params,
                "seed": config.project.seed
            }
        )
        
        # 7. Train Champion Model (Oversampled SMOTE Random Forest)
        champion_pipeline = train_champion(
            X_train,
            y_train,
            preprocessor,
            cv=cv_folds,
            quick_mode=quick_mode,
            random_state=config.project.seed,
            config=config
        )
        
        # Serialize champion model
        champion_model_path = os.path.join(
            config.directories.models_dir,
            config.models.champion.name,
            f"{config.models.champion.name}.joblib"
        )
        
        # Fetch best estimator params if tuning was performed, otherwise default params
        champion_metadata = {
            "model_name": config.models.champion.name,
            "type": config.models.champion.type,
            "hyperparameters": config.models.champion.params,
            "quick_mode": quick_mode,
            "seed": config.project.seed
        }
        
        save_model(champion_pipeline, champion_model_path, metadata=champion_metadata)
        
        logger.info("=======================================================================")
        logger.info("MODEL TRAINING COMPLETED SUCCESSFULY")
        logger.info("=======================================================================")
        click.secho("Training stage completed successfully!", fg="green")
        
    except FraudPipelineError as e:
        logger.error(f"Training Stage Aborted: {e.message}")
        sys.exit(2)
    except Exception as e:
        logger.critical(f"Unhandled Failure during training: {e}")
        sys.exit(3)


@cli.command()
@click.pass_context
def evaluate(ctx: click.Context) -> None:
    """Evaluates serialized baseline and champion models on testing partition."""
    config: PipelineConfig = ctx.obj["config"]
    
    logger.info("=======================================================================")
    logger.info("STARTING MODEL EVALUATION STAGE")
    logger.info("=======================================================================")
    
    try:
        # Load test dataset
        df = load_data(config.data_path)
        df = impute_missing_values(df)
        df = handle_outliers(df, columns=config.data.outlier_capping_columns)
        
        _, X_test, _, y_test = split_data(
            df,
            target_column=config.data.target_column,
            test_size=config.data.test_size,
            random_state=config.project.seed,
            stratify=config.data.stratify
        )
        
        # 1. Evaluate Baseline Model
        baseline_path = os.path.join(
            config.directories.models_dir,
            config.models.baseline.name,
            f"{config.models.baseline.name}.joblib"
        )
        baseline_pipeline = FraudPredictor(baseline_path).model
        
        baseline_metrics = evaluate_pipeline(
            baseline_pipeline, 
            X_test, 
            y_test, 
            config.models.baseline.name,
            reports_dir=config.directories.reports_dir
        )
        
        baseline_preds = baseline_pipeline.predict(X_test)
        baseline_probs = baseline_pipeline.predict_proba(X_test)[:, 1]
        
        plot_confusion_matrix(y_test, baseline_preds, config.models.baseline.name, output_dir=config.directories.visuals_dir)
        plot_evaluation_curves(y_test, baseline_probs, config.models.baseline.name, output_dir=config.directories.visuals_dir)
        
        # 2. Evaluate Champion Model
        champion_path = os.path.join(
            config.directories.models_dir,
            config.models.champion.name,
            f"{config.models.champion.name}.joblib"
        )
        champion_pipeline = FraudPredictor(champion_path).model
        
        champion_metrics = evaluate_pipeline(
            champion_pipeline,
            X_test,
            y_test,
            config.models.champion.name,
            reports_dir=config.directories.reports_dir
        )
        
        champion_preds = champion_pipeline.predict(X_test)
        champion_probs = champion_pipeline.predict_proba(X_test)[:, 1]
        
        plot_confusion_matrix(y_test, champion_preds, config.models.champion.name, output_dir=config.directories.visuals_dir)
        plot_evaluation_curves(y_test, champion_probs, config.models.champion.name, output_dir=config.directories.visuals_dir)
        
        # Sort and plot Feature Importances (champion only)
        feature_names = list(X_test.columns)
        scaled_feats = config.data.scaling_columns
        unscaled_feats = [col for col in feature_names if col not in scaled_feats]
        reordered_features = scaled_feats + unscaled_feats
        
        plot_feature_importance(
            champion_pipeline, 
            reordered_features, 
            config.models.champion.name,
            output_dir=config.directories.visuals_dir
        )
        
        # Print comparison resumes
        logger.info("=======================================================================")
        logger.info("METRIC RESUME PERFORMANCE COMPARISON")
        logger.info("=======================================================================")
        logger.info(f"{'Evaluation Metric':25s} | {'Baseline (LR)':15s} | {'Champion (RF)':15s}")
        logger.info("-" * 65)
        for metric in baseline_metrics.keys():
            logger.info(
                f"{metric:25s} | {baseline_metrics[metric]:15.4f} | "
                f"{champion_metrics[metric]:15.4f}"
            )
        logger.info("=======================================================================")
        click.secho("Evaluation completed. Reports and visuals exported successfully!", fg="green")
        
    except FraudPipelineError as e:
        logger.error(f"Evaluation Stage Aborted: {e.message}")
        sys.exit(2)
    except Exception as e:
        logger.critical(f"Unhandled Failure during evaluation: {e}")
        sys.exit(3)


@cli.command()
@click.pass_context
def predict(ctx: click.Context) -> None:
    """Executes a live batch and single prediction mock test using the Champion model."""
    config: PipelineConfig = ctx.obj["config"]
    
    logger.info("=======================================================================")
    logger.info("STARTING LIVE INFERENCE TEST STAGE")
    logger.info("=======================================================================")
    
    try:
        champion_path = os.path.join(
            config.directories.models_dir,
            config.models.champion.name,
            f"{config.models.champion.name}.joblib"
        )
        
        if not os.path.exists(champion_path):
            raise click.ClickException(f"Trained champion model not found at {champion_path}. Run 'train' command first.")
            
        predictor = FraudPredictor(champion_path)
        
        # Fetch a few raw lines from dataset for scoring
        df = load_data(config.data_path)
        mock_batch = df.head(5).copy()
        mock_labels = mock_batch["Class"].values
        
        preds, probs = predictor.predict_batch(mock_batch)
        
        logger.info("Batch Prediction Results Summary:")
        for idx in range(len(preds)):
            logger.info(
                f"Tx #{idx+1} | Ground Truth: {mock_labels[idx]} | "
                f"Prediction: {preds[idx]} ({'Fraudulent' if preds[idx] == 1 else 'Legitimate'}) | "
                f"Confidence Score: {probs[idx]:.4f}"
            )
            
        # Single record scoring demo
        single_record = mock_batch.iloc[0].to_dict()
        result = predictor.predict_single(single_record)
        
        logger.info("Single Record Result details:")
        logger.info(f"{result}")
        click.secho("Mock inference successfully performed!", fg="green")
        
    except Exception as e:
        logger.critical(f"Unhandled failure during prediction test: {e}")
        sys.exit(3)


@cli.command()
@click.option("--port", default=8000, help="Port to run FastAPI server on.")
@click.option("--host", default="0.0.0.0", help="Binding host interface.")
@click.pass_context
def serve(ctx: click.Context, port: int, host: str) -> None:
    """Spins up the FastAPI real-time inference microservice."""
    try:
        import uvicorn
    except ImportError:
        click.secho(
            "Error: 'uvicorn' is required for the web server but not installed.\n"
            "Please install standard extras: pip install fastapi uvicorn", 
            fg="red", err=True
        )
        sys.exit(1)
        
    config: PipelineConfig = ctx.obj["config"]
    # Pass configuration to environment so FastAPI can pick it up
    os.environ["MODEL_PATH"] = config.model_path
    
    click.echo(f"Starting FastAPI Microservice on {host}:{port}...")
    uvicorn.run("app.main_api:app", host=host, port=port, reload=True)


@cli.command()
@click.option("--max-samples", default=500, help="Max samples to use for SHAP computation.")
@click.option("--sample-index", default=0, help="Sample index for waterfall plot explanation.")
@click.pass_context
def explain(ctx: click.Context, max_samples: int, sample_index: int) -> None:
    """Generates SHAP feature importance plots and a Markdown explainability report."""
    config: PipelineConfig = ctx.obj["config"]

    logger.info("=======================================================================")
    logger.info("STARTING MODEL EXPLAINABILITY STAGE")
    logger.info("=======================================================================")

    try:
        from src.explain import compute_shap_values, plot_shap_summary, plot_shap_waterfall, generate_shap_report

        champion_path = os.path.join(
            config.directories.models_dir,
            config.models.champion.name,
            f"{config.models.champion.name}.joblib"
        )
        if not os.path.exists(champion_path):
            raise click.ClickException(f"Champion model not found at {champion_path}. Run 'train' first.")

        df = load_data(config.data_path)
        df = impute_missing_values(df)
        _, X_test, _, _ = split_data(
            df,
            target_column=config.data.target_column,
            test_size=config.data.test_size,
            random_state=config.project.seed,
            stratify=config.data.stratify
        )

        champion_pipeline = FraudPredictor(champion_path).model
        shap_values, explainer, X_transformed = compute_shap_values(
            champion_pipeline, X_test, max_samples=max_samples
        )

        feature_names = list(X_test.columns)
        plot_shap_summary(shap_values, X_transformed, feature_names,
                          output_dir=config.directories.visuals_dir,
                          model_name=config.models.champion.name)
        plot_shap_waterfall(shap_values, X_transformed, sample_index=sample_index,
                            feature_names=feature_names,
                            output_dir=config.directories.visuals_dir,
                            model_name=config.models.champion.name)
        generate_shap_report(shap_values, X_transformed, feature_names,
                             output_dir=config.directories.reports_dir,
                             model_name=config.models.champion.name)

        click.secho("Explainability plots and report generated successfully!", fg="green")

    except Exception as e:
        logger.critical(f"Explainability stage failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    cli()
