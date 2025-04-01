import argparse
from pathlib import Path
import random
import pandas as pd
from tqdm import tqdm
from typing import List
from PIL import Image
from evaluate import RetrievalEvaluator
from retriever import CLIPRetrievalSystem
from transformations import (
    BrightnessVariation,
    CameraRotation,
    GaussianNoise,
    ImageTransformation,
    MotionBlur,
)
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
DATA_DIR = Path("data/Yummly28K/images27638")
TABLES_DIR = Path("tables")
PLOTS_DIR = Path("plots")


def evaluate_images(
    images: List[Path],
    transformations: List[ImageTransformation],
    show_images: bool,
    evaluator: RetrievalEvaluator,
):
    results = {}
    for image_path in tqdm(images, desc="Evaluating images"):
        image_name = image_path.name
        image = Image.open(image_path).convert("RGB")
        result = evaluator.evaluate_transformations(image, transformations, show_images)
        results[image_name] = result

    return results


def visualize_results(df: pd.DataFrame, plots_dir: Path = PLOTS_DIR):
    # Reset index to convert the multi-index to columns
    df = df.reset_index()

    # Create directory for plots
    plots_dir.mkdir(exist_ok=True, parents=True)

    print(f"Generating and saving plots to {plots_dir}...")

    # Create figures for all metric types
    metric_prefixes = ["ndcg@", "map@", "recall@", "accuracy@"]

    # Visualize each metric type with boxplots
    for metric_prefix in metric_prefixes:
        # Extract all columns for this metric (e.g., ndcg@1, ndcg@5, etc.)
        metric_columns = [col for col in df.columns if col.startswith(metric_prefix)]

        if not metric_columns:
            continue

        # Create a new dataframe for plotting
        plot_data = []
        for col in metric_columns:
            temp_df = df[["transformation", col]].copy()
            temp_df["metric"] = col
            temp_df["value"] = temp_df[col]
            plot_data.append(temp_df[["transformation", "metric", "value"]])

        plot_df = pd.concat(plot_data)

        # Create and save box plot
        plt.figure(figsize=(12, 6))
        _ = plot_df.boxplot(column="value", by="transformation", grid=False)
        plt.title(f"{metric_prefix.upper().rstrip('@')} Metrics by Transformation")
        plt.suptitle("")  # Remove default suptitle
        plt.xlabel("Transformation")
        plt.ylabel(f"{metric_prefix.upper().rstrip('@')} Value")
        plt.tight_layout()

        # Save figure directly
        plot_path = plots_dir / f"boxplot_{metric_prefix.rstrip('@')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {plot_path}")

    # Create comparison plots across transformations
    k_values = [
        int(col.split("@")[1])
        for col in df.columns
        if col.startswith(metric_prefixes[0])
    ]

    for k in k_values:
        # Create metrics at k bar charts
        metrics_at_k = [
            f"{prefix}{k}" for prefix in metric_prefixes if f"{prefix}{k}" in df.columns
        ]

        # Group by transformation and calculate mean for each metric
        grouped_df = df.groupby("transformation")[metrics_at_k].mean()

        # Transpose DataFrame so metrics are on the x-axis
        grouped_df = grouped_df.transpose()

        # Create and save bar chart
        plt.figure(figsize=(14, 8))
        grouped_df.plot(kind="bar", figsize=(14, 8), ax=plt.gca())
        plt.title(f"Comparison of Metrics @{k} Across Transformations")
        plt.xlabel("Metric")
        plt.ylabel("Metric Value")
        plt.legend(title="Transformations")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save figure directly
        plot_path = plots_dir / f"barplot_k{k}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {plot_path}")

    # Create and save heatmap of all metrics
    plt.figure(figsize=(16, 10))
    all_metrics = [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in metric_prefixes)
    ]
    heatmap_data = df.groupby("transformation")[all_metrics].mean()

    # Plot heatmap
    plt.imshow(heatmap_data, cmap="viridis", aspect="auto")
    plt.colorbar(label="Metric Value")
    plt.xticks(range(len(all_metrics)), all_metrics, rotation=45, ha="right")
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)  # type: ignore
    plt.title("Heatmap of All Metrics Across Transformations")

    # Add text annotations in the heatmap cells
    for i in range(len(heatmap_data.index)):
        for j in range(len(all_metrics)):
            plt.text(
                j,
                i,
                f"{heatmap_data.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if heatmap_data.iloc[i, j] < 0.5 else "black",  # type: ignore
            )

    plt.tight_layout()

    # Save heatmap directly
    plot_path = plots_dir / "heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")

    print(f"All plots saved to {plots_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of experiment iterations to run. Default is 10.",
    )
    parser.add_argument(
        "--show-images",
        action="store_true",
        help="Show the results in a window. Default is False.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the results to a file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of images to evaluate. Default is 10.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to the data directory. Default is data/Yummly28K/images27638.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Path to the results directory. Default is results.",
    )
    parser.add_argument(
        "--min-camera-rotation",
        type=float,
        default=-30,
        help="Minimum camera rotation angle. Default is -30.",
    )
    parser.add_argument(
        "--max-camera-rotation",
        type=float,
        default=30,
        help="Maximum camera rotation angle. Default is 30.",
    )
    parser.add_argument(
        "--min-brightness-variation",
        type=float,
        default=0.4,
        help="Minimum brightness variation factor. Default is 0.4.",
    )
    parser.add_argument(
        "--max-brightness-variation",
        type=float,
        default=0.8,
        help="Maximum brightness variation factor. Default is 0.8.",
    )
    parser.add_argument(
        "--min-gaussian-noise",
        type=float,
        default=0.01,
        help="Minimum Gaussian noise standard deviation. Default is 0.01.",
    )
    parser.add_argument(
        "--max-gaussian-noise",
        type=float,
        default=0.03,
        help="Maximum Gaussian noise standard deviation. Default is 0.03.",
    )
    parser.add_argument(
        "--min-motion-blur",
        type=int,
        default=3,
        help="Minimum motion blur kernel size. Default is 3.",
    )
    parser.add_argument(
        "--max-motion-blur",
        type=int,
        default=5,
        help="Maximum motion blur kernel size. Default is 5.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=TABLES_DIR,
        help="Path to the tables directory. Default is tables.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Path to the plots directory. Default is plots.",
    )

    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tables_dir).mkdir(parents=True, exist_ok=True)
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)  # reset the seed for each iteration
    images = random.sample(list(args.data_dir.glob("*.jpg")), args.num_images)

    retriever = CLIPRetrievalSystem()
    evaluator = RetrievalEvaluator(retriever)

    rows = []
    for i in tqdm(range(args.iterations), desc="Running experiments"):
        # change the seed for each iteration to get different, but reproducible, transformations
        transformations = [
            CameraRotation(
                angle_range=(args.min_camera_rotation, args.max_camera_rotation),
                seed=args.seed + i,
            ),
            BrightnessVariation(
                factor_range=(
                    args.min_brightness_variation,
                    args.max_brightness_variation,
                ),
                seed=args.seed + i,
            ),
            GaussianNoise(
                std_range=(args.min_gaussian_noise, args.max_gaussian_noise),
                seed=args.seed + i,
            ),
            MotionBlur(
                kernel_size_range=(args.min_motion_blur, args.max_motion_blur),
                seed=args.seed + i,
            ),
        ]
        results = evaluate_images(
            images=images,
            transformations=transformations,
            show_images=args.show_images,
            evaluator=evaluator,
        )

        if args.no_save:
            continue

        for image_name, result in results.items():
            file_path = (
                Path(args.tables_dir)
                / f"iteration_{i:03d}"
                / f"{Path(image_name).stem}.txt"
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)

            evaluator.write_comparison_results(result, file_path)

        rows.extend(
            [
                {
                    "iteration": i,
                    "transformation": transformation_name,
                    "filename": filename,
                    **metrics,
                }
                for filename, transformation in results.items()
                for transformation_name, metrics in transformation.items()
            ]
        )

    results_file = args.results_dir / "metrics.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(rows).set_index(
        ["iteration", "transformation", "filename"]
    )
    results_df.to_csv(results_file)

    visualize_results(results_df, plots_dir=args.plots_dir)


if __name__ == "__main__":
    main()
