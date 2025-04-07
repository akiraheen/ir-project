import argparse
import json
from pathlib import Path
import random
import pandas as pd
from tqdm import tqdm
from typing import List, Optional
from PIL import Image
from evaluate import RetrievalEvaluator
from reranker import Reranker, RerankerName
from retriever import CLIPRetrievalSystem
from transformations import (
    BrightnessVariation,
    CameraRotation,
    GaussianNoise,
    Identity,
    ImageTransformation,
    MotionBlur,
)
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
DATA_DIR = Path("data/Yummly28K/images27638")
METADATA_DIR = Path("data/Yummly28K/metadata27638")
TABLES_DIR = Path("tables")
PLOTS_DIR = Path("plots")


def evaluate_images(
    images: List[Path],
    transformations: List[ImageTransformation],
    show_images: bool,
    evaluator: RetrievalEvaluator,
    reranking: Optional[RerankerName] = None,
    metadata_dir: Path = METADATA_DIR,
):
    results = {}
    for image_path in tqdm(images, desc="Evaluating images"):
        image_name = image_path.name
        # skip the `img` part of the filename
        image_nr = image_path.stem[3:]
        metadata_path = Path(METADATA_DIR) / f"meta{image_nr}.json"
        metadata = json.loads(metadata_path.read_text())
        image_id = metadata["id"]

        image = Image.open(image_path).convert("RGB")
        result = evaluator.evaluate_transformations(
            original_image=image,
            original_id=image_id,
            transformations=transformations,
            show_images=show_images,
            re_ranking=reranking,
        )
        results[image_name] = result

    return results


def visualize_results(df: pd.DataFrame, plots_dir: Path = PLOTS_DIR):
    # Reset index to convert the multi-index to columns
    df = df.reset_index()

    # Create directory for plots
    plots_dir.mkdir(exist_ok=True, parents=True)

    print(f"Generating and saving plots to {plots_dir}...")

    # Create boxplot for MRR
    plt.figure(figsize=(12, 6))
    _ = df.boxplot(column="mrr", by="transformation", grid=False)
    plt.title("MRR by Transformation")
    plt.suptitle("")  # Remove default suptitle
    plt.xlabel("Transformation")
    plt.ylabel("MRR Value")
    plt.tight_layout()

    # Save figure directly
    plot_path = plots_dir / "boxplot_mrr.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")

    # Create bar chart for average MRR by transformation
    plt.figure(figsize=(14, 8))
    grouped_df = df.groupby("transformation")["mrr"].mean()
    grouped_df.plot(kind="bar", figsize=(14, 8))
    plt.title("Average MRR by Transformation")
    plt.xlabel("Transformation")
    plt.ylabel("Mean MRR")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save figure directly
    plot_path = plots_dir / "barplot_mrr.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")

    print(f"All plots saved to {plots_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of experiment iterations to run. Default is 5.",
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
        default=100,
        help="Number of images to evaluate. Default is 100.",
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
    parser.add_argument(
        "--reranking",
        type=str,
        default=None,
        choices=["jaccard", "bm25"],
        help="Reranking method to use. Default is None.",
    )

    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tables_dir).mkdir(parents=True, exist_ok=True)
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)  # reset the seed for each iteration
    images = random.sample(list(args.data_dir.glob("*.jpg")), args.num_images)

    retriever = CLIPRetrievalSystem()
    reranker = Reranker(retriever)
    evaluator = RetrievalEvaluator(retriever, reranker)
    # evaluator = RetrievalEvaluator(reranker)

    # TODO: query on original images and rerank

    rows = []
    for i in tqdm(range(args.iterations), desc="Running experiments"):
        # change the seed for each iteration to get different, but reproducible, transformations
        transformations = [
            Identity(seed=args.seed + i),  # baseline query with no transformations
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
            reranking=args.reranking,
        )

        if args.no_save:
            continue

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

    results_df = pd.DataFrame(rows)

    summary_file = args.results_dir / "summary.csv"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    # Calculate mean and std for each transformation
    mean_df = (
        results_df.copy()
        .drop(columns=["iteration", "filename"])
        .groupby(["transformation"])
        .mean()
        .round(2)
    )

    std_df = (
        results_df.copy()
        .drop(columns=["iteration", "filename"])
        .groupby(["transformation"])
        .std()
        .round(2)
    )

    # Rename mrr to mean_mrr in the mean dataframe
    mean_df = mean_df.rename(columns={"mrr": "mean_mrr"})
    std_df = std_df.rename(columns={"mrr": "std_mrr"})

    # Combine mean and std dataframes
    summary_df = pd.concat([mean_df, std_df], axis=1)

    summary_df.to_csv(summary_file)

    visualize_results(results_df, plots_dir=args.plots_dir)


if __name__ == "__main__":
    main()
