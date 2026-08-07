"""Figures for the representation ladder exhibit."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ladder_figure(sources: list[tuple[str, Image.Image]], averages: dict[str, Image.Image], weight_label: str, path: Path) -> None:
    """Show fixed inputs/weights against outputs from multiple backends."""

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = max(len(sources), len(averages), 1)
    fig, axes = plt.subplots(2, columns, figsize=(3 * columns, 5.8), squeeze=False)
    for index, (name, image) in enumerate(sources):
        axes[0][index].imshow(image)
        axes[0][index].set_title(name, fontsize=10)
        axes[0][index].axis("off")
    for index in range(len(sources), columns):
        axes[0][index].axis("off")
    for index, (backend, image) in enumerate(averages.items()):
        axes[1][index].imshow(image)
        axes[1][index].set_title(backend, fontsize=10)
        axes[1][index].axis("off")
    for index in range(len(averages), columns):
        axes[1][index].axis("off")
    fig.suptitle(f"Same entities and weights ({weight_label}); different representation spaces", fontsize=13)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def comparison_figure(
    sources: list[tuple[str, Image.Image]],
    averages_by_intent: dict[str, dict[str, Image.Image]],
    weight_labels: dict[str, str],
    path: Path,
) -> None:
    """Render sources and one row of averages per weighting intent.

    The layout keeps entities, weighting intents, and implemented representation
    spaces comparable in a single image.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    intents = list(averages_by_intent)
    backends = list(next(iter(averages_by_intent.values()), {}))
    columns = max(len(sources), len(backends), 1)
    rows = 1 + len(intents)
    fig, axes = plt.subplots(rows, columns, figsize=(1.9 * columns, 1.7 * rows), squeeze=False)
    for index, (name, image) in enumerate(sources):
        axes[0][index].imshow(image)
        axes[0][index].set_title(name, fontsize=9)
    for row, intent in enumerate(intents, start=1):
        for index, backend in enumerate(backends):
            image = averages_by_intent[intent].get(backend)
            if image is None:
                continue
            axes[row][index].imshow(image)
            if row == 1:
                axes[row][index].set_title(backend, fontsize=9)
        axes[row][0].set_ylabel(weight_labels.get(intent, intent), fontsize=9)
    for row in range(rows):
        for column in range(columns):
            axes[row][column].set_xticks([])
            axes[row][column].set_yticks([])
            for spine in axes[row][column].spines.values():
                spine.set_visible(False)
            # Hide cells with no image; keep labelled first-column axes visible.
            if not axes[row][column].images and not axes[row][column].get_ylabel():
                axes[row][column].set_axis_off()
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def strip_figure(images: dict[str, Image.Image], title: str, path: Path) -> None:
    """Render a one-row comparison strip such as the population-alpha morph."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, max(1, len(images)), figsize=(2.6 * max(1, len(images)), 2.4), squeeze=False)
    for index, (label, image) in enumerate(images.items()):
        axes[0][index].imshow(image)
        axes[0][index].set_title(label, fontsize=9)
        axes[0][index].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def eigenflag_figure(images: list[Image.Image], path: Path) -> None:
    """Save PCA component images so the learned linear basis is inspectable."""

    if not images:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = min(8, len(images))
    fig, axes = plt.subplots(1, columns, figsize=(2.1 * columns, 2.0), squeeze=False)
    for index, image in enumerate(images[:columns]):
        axes[0][index].imshow(image)
        axes[0][index].set_title(f"PC{index + 1}", fontsize=9)
        axes[0][index].axis("off")
    fig.suptitle("Eigenflags: top PCA components", fontsize=12)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def embedding_diagnostic(points: dict[str, tuple[float, float]], averages: dict[str, tuple[float, float]], path: Path) -> None:
    """Plot the first two PCA coordinates for corpus flags and averages."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if points:
        values = np.asarray(list(points.values()))
        ax.scatter(values[:, 0], values[:, 1], s=10, alpha=0.35, label="corpus")
        for code in ("fr", "uy", "ps"):
            if code in points:
                ax.scatter([points[code][0]], [points[code][1]], s=48, label=code)
                ax.text(points[code][0], points[code][1], f" {code}", fontsize=9)
    for label, point in averages.items():
        ax.scatter([point[0]], [point[1]], marker="x", s=70, label=label)
    ax.set_title("PCA embedding diagnostic")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
