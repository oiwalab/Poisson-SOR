"""可視化モジュール

ポテンシャル分布、電極パターン、収束履歴などを可視化
"""

import numpy as np
import matplotlib.pyplot as plt

# from pathlib import Path
from typing import Optional, Tuple


def plot_potential_slice(
    phi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    z_index: Optional[int] = None,
    electrode_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Potential Distribution",
) -> None:
    """ポテンシャル分布の2Dスライスをプロット

    Parameters
    ----------
    phi : np.ndarray
        ポテンシャル分布 (nx, ny, nz)
    x, y, z : np.ndarray
        座標配列 (m)
    z_index : int, optional
        z方向のスライス位置（インデックス）。Noneの場合は中央
    electrode_mask : np.ndarray, optional
        電極マスク (nx, ny, nz)
    save_path : str, optional
        保存先のパス
    title : str
        グラフのタイトル
    """
    if z_index is None:
        z_index = phi.shape[2] // 2

    # スライスを取得
    phi_slice = phi[:, :, z_index]

    # プロット
    fig, ax = plt.subplots(figsize=(8, 6))

    # ポテンシャル分布
    im = ax.pcolormesh(x * 1e9, y * 1e9, phi_slice.T, cmap="RdBu_r", shading="auto")

    # 電極位置を重ねて表示
    if electrode_mask is not None:
        electrode_slice = electrode_mask[:, :, z_index]
        if electrode_slice.any():
            ax.contour(
                x * 1e9,
                y * 1e9,
                electrode_slice.T,
                colors="black",
                linewidths=2,
                levels=[0.5],
            )

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(f"{title} at z={z[z_index] * 1e9:.1f} nm")
    ax.set_aspect("equal")

    # カラーバー
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Potential (V)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()


def plot_multiple_slices(
    phi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    z_indices: Optional[list] = None,
    electrode_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """複数のz位置でのポテンシャル分布をプロット

    Parameters
    ----------
    phi : np.ndarray
        ポテンシャル分布 (nx, ny, nz)
    x, y, z : np.ndarray
        座標配列 (m)
    z_indices : list, optional
        z方向のスライス位置のリスト。Noneの場合は均等に4つ
    electrode_mask : np.ndarray, optional
        電極マスク (nx, ny, nz)
    save_path : str, optional
        保存先のパス
    """
    if z_indices is None:
        nz = phi.shape[2]
        z_indices = [nz // 4, nz // 2, 3 * nz // 4, -1]

    n_slices = len(z_indices)
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 4))

    if n_slices == 1:
        axes = [axes]

    vmin = phi.min()
    vmax = phi.max()

    for i, z_idx in enumerate(z_indices):
        ax = axes[i]
        phi_slice = phi[:, :, z_idx]

        # ポテンシャル分布
        im = ax.pcolormesh(
            x * 1e9,
            y * 1e9,
            phi_slice.T,
            cmap="RdBu_r",
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )

        # 電極位置
        if electrode_mask is not None:
            electrode_slice = electrode_mask[:, :, z_idx]
            if electrode_slice.any():
                ax.contour(
                    x * 1e9,
                    y * 1e9,
                    electrode_slice.T,
                    colors="black",
                    linewidths=2,
                    levels=[0.5],
                )

        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(f"z={z[z_idx] * 1e9:.1f} nm")
        ax.set_aspect("equal")

    # カラーバー（共通）
    fig.colorbar(im, ax=axes, label="Potential (V)", shrink=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()


def plot_electrode_pattern(
    electrode_mask: np.ndarray,
    electrode_voltages: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z_index: int = -1,
    save_path: Optional[str] = None,
) -> None:
    """電極パターンを可視化

    Parameters
    ----------
    electrode_mask : np.ndarray
        電極マスク (nx, ny, nz)
    electrode_voltages : np.ndarray
        電極電圧 (nx, ny, nz)
    x, y : np.ndarray
        座標配列 (m)
    z_index : int
        z方向のインデックス（デフォルトは表面）
    save_path : str, optional
        保存先のパス
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # 電極マスクとデータのスライス
    mask_slice = electrode_mask[:, :, z_index]
    voltage_slice = electrode_voltages[:, :, z_index].copy()

    # 電極がない部分はNaNに
    voltage_slice[~mask_slice] = np.nan

    # プロット
    im = ax.pcolormesh(
        x * 1e9, y * 1e9, voltage_slice.T, cmap="viridis", shading="auto"
    )

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title("Electrode Pattern and Voltages")
    ax.set_aspect("equal")

    # カラーバー
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Voltage (V)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()


def plot_convergence(
    residual_history: list,
    save_path: Optional[str] = None,
) -> None:
    """収束履歴をプロット

    Parameters
    ----------
    residual_history : list
        残差の履歴
    save_path : str, optional
        保存先のパス
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    iterations = range(1, len(residual_history) + 1)
    ax.semilogy(iterations, residual_history, "b-", linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual (L2 norm)")
    ax.set_title("Convergence History")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()


def save_results(
    phi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    info: dict,
    save_path: str,
) -> None:
    """計算結果をファイルに保存

    Parameters
    ----------
    phi : np.ndarray
        ポテンシャル分布
    x, y, z : np.ndarray
        座標配列
    info : dict
        収束情報
    save_path : str
        保存先のパス（.npz形式）
    """
    np.savez(
        save_path,
        phi=phi,
        x=x,
        y=y,
        z=z,
        converged=info.get("converged", False),
        iterations=info.get("iterations", 0),
        final_residual=info.get("final_residual", 0.0),
    )
    print(f"Results saved to: {save_path}")


def load_results(
    file_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """保存した結果を読み込み

    Parameters
    ----------
    file_path : str
        ファイルパス（.npz形式）

    Returns
    -------
    phi, x, y, z : np.ndarray
        ポテンシャル分布と座標
    info : dict
        収束情報
    """
    data = np.load(file_path)

    phi = data["phi"]
    x = data["x"]
    y = data["y"]
    z = data["z"]

    info = {
        "converged": bool(data["converged"]),
        "iterations": int(data["iterations"]),
        "final_residual": float(data["final_residual"]),
    }

    return phi, x, y, z, info
