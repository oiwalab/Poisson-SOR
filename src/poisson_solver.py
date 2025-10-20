"""SOR法による3Dポアソン方程式ソルバ

誘電率が不均一な系でのポアソン方程式 -∇⋅(ε∇ϕ)=ρ を解く
"""

import numpy as np
from typing import Dict, Optional, Tuple


class PoissonSolver:
    """SOR法による3Dポアソンソルバ

    Parameters
    ----------
    epsilon : np.ndarray
        誘電率分布 (nx, ny, nz)
    grid_spacing : float
        格子間隔 h (m) - 等方格子のみサポート (dx = dy = dz = h)
    boundary_conditions : Dict
        境界条件の設定
    omega : float
        SOR緩和パラメータ (1 < omega < 2)
    tolerance : float
        収束判定の閾値
    max_iterations : int
        最大反復回数
    """

    def __init__(
        self,
        epsilon: np.ndarray,
        grid_spacing: float,
        boundary_conditions: Dict,
        omega: float = 1.8,
        tolerance: float = 1e-6,
        max_iterations: int = 10000,
        electrode_mask: Optional[np.ndarray] = None,
        electrode_voltages: Optional[np.ndarray] = None,
    ):
        self.epsilon = epsilon
        self.nx, self.ny, self.nz = epsilon.shape
        self.h = grid_spacing  # 等方格子間隔
        self.boundary_conditions = boundary_conditions
        self.omega = omega
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # 電極マスクと電圧
        self.electrode_mask = electrode_mask
        self.electrode_voltages = electrode_voltages

        # 真空の誘電率 (F/m)
        self.epsilon_0 = 8.854187817e-12

        # 収束履歴
        self.residual_history = []

    def solve(
        self,
        rho: Optional[np.ndarray] = None,
        phi_initial: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """ポアソン方程式を解く

        Parameters
        ----------
        rho : np.ndarray, optional
            電荷密度分布 (C/m^3), shape=(nx, ny, nz)
            Noneの場合はゼロとして扱う
        phi_initial : np.ndarray, optional
            初期ポテンシャル分布 (V)

        Returns
        -------
        phi : np.ndarray
            ポテンシャル分布 (V), shape=(nx, ny, nz)
        info : Dict
            収束情報（反復回数、最終残差等）
        """
        # 電荷密度の初期化
        if rho is None:
            rho = np.zeros((self.nx, self.ny, self.nz))

        # ポテンシャルの初期化
        if phi_initial is None:
            phi = np.zeros((self.nx, self.ny, self.nz))
        else:
            phi = phi_initial.copy()

        self.residual_history = []

        # 電極領域の電位を設定（固定値）
        if self.electrode_mask is not None and self.electrode_voltages is not None:
            phi[self.electrode_mask] = self.electrode_voltages[self.electrode_mask]

        # 初期の境界条件を適用
        phi = self.apply_boundary_conditions(phi)

        # SOR反復
        for iteration in range(self.max_iterations):
            # SOR更新（内部点のみ、境界は更新しない）
            phi = self._sor_iteration(phi, rho)

            # 境界条件を再適用（念のため）
            phi = self.apply_boundary_conditions(phi)

            # 残差計算
            residual = self.compute_residual(phi, rho)
            self.residual_history.append(residual)

            if verbose:
                if (iteration + 1) % 100 == 0:
                    print("=" * 40)
                if (iteration + 1) % 10 == 0:
                    print(f"Iteration {iteration + 1}: Residual = {residual:.6e}")

            # 収束判定
            if residual < self.tolerance:
                info = {
                    "converged": True,
                    "iterations": iteration + 1,
                    "final_residual": residual,
                }
                return phi, info
            if np.isnan(residual) or np.isinf(residual):
                raise ValueError("Residual became NaN or Inf, diverging solution.")

        # 最大反復回数に到達
        info = {
            "converged": False,
            "iterations": self.max_iterations,
            "final_residual": residual,
        }
        return phi, info

    def _sor_iteration(self, phi: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """SOR法による1回の反復更新

        誘電率が不均一な場合の有限差分式を使用
        界面での誘電率は調和平均を使用
        """
        phi_new = phi.copy()

        # 内部の格子点のみ更新（境界は別途処理）
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                for k in range(1, self.nz - 1):
                    # 電極領域はスキップ（電位固定）
                    if self.electrode_mask is not None and self.electrode_mask[i, j, k]:
                        continue

                    # 隣接点の誘電率を調和平均で近似
                    # ε_{i+1/2,j,k} = 2*ε_i*ε_{i+1} / (ε_i + ε_{i+1})
                    eps_i = self.epsilon[i, j, k]
                    eps_ip = self.epsilon[i + 1, j, k]
                    eps_im = self.epsilon[i - 1, j, k]
                    eps_jp = self.epsilon[i, j + 1, k]
                    eps_jm = self.epsilon[i, j - 1, k]
                    eps_kp = self.epsilon[i, j, k + 1]
                    eps_km = self.epsilon[i, j, k - 1]

                    eps_xp = (
                        2 * eps_i * eps_ip / (eps_i + eps_ip)
                        if (eps_i + eps_ip) > 0
                        else 0
                    )
                    eps_xm = (
                        2 * eps_i * eps_im / (eps_i + eps_im)
                        if (eps_i + eps_im) > 0
                        else 0
                    )
                    eps_yp = (
                        2 * eps_i * eps_jp / (eps_i + eps_jp)
                        if (eps_i + eps_jp) > 0
                        else 0
                    )
                    eps_ym = (
                        2 * eps_i * eps_jm / (eps_i + eps_jm)
                        if (eps_i + eps_jm) > 0
                        else 0
                    )
                    eps_zp = (
                        2 * eps_i * eps_kp / (eps_i + eps_kp)
                        if (eps_i + eps_kp) > 0
                        else 0
                    )
                    eps_zm = (
                        2 * eps_i * eps_km / (eps_i + eps_km)
                        if (eps_i + eps_km) > 0
                        else 0
                    )

                    # 係数計算（等方格子）
                    h2 = self.h**2
                    ax = eps_xp / h2
                    bx = eps_xm / h2
                    ay = eps_yp / h2
                    by = eps_ym / h2
                    az = eps_zp / h2
                    bz = eps_zm / h2

                    A = ax + bx + ay + by + az + bz

                    # 右辺の計算
                    B = (
                        ax * phi[i + 1, j, k]
                        + bx * phi[i - 1, j, k]
                        + ay * phi[i, j + 1, k]
                        + by * phi[i, j - 1, k]
                        + az * phi[i, j, k + 1]
                        + bz * phi[i, j, k - 1]
                        - rho[i, j, k] / self.epsilon_0
                    )

                    # SOR更新
                    if A != 0:
                        phi_new[i, j, k] = (1 - self.omega) * phi[
                            i, j, k
                        ] + self.omega * (B / A)

        return phi_new

    def apply_boundary_conditions(self, phi: np.ndarray) -> np.ndarray:
        """境界条件を適用

        基本的なNeumann/Dirichlet境界条件と周期境界条件に対応
        """
        phi_new = phi.copy()
        bc = self.boundary_conditions

        # z方向の境界条件
        if bc.get("z_bottom", {}).get("type") == "neumann":
            value = bc["z_bottom"].get("value", 0.0)
            # ∂φ/∂z = value を中心差分で近似
            phi_new[:, :, 0] = phi_new[:, :, 1] - value * self.h
        elif bc.get("z_bottom", {}).get("type") == "dirichlet":
            value = bc["z_bottom"].get("value", 0.0)
            phi_new[:, :, 0] = value

        # z_topの境界条件
        if bc.get("z_top", {}).get("type") == "neumann":
            value = bc["z_top"].get("value", 0.0)
            phi_new[:, :, -1] = phi_new[:, :, -2] + value * self.h
        elif bc.get("z_top", {}).get("type") == "dirichlet":
            value = bc["z_top"].get("value", 0.0)
            phi_new[:, :, -1] = value

        # x方向の境界条件
        if bc.get("x_sides", {}).get("type") == "neumann":
            value = bc["x_sides"].get("value", 0.0)
            phi_new[0, :, :] = phi_new[1, :, :] - value * self.h
            phi_new[-1, :, :] = phi_new[-2, :, :] + value * self.h
        elif bc.get("x_sides", {}).get("type") == "dirichlet":
            value = bc["x_sides"].get("value", 0.0)
            phi_new[0, :, :] = value
            phi_new[-1, :, :] = value
        elif bc.get("x_sides", {}).get("type") == "periodic":
            phi_new[0, :, :] = phi_new[-2, :, :]
            phi_new[-1, :, :] = phi_new[1, :, :]

        # y方向の境界条件
        if bc.get("y_sides", {}).get("type") == "neumann":
            value = bc["y_sides"].get("value", 0.0)
            phi_new[:, 0, :] = phi_new[:, 1, :] - value * self.h
            phi_new[:, -1, :] = phi_new[:, -2, :] + value * self.h
        elif bc.get("y_sides", {}).get("type") == "dirichlet":
            value = bc["y_sides"].get("value", 0.0)
            phi_new[:, 0, :] = value
            phi_new[:, -1, :] = value
        elif bc.get("y_sides", {}).get("type") == "periodic":
            phi_new[:, 0, :] = phi_new[:, -2, :]
            phi_new[:, -1, :] = phi_new[:, 1, :]

        return phi_new

    def apply_surface_boundary(self, phi: np.ndarray) -> np.ndarray:
        """表面での混合境界条件を適用

        電極がある位置: Dirichlet境界条件（電圧固定）
        電極がない位置: Neumann境界条件（∂ϕ/∂z = 0）

        Parameters
        ----------
        phi : np.ndarray
            ポテンシャル分布

        Returns
        -------
        phi_new : np.ndarray
            境界条件適用後のポテンシャル分布
        """
        phi_new = phi.copy()
        k_surface = -1  # 表面のz方向インデックス

        if self.electrode_mask is None or self.electrode_voltages is None:
            # 電極情報がない場合はデフォルトのNeumann境界条件
            default_value = self.boundary_conditions.get("z_top", {}).get(
                "default_neumann_value", 0.0
            )
            if default_value == 0.0:
                phi_new[:, :, k_surface] = phi_new[:, :, k_surface - 1]
            else:
                phi_new[:, :, k_surface] = (
                    phi_new[:, :, k_surface - 1] + default_value * self.h
                )
            return phi_new

        # 各グリッド点で境界条件を適用
        for i in range(self.nx):
            for j in range(self.ny):
                if self.electrode_mask[i, j, k_surface]:
                    # 電極がある場合: Dirichlet境界条件
                    phi_new[i, j, k_surface] = self.electrode_voltages[i, j, k_surface]
                else:
                    # 電極がない場合: Neumann境界条件 (∂ϕ/∂z = 0)
                    default_value = self.boundary_conditions.get("z_top", {}).get(
                        "default_neumann_value", 0.0
                    )
                    if default_value == 0.0:
                        phi_new[i, j, k_surface] = phi_new[i, j, k_surface - 1]
                    else:
                        phi_new[i, j, k_surface] = (
                            phi_new[i, j, k_surface - 1] + default_value * self.h
                        )

        return phi_new

    def compute_residual(self, phi: np.ndarray, rho: np.ndarray) -> float:
        """残差を計算

        L2ノルムを使用
        """
        residual_array = np.zeros_like(phi)

        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                for k in range(1, self.nz - 1):
                    # ラプラシアンを計算（調和平均を使用）
                    eps_i = self.epsilon[i, j, k]
                    eps_ip = self.epsilon[i + 1, j, k]
                    eps_im = self.epsilon[i - 1, j, k]
                    eps_jp = self.epsilon[i, j + 1, k]
                    eps_jm = self.epsilon[i, j - 1, k]
                    eps_kp = self.epsilon[i, j, k + 1]
                    eps_km = self.epsilon[i, j, k - 1]

                    eps_xp = (
                        2 * eps_i * eps_ip / (eps_i + eps_ip)
                        if (eps_i + eps_ip) > 0
                        else 0
                    )
                    eps_xm = (
                        2 * eps_i * eps_im / (eps_i + eps_im)
                        if (eps_i + eps_im) > 0
                        else 0
                    )
                    eps_yp = (
                        2 * eps_i * eps_jp / (eps_i + eps_jp)
                        if (eps_i + eps_jp) > 0
                        else 0
                    )
                    eps_ym = (
                        2 * eps_i * eps_jm / (eps_i + eps_jm)
                        if (eps_i + eps_jm) > 0
                        else 0
                    )
                    eps_zp = (
                        2 * eps_i * eps_kp / (eps_i + eps_kp)
                        if (eps_i + eps_kp) > 0
                        else 0
                    )
                    eps_zm = (
                        2 * eps_i * eps_km / (eps_i + eps_km)
                        if (eps_i + eps_km) > 0
                        else 0
                    )

                    # 等方格子での計算
                    laplacian = (
                        eps_xp * (phi[i + 1, j, k] - phi[i, j, k])
                        - eps_xm * (phi[i, j, k] - phi[i - 1, j, k])
                        + eps_yp * (phi[i, j + 1, k] - phi[i, j, k])
                        - eps_ym * (phi[i, j, k] - phi[i, j - 1, k])
                        + eps_zp * (phi[i, j, k + 1] - phi[i, j, k])
                        - eps_zm * (phi[i, j, k] - phi[i, j, k - 1])
                    ) / self.h**2

                    # ポアソン方程式の残差: -∇⋅(ε∇ϕ) - ρ/ε₀
                    residual_array[i, j, k] = -laplacian - rho[i, j, k] / self.epsilon_0

        # L2ノルム（等方格子）
        return np.sqrt(np.mean(residual_array**2)) * self.h**2
