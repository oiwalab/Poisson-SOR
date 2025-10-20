"""PoissonSolverとStructureManagerのテストケース

基本的な機能の動作確認と検証
"""

import numpy as np
import pytest
import sys
import tempfile
import yaml
from pathlib import Path

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poisson_solver import PoissonSolver
from structure_manager import StructureManager


def test_uniform_dielectric_neumann():
    """一様誘電率・Neumann境界条件でのテスト

    ρ=0の場合、Neumann境界条件では全域で一定のポテンシャルになるはず
    """
    # 小さなグリッド
    nx, ny, nz = 10, 10, 10
    h = 1e-9  # 1nm (等方格子)

    # 一様な誘電率（Si）
    epsilon = np.ones((nx, ny, nz)) * 11.7

    # Neumann境界条件（全て∂φ/∂n = 0）
    boundary_conditions = {
        "z_top": {"type": "neumann", "value": 0.0},
        "z_bottom": {"type": "neumann", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    # ソルバ初期化
    solver = PoissonSolver(
        epsilon=epsilon,
        grid_spacing=h,
        boundary_conditions=boundary_conditions,
        omega=1.5,
        tolerance=1e-6,
        max_iterations=1000,
    )

    # 解く（電荷密度ゼロ）
    phi, info = solver.solve()

    # ρ=0、Neumann境界なので、ポテンシャルは一定値になるはず
    assert phi.std() < 1e-6, (
        "Potential should be constant with zero charge and Neumann BC"
    )
    assert info["converged"], "Solver should converge"
    assert info["iterations"] >= 1


def test_parallel_plate_capacitor():
    """平行平板コンデンサのテスト（簡易版）

    z方向のみ変化する1D問題として近似
    """
    # 等方格子（1D問題のため小さめのグリッド）
    nx, ny, nz = 3, 3, 11
    h = 2e-9  # 2nm (等方格子)

    # 一様な誘電率
    epsilon = np.ones((nx, ny, nz)) * 3.9  # SiO2

    # 境界条件: 上下でDirichlet（電圧固定）
    boundary_conditions = {
        "z_top": {"type": "dirichlet", "value": 1.0},  # 1V
        "z_bottom": {"type": "dirichlet", "value": 0.0},  # 0V
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    # ソルバ初期化
    solver = PoissonSolver(
        epsilon=epsilon,
        grid_spacing=h,
        boundary_conditions=boundary_conditions,
        omega=1.5,
        tolerance=1e-8,
        max_iterations=5000,
    )

    # Dirichlet境界条件を手動で設定する初期条件
    phi_initial = np.zeros((nx, ny, nz))
    phi_initial[:, :, 0] = 0.0
    phi_initial[:, :, -1] = 1.0

    # 解く
    phi, info = solver.solve(phi_initial=phi_initial)

    # NaNチェック
    assert not np.isnan(phi).any(), "Solution should not contain NaN"

    # 境界条件が正しく設定されているかチェック
    assert np.abs(phi[:, :, 0].mean() - 0.0) < 1e-6, "Bottom boundary should be 0V"
    assert np.abs(phi[:, :, -1].mean() - 1.0) < 1e-6, "Top boundary should be 1V"

    # 解析解と比較: 平行平板では φ(z) = V_top * z / L （線形）
    z_coords = np.arange(nz) * h
    L = (nz - 1) * h  # 計算領域のz方向長さ
    phi_analytical = z_coords / L  # 0から1までの線形

    # 中央の点で比較
    phi_numerical = phi[1, 1, :]

    print(f"\nParallel plate capacitor test:")
    print(f"Converged: {info['converged']}, Iterations: {info['iterations']}")
    print(f"Max absolute error: {np.abs(phi_numerical - phi_analytical).max():.6e}")

    # 内部の点で解析解との誤差をチェック（境界を除く）
    error = np.abs(phi_numerical[1:-1] - phi_analytical[1:-1])
    max_error = error.max()
    assert max_error < 0.01, (
        f"Max error {max_error:.6e} should be < 0.01 (1% of voltage range)"
    )


def test_structure_manager_valid_config():
    """StructureManagerで正常な設定の読み込みテスト"""
    # 正常なケース
    valid_config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {"material": "Si", "z_range": [0, 50e-9], "epsilon_r": 11.7},
            {"material": "SiO2", "z_range": [50e-9, 100e-9], "epsilon_r": 3.9},
        ],
        "boundary_conditions": {
            "z_top": {"type": "neumann", "value": 0.0},
            "z_bottom": {"type": "neumann", "value": 0.0},
            "x_sides": {"type": "neumann", "value": 0.0},
            "y_sides": {"type": "neumann", "value": 0.0},
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(valid_config, f)
        temp_file = f.name

    try:
        manager = StructureManager()
        manager.load_from_yaml(temp_file)

        # 誘電率分布をチェック
        assert manager.epsilon_array is not None
        assert manager.epsilon_array.shape == (11, 11, 11)  # 100nm / 10nm + 1

        # 層の境界での誘電率をチェック
        k_interface = int(50e-9 / 10e-9)
        eps_si = manager.epsilon_array[5, 5, k_interface - 1]
        eps_sio2 = manager.epsilon_array[5, 5, k_interface]
        assert np.abs(eps_si - 11.7) < 1e-6
        assert np.abs(eps_sio2 - 3.9) < 1e-6

    finally:
        Path(temp_file).unlink()


def test_structure_manager_layer_gap():
    """StructureManagerの層の隙間検出テスト"""
    gap_config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {"material": "Si", "z_range": [0, 40e-9], "epsilon_r": 11.7},
            {"material": "SiO2", "z_range": [50e-9, 100e-9], "epsilon_r": 3.9},  # 隙間
        ],
        "boundary_conditions": {},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(gap_config, f)
        temp_file = f.name

    try:
        manager = StructureManager()
        with pytest.raises(ValueError, match="Gap detected"):
            manager.load_from_yaml(temp_file)
    finally:
        Path(temp_file).unlink()


def test_structure_manager_layer_overlap():
    """StructureManagerの層の重複検出テスト"""
    overlap_config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {"material": "Si", "z_range": [0, 60e-9], "epsilon_r": 11.7},
            {"material": "SiO2", "z_range": [50e-9, 100e-9], "epsilon_r": 3.9},  # 重複
        ],
        "boundary_conditions": {},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(overlap_config, f)
        temp_file = f.name

    try:
        manager = StructureManager()
        with pytest.raises(ValueError, match="Overlap detected"):
            manager.load_from_yaml(temp_file)
    finally:
        Path(temp_file).unlink()


def test_electrode_overlap_detection():
    """電極重複検出のテスト"""
    overlap_config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {"material": "Si", "z_range": [0, 100e-9], "epsilon_r": 11.7},
        ],
        "electrodes": [
            {
                "name": "gate1",
                "shape": "rectangle",
                "x_range": [10e-9, 40e-9],
                "y_range": [10e-9, 40e-9],
                "z_position": 100e-9,
                "voltage": -0.5,
            },
            {
                "name": "gate2",
                "shape": "rectangle",
                "x_range": [30e-9, 60e-9],  # gate1と重複
                "y_range": [30e-9, 60e-9],
                "z_position": 100e-9,
                "voltage": -1.0,
            },
        ],
        "boundary_conditions": {},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(overlap_config, f)
        temp_file = f.name

    try:
        manager = StructureManager()
        with pytest.raises(ValueError, match="Electrode overlap detected"):
            manager.load_from_yaml(temp_file)
    finally:
        Path(temp_file).unlink()


def test_electrode_no_overlap():
    """電極重複なしのテスト"""
    no_overlap_config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {"material": "Si", "z_range": [0, 100e-9], "epsilon_r": 11.7},
        ],
        "electrodes": [
            {
                "name": "gate1",
                "shape": "rectangle",
                "x_range": [10e-9, 30e-9],
                "y_range": [10e-9, 30e-9],
                "z_position": 100e-9,
                "voltage": -0.5,
            },
            {
                "name": "gate2",
                "shape": "rectangle",
                "x_range": [40e-9, 60e-9],  # gate1と重複なし
                "y_range": [40e-9, 60e-9],
                "z_position": 100e-9,
                "voltage": -1.0,
            },
        ],
        "boundary_conditions": {},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(no_overlap_config, f)
        temp_file = f.name

    try:
        manager = StructureManager()
        manager.load_from_yaml(temp_file)  # エラーが出ないことを確認

        # 電極マスクのチェック
        assert manager.electrode_mask is not None
        assert manager.electrode_mask.any(), (
            "Electrode mask should have some True values"
        )

        # 電極電圧のチェック
        assert manager.electrode_voltages is not None
        assert manager.electrode_voltages.min() == -1.0
        assert manager.electrode_voltages.max() == 0.0  # 電極がない場所は0V

    finally:
        Path(temp_file).unlink()


def test_electrode_volume():
    """電極を3Dボリュームとして扱うテスト"""
    # 小さなグリッド
    nx, ny, nz = 11, 11, 11
    h = 10e-9  # 10nm (等方格子)

    # 一様な誘電率
    epsilon = np.ones((nx, ny, nz)) * 11.7

    # 電極マスク: 中央に3Dボリュームとして電極（上部2層分）
    electrode_mask = np.zeros((nx, ny, nz), dtype=bool)
    k_electrode_bottom = 9  # 電極底面のz-index
    electrode_mask[4:7, 4:7, k_electrode_bottom:] = True  # 中央3x3、上部2層

    # 電極電圧: -0.5V
    electrode_voltages = np.zeros((nx, ny, nz))
    electrode_voltages[4:7, 4:7, k_electrode_bottom:] = -0.5

    # 境界条件（z_topは全面Neumann）
    boundary_conditions = {
        "z_top": {"type": "neumann", "value": 0.0},
        "z_bottom": {"type": "neumann", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    # ソルバ初期化
    solver = PoissonSolver(
        epsilon=epsilon,
        grid_spacing=h,
        boundary_conditions=boundary_conditions,
        omega=1.0,
        tolerance=1e-6,
        max_iterations=10000,
        electrode_mask=electrode_mask,
        electrode_voltages=electrode_voltages,
    )

    # 初期条件を設定（電極電圧に近い値から開始）
    phi_initial = np.zeros((nx, ny, nz))
    phi_initial[electrode_mask] = -0.5

    # 解く
    phi, info = solver.solve(phi_initial=phi_initial)

    # 電極領域全体の電圧をチェック
    electrode_phi = phi[electrode_mask]
    assert np.allclose(electrode_phi, -0.5, atol=1e-6), (
        f"Electrode potential should be -0.5V, but got mean={electrode_phi.mean():.6f}"
    )

    # 電極外部の電位は電極電圧より大きい（0Vに近い）
    non_electrode_phi = phi[0, 0, 5]  # 中央層の端点
    assert non_electrode_phi > electrode_phi.mean(), (
        "Non-electrode region should have higher potential than electrode"
    )

    # NaNチェック
    assert not np.isnan(phi).any(), "Solution should not contain NaN"

    assert info["converged"], "Solver should converge"
