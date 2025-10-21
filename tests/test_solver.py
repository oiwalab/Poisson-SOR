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
    # 小さなグリッド (新座標系: shape = (nz, nx, ny))
    nz, nx, ny = 10, 10, 10
    h = 1e-9  # 1nm (等方格子)

    # 一様な誘電率（Si） - 配列shape: (nz, nx, ny)
    epsilon = np.ones((nz, nx, ny)) * 11.7

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

    新座標系:
    - z_top (k=0, z=0nm): 1V
    - z_bottom (k=nz-1, z=-20nm): 0V
    """
    # 等方格子（1D問題のため小さめのグリッド）
    # 配列shape: (nz, nx, ny)
    nz, nx, ny = 11, 3, 3
    h = 2e-9  # 2nm (等方格子)

    # 一様な誘電率 - 配列shape: (nz, nx, ny)
    epsilon = np.ones((nz, nx, ny)) * 3.9  # SiO2

    # 境界条件: 上下でDirichlet（電圧固定）
    boundary_conditions = {
        "z_top": {"type": "dirichlet", "value": 1.0},  # 表面 (k=0): 1V
        "z_bottom": {"type": "dirichlet", "value": 0.0},  # 底面 (k=nz-1): 0V
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
    # 配列shape: (nz, nx, ny)
    phi_initial = np.zeros((nz, nx, ny))
    phi_initial[0, :, :] = 1.0  # z_top (k=0): 1V
    phi_initial[-1, :, :] = 0.0  # z_bottom (k=nz-1): 0V

    # 解く
    phi, info = solver.solve(phi_initial=phi_initial)

    # NaNチェック
    assert not np.isnan(phi).any(), "Solution should not contain NaN"

    # 境界条件が正しく設定されているかチェック
    assert np.abs(phi[0, :, :].mean() - 1.0) < 1e-6, "Top boundary (k=0) should be 1V"
    assert np.abs(phi[-1, :, :].mean() - 0.0) < 1e-6, (
        "Bottom boundary (k=nz-1) should be 0V"
    )

    # 解析解と比較: 平行平板では φ(k) = V_top * (1 - k/K) （線形、1→0）
    # k=0 → φ=1, k=nz-1 → φ=0
    k_coords = np.arange(nz)
    K = nz - 1
    phi_analytical = 1.0 - k_coords / K  # 1から0までの線形

    # 中央の点で比較 (配列shape: (nz, nx, ny))
    phi_numerical = phi[:, 1, 1]

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
    """StructureManagerで正常な設定の読み込みテスト

    新座標系:
    - SiO2: z=0nm → z=-50nm
    - Si: z=-50nm → z=-100nm
    """
    # 正常なケース（新座標系: z_range = [z_max, z_min]）
    valid_config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {"material": "SiO2", "z_range": [0, -50e-9], "epsilon_r": 3.9},  # 表面側
            {
                "material": "Si",
                "z_range": [-50e-9, -100e-9],
                "epsilon_r": 11.7,
            },  # 底面側
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

        # 誘電率分布をチェック（配列shape: (nz, nx, ny)）
        assert manager.epsilon_array is not None
        assert manager.epsilon_array.shape == (11, 11, 11)  # 100nm / 10nm + 1

        # 層の境界での誘電率をチェック
        # z=-50nm → k=5 (k = -z/h = -(-50e-9)/10e-9 = 5)
        k_interface = int(-(-50e-9) / 10e-9)
        eps_sio2 = manager.epsilon_array[k_interface - 1, 5, 5]  # k=4 (SiO2側)
        eps_si = manager.epsilon_array[k_interface, 5, 5]  # k=5 (Si側)
        assert np.abs(eps_sio2 - 3.9) < 1e-6
        assert np.abs(eps_si - 11.7) < 1e-6

    finally:
        Path(temp_file).unlink()


def test_structure_manager_layer_gap():
    """StructureManagerの層の隙間検出テスト

    新座標系で隙間を検出
    """
    gap_config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {
                "material": "SiO2",
                "z_range": [0, -40e-9],
                "epsilon_r": 3.9,
            },  # z=0 → z=-40nm
            {
                "material": "Si",
                "z_range": [-50e-9, -100e-9],
                "epsilon_r": 11.7,
            },  # z=-50nm → z=-100nm (隙間あり)
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
    """StructureManagerの層の重複検出テスト

    新座標系で重複を検出
    """
    overlap_config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {
                "material": "SiO2",
                "z_range": [0, -60e-9],
                "epsilon_r": 3.9,
            },  # z=0 → z=-60nm
            {
                "material": "Si",
                "z_range": [-50e-9, -100e-9],
                "epsilon_r": 11.7,
            },  # z=-50nm → z=-100nm (重複あり)
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
    """電極重複検出のテスト

    新座標系: 電極は表面(z=0)に配置
    """
    overlap_config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {"material": "Si", "z_range": [0, -100e-9], "epsilon_r": 11.7},
        ],
        "electrodes": [
            {
                "name": "gate1",
                "shape": "rectangle",
                "x_range": [10e-9, 40e-9],
                "y_range": [10e-9, 40e-9],
                "z_position": -10e-9,  # 電極底面（負の値）
                "voltage": -0.5,
            },
            {
                "name": "gate2",
                "shape": "rectangle",
                "x_range": [30e-9, 60e-9],  # gate1と重複
                "y_range": [30e-9, 60e-9],
                "z_position": -10e-9,  # 電極底面（負の値）
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
    """電極重複なしのテスト

    新座標系: 電極は表面(z=0)に配置
    """
    no_overlap_config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {"material": "Si", "z_range": [0, -100e-9], "epsilon_r": 11.7},
        ],
        "electrodes": [
            {
                "name": "gate1",
                "shape": "rectangle",
                "x_range": [10e-9, 30e-9],
                "y_range": [10e-9, 30e-9],
                "z_position": -10e-9,  # 電極底面（負の値）
                "voltage": -0.5,
            },
            {
                "name": "gate2",
                "shape": "rectangle",
                "x_range": [40e-9, 60e-9],  # gate1と重複なし
                "y_range": [40e-9, 60e-9],
                "z_position": -10e-9,  # 電極底面（負の値）
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

        # 電極マスクのチェック（配列shape: (nz, nx, ny)）
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
    """電極を3Dボリュームとして扱うテスト

    新座標系:
    - 配列shape: (nz, nx, ny)
    - 電極は表面(k=0, z=0)から下に伸びる3Dボリューム
    """
    # 小さなグリッド (配列shape: (nz, nx, ny))
    nz, nx, ny = 11, 11, 11
    h = 10e-9  # 10nm (等方格子)

    # 一様な誘電率
    epsilon = np.ones((nz, nx, ny)) * 11.7

    # 電極マスク: 中央に3Dボリュームとして電極（上部2層分: k=0,1）
    # 配列shape: (nz, nx, ny)
    electrode_mask = np.zeros((nz, nx, ny), dtype=bool)
    k_electrode_top = 0  # 表面
    k_electrode_bottom = 1  # 電極底面のz-index (2層分: k=0,1)
    electrode_mask[k_electrode_top : k_electrode_bottom + 1, 4:7, 4:7] = (
        True  # 上部2層、中央3x3
    )

    # 電極電圧: -0.5V
    electrode_voltages = np.zeros((nz, nx, ny))
    electrode_voltages[k_electrode_top : k_electrode_bottom + 1, 4:7, 4:7] = -0.5

    # 境界条件（全面Neumann）
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
        omega=1.8,
        tolerance=1e-6,
        max_iterations=10000,
        electrode_mask=electrode_mask,
        electrode_voltages=electrode_voltages,
    )

    # 初期条件を設定（電極電圧に近い値から開始）
    phi_initial = np.zeros((nz, nx, ny))
    phi_initial[electrode_mask] = -0.5

    # 解く
    phi, info = solver.solve(phi_initial=phi_initial)

    # 電極領域全体の電圧をチェック
    electrode_phi = phi[electrode_mask]
    assert np.allclose(electrode_phi, -0.5, atol=1e-6), (
        f"Electrode potential should be -0.5V, but got mean={electrode_phi.mean():.6f}"
    )

    # 電極外部の電位は電極電圧より大きい（0Vに近い）
    # 配列shape: (nz, nx, ny) → [k, i, j]
    non_electrode_phi = phi[5, 0, 0]  # 中央層の端点
    assert non_electrode_phi > electrode_phi.mean(), (
        "Non-electrode region should have higher potential than electrode"
    )

    # NaNチェック
    assert not np.isnan(phi).any(), "Solution should not contain NaN"

    assert info["converged"], "Solver should converge"
