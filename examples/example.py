"""SOR法ポアソンソルバの実行例

YAML設定ファイルから構造を読み込み、ポアソン方程式を解いて結果を可視化
"""

import sys
from pathlib import Path
import numpy as np

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from structure_manager import StructureManager
from poisson_solver import PoissonSolver
import visualizer as vis


def main():
    """メイン実行関数"""
    print("=" * 60)
    print("SOR法による3次元ポアソン方程式ソルバ")
    print("=" * 60)

    # 設定ファイルのパス
    config_path = Path(__file__).parent.parent / 'configs' / 'example.yaml'

    # 構造管理クラスの初期化と設定読み込み
    print("\n[1] 構造定義を読み込み中...")
    manager = StructureManager()
    manager.load_from_yaml(str(config_path))

    # 構造の要約を表示
    print(manager.get_summary())

    # ソルバパラメータの取得
    solver_config = manager.config.get('solver', {})
    omega = solver_config.get('omega', 1.5)
    max_iterations = solver_config.get('max_iterations', 10000)
    tolerance = solver_config.get('tolerance', 1e-6)

    # ソルバの初期化
    print("\n[2] ソルバを初期化中...")
    solver = PoissonSolver(
        epsilon=manager.epsilon_array,
        grid_spacing=(manager.dx, manager.dy, manager.dz),
        boundary_conditions=manager.config.get('boundary_conditions', {}),
        omega=omega,
        tolerance=tolerance,
        max_iterations=max_iterations,
        electrode_mask=manager.electrode_mask,
        electrode_voltages=manager.electrode_voltages,
    )

    print(f"  Grid size: ({manager.nx}, {manager.ny}, {manager.nz})")
    print(f"  Omega: {omega}")
    print(f"  Tolerance: {tolerance:.2e}")
    print(f"  Max iterations: {max_iterations}")

    # ポアソン方程式を解く
    print("\n[3] ポアソン方程式を解いています...")
    print("  (電荷密度ρ=0の場合)")

    phi, info = solver.solve(rho=manager.charge_density)

    # 結果の表示
    print("\n[4] 計算結果:")
    print(f"  収束: {info['converged']}")
    print(f"  反復回数: {info['iterations']}")
    print(f"  最終残差: {info['final_residual']:.2e}")
    print(f"  ポテンシャル範囲: [{phi.min():.4f}, {phi.max():.4f}] V")

    # 座標取得
    x, y, z = manager.get_grid_coordinates()

    # 結果の保存
    print("\n[5] 結果を保存中...")
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # npz形式で保存
    vis.save_results(
        phi=phi,
        x=x,
        y=y,
        z=z,
        info=info,
        save_path=str(results_dir / 'potential_distribution.npz')
    )

    # 可視化
    print("\n[6] 結果を可視化中...")

    # 電極パターン
    print("  - 電極パターン")
    vis.plot_electrode_pattern(
        electrode_mask=manager.electrode_mask,
        electrode_voltages=manager.electrode_voltages,
        x=x,
        y=y,
        z_index=-1,
        save_path=str(figures_dir / 'electrode_pattern.png')
    )

    # 複数のz位置でのポテンシャル分布
    print("  - ポテンシャル分布（複数スライス）")
    nz = manager.nz
    z_indices = [nz // 4, nz // 2, 3 * nz // 4, -1]
    vis.plot_multiple_slices(
        phi=phi,
        x=x,
        y=y,
        z=z,
        z_indices=z_indices,
        electrode_mask=manager.electrode_mask,
        save_path=str(figures_dir / 'potential_slices.png')
    )

    # 収束履歴
    print("  - 収束履歴")
    vis.plot_convergence(
        residual_history=solver.residual_history,
        save_path=str(figures_dir / 'convergence_history.png')
    )

    # 特定の深さでの詳細プロット
    print("  - Si/SiO2界面付近のポテンシャル分布")
    interface_z_index = int(40e-9 / manager.dz)  # Si/SiO2界面
    vis.plot_potential_slice(
        phi=phi,
        x=x,
        y=y,
        z=z,
        z_index=interface_z_index,
        electrode_mask=manager.electrode_mask,
        save_path=str(figures_dir / 'potential_at_interface.png'),
        title='Potential at Si/SiO2 Interface'
    )

    print("\n" + "=" * 60)
    print("計算完了！")
    print(f"結果は {results_dir} に保存されました。")
    print("=" * 60)


if __name__ == '__main__':
    main()
