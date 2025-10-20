"""半導体構造管理クラス

材料層構造、電極配置、誘電率分布を管理
"""

import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
# from pathlib import Path


class StructureManager:
    """半導体構造管理クラス

    YAMLファイルから構造定義を読み込み、
    誘電率分布、電極マスク、電圧分布を生成
    """

    def __init__(self):
        self.config: Dict = {}
        self.layers: List[Dict] = []
        self.electrodes: List[Dict] = []

        # 計算グリッド
        self.nx: int = 0
        self.ny: int = 0
        self.nz: int = 0
        self.h: float = 0.0  # 等方格子間隔

        # 計算領域のサイズ
        self.size_x: float = 0.0
        self.size_y: float = 0.0
        self.size_z: float = 0.0

        # 誘電率分布 (nx, ny, nz)
        self.epsilon_array: Optional[np.ndarray] = None

        # 電極マスク (nx, ny, nz) - True=電極あり
        self.electrode_mask: Optional[np.ndarray] = None

        # 電極電圧 (nx, ny, nz)
        self.electrode_voltages: Optional[np.ndarray] = None

        # 電荷密度 (nx, ny, nz)
        self.charge_density: Optional[np.ndarray] = None

    def load_from_yaml(self, yaml_path: str) -> None:
        """YAMLファイルから構造定義を読み込み

        Parameters
        ----------
        yaml_path : str
            YAML設定ファイルのパス
        """
        with open(yaml_path, "r") as f:
            self.config = yaml.safe_load(f)

        # 設定を抽出
        self.layers = self.config.get("layers", [])
        self.electrodes = self.config.get("electrodes", [])

        # 計算領域の設定
        domain = self.config.get("domain", {})
        size = domain.get("size", [100e-9, 100e-9, 100e-9])
        grid_spacing = domain.get("grid_spacing", 10e-9)

        self.size_x, self.size_y, self.size_z = size
        self.h = grid_spacing  # 等方格子間隔

        # グリッド点数を計算
        self.nx = int(self.size_x / self.h) + 1
        self.ny = int(self.size_y / self.h) + 1
        self.nz = int(self.size_z / self.h) + 1

        # 配列を初期化
        self._initialize_arrays()

        # 層構造をバリデーション
        self._validate_layers()

        # 誘電率分布を生成
        self.generate_epsilon_array()

        # 電極構造を生成
        if self.electrodes:
            self.generate_electrode_mask()
            self.get_electrode_voltages()
            self.check_electrode_overlap()

    def _initialize_arrays(self) -> None:
        """内部配列を初期化"""
        self.epsilon_array = np.ones((self.nx, self.ny, self.nz))
        self.electrode_mask = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        self.electrode_voltages = np.zeros((self.nx, self.ny, self.nz))
        self.charge_density = np.zeros((self.nx, self.ny, self.nz))

    def _validate_layers(self) -> None:
        """層構造の妥当性をチェック

        - z方向の範囲が重複していないか
        - z方向の範囲に隙間がないか
        - z方向の範囲が計算領域内か

        Raises
        ------
        ValueError
            層構造に問題がある場合
        """
        if not self.layers:
            return

        # z_rangeでソート
        sorted_layers = sorted(self.layers, key=lambda x: x.get("z_range", [0, 0])[0])

        # 計算領域の範囲
        domain_z_min = 0.0
        domain_z_max = self.size_z

        # 最初の層が0から始まるかチェック
        first_z_min = sorted_layers[0].get("z_range", [0, 0])[0]
        if abs(first_z_min - domain_z_min) > 1e-12:
            raise ValueError(
                f"First layer must start at z=0, but starts at z={first_z_min * 1e9:.2f} nm"
            )

        # 各層をチェック
        for i, layer in enumerate(sorted_layers):
            z_range = layer.get("z_range", [0, 0])
            material = layer.get("material", "Unknown")

            # 範囲の妥当性チェック
            if z_range[0] >= z_range[1]:
                raise ValueError(
                    f"Layer '{material}': Invalid z_range {z_range}. "
                    f"z_min must be less than z_max"
                )

            # 計算領域内かチェック
            if z_range[0] < domain_z_min or z_range[1] > domain_z_max:
                raise ValueError(
                    f"Layer '{material}': z_range [{z_range[0] * 1e9:.2f}, {z_range[1] * 1e9:.2f}] nm "
                    f"is outside domain [0, {domain_z_max * 1e9:.2f}] nm"
                )

            # 次の層との連続性をチェック
            if i < len(sorted_layers) - 1:
                next_layer = sorted_layers[i + 1]
                next_z_min = next_layer.get("z_range", [0, 0])[0]
                current_z_max = z_range[1]

                # 隙間チェック（数値誤差を考慮して1e-12 m = 1e-3 nm以下なら許容）
                gap = next_z_min - current_z_max
                if abs(gap) > 1e-12:
                    if gap > 0:
                        raise ValueError(
                            f"Gap detected between layer '{material}' "
                            f"(ends at {current_z_max * 1e9:.2f} nm) and "
                            f"next layer (starts at {next_z_min * 1e9:.2f} nm). "
                            f"Gap size: {gap * 1e9:.2f} nm"
                        )
                    else:
                        raise ValueError(
                            f"Overlap detected between layer '{material}' "
                            f"(ends at {current_z_max * 1e9:.2f} nm) and "
                            f"next layer (starts at {next_z_min * 1e9:.2f} nm). "
                            f"Overlap size: {-gap * 1e9:.2f} nm"
                        )

        # 最後の層が計算領域の終わりまでカバーしているかチェック
        last_z_max = sorted_layers[-1].get("z_range", [0, 0])[1]
        if abs(last_z_max - domain_z_max) > 1e-12:
            raise ValueError(
                f"Last layer must end at z={domain_z_max * 1e9:.2f} nm, "
                f"but ends at z={last_z_max * 1e9:.2f} nm"
            )

    def generate_epsilon_array(self) -> np.ndarray:
        """誘電率分布を生成

        層構造に基づいて3D誘電率配列を生成

        Returns
        -------
        epsilon_array : np.ndarray
            比誘電率分布 (nx, ny, nz)
        """
        if self.epsilon_array is None:
            self._initialize_arrays()

        # デフォルトは真空（εr=1）
        self.epsilon_array[:, :, :] = 1.0

        # 各層の誘電率を設定
        for layer in self.layers:
            # material = layer.get('material', 'Unknown')
            z_range = layer.get("z_range", [0, 0])
            epsilon_r = layer.get("epsilon_r", 1.0)

            # z方向のインデックス範囲を計算
            k_min = int(z_range[0] / self.h)
            k_max = int(z_range[1] / self.h)

            # 範囲チェック
            k_min = max(0, min(k_min, self.nz - 1))
            k_max = max(0, min(k_max, self.nz - 1))

            # 誘電率を設定
            self.epsilon_array[:, :, k_min : k_max + 1] = epsilon_r

        return self.epsilon_array

    def generate_electrode_mask(self) -> np.ndarray:
        """電極マスクを生成

        電極がある位置をTrueとする3Dブーリアン配列

        Returns
        -------
        electrode_mask : np.ndarray
            電極マスク (nx, ny, nz), dtype=bool
        """
        if self.electrode_mask is None:
            self._initialize_arrays()

        self.electrode_mask[:, :, :] = False

        for electrode in self.electrodes:
            shape = electrode.get("shape", "rectangle")

            if shape == "rectangle":
                self._add_rectangle_electrode(electrode)
            elif shape == "from_file":
                raise NotImplementedError(
                    "Loading electrode pattern from file is not yet implemented. "
                    "Use 'rectangle' shape for now."
                )
            else:
                raise ValueError(f"Unknown electrode shape: {shape}")

        return self.electrode_mask

    def _add_rectangle_electrode(self, electrode: Dict) -> None:
        """矩形電極をマスクに追加（3Dボリュームとして）

        Parameters
        ----------
        electrode : Dict
            電極定義（x_range, y_range, z_position等）
            z_positionは電極の底面位置、そこから計算領域の上端まで電極領域とする
        """
        x_range = electrode.get("x_range", [0, 0])
        y_range = electrode.get("y_range", [0, 0])
        z_position = electrode.get("z_position", 0)

        # インデックスに変換
        i_min = int(x_range[0] / self.h)
        i_max = int(x_range[1] / self.h)
        j_min = int(y_range[0] / self.h)
        j_max = int(y_range[1] / self.h)
        k_min = int(z_position / self.h)  # 電極底面
        k_max = self.nz - 1  # 計算領域の上端（z_top）

        # 範囲チェック
        i_min = max(0, min(i_min, self.nx - 1))
        i_max = max(0, min(i_max, self.nx - 1))
        j_min = max(0, min(j_min, self.ny - 1))
        j_max = max(0, min(j_max, self.ny - 1))
        k_min = max(0, min(k_min, self.nz - 1))
        k_max = max(0, min(k_max, self.nz - 1))

        # マスクを設定（z_position以上の3Dボリューム）
        self.electrode_mask[i_min : i_max + 1, j_min : j_max + 1, k_min : k_max + 1] = True

    def get_electrode_voltages(self) -> np.ndarray:
        """電極電圧分布を取得

        各グリッド点での電極電圧を設定

        Returns
        -------
        electrode_voltages : np.ndarray
            電極電圧 (V), shape=(nx, ny, nz)
        """
        if self.electrode_voltages is None:
            self._initialize_arrays()

        self.electrode_voltages[:, :, :] = 0.0

        for electrode in self.electrodes:
            voltage = electrode.get("voltage", 0.0)
            shape = electrode.get("shape", "rectangle")

            if shape == "rectangle":
                x_range = electrode.get("x_range", [0, 0])
                y_range = electrode.get("y_range", [0, 0])
                z_position = electrode.get("z_position", 0)

                # インデックスに変換
                i_min = int(x_range[0] / self.h)
                i_max = int(x_range[1] / self.h)
                j_min = int(y_range[0] / self.h)
                j_max = int(y_range[1] / self.h)
                k_min = int(z_position / self.h)  # 電極底面
                k_max = self.nz - 1  # 計算領域の上端（z_top）

                # 範囲チェック
                i_min = max(0, min(i_min, self.nx - 1))
                i_max = max(0, min(i_max, self.nx - 1))
                j_min = max(0, min(j_min, self.ny - 1))
                j_max = max(0, min(j_max, self.ny - 1))
                k_min = max(0, min(k_min, self.nz - 1))
                k_max = max(0, min(k_max, self.nz - 1))

                # 電圧を設定（z_position以上の3Dボリューム）
                self.electrode_voltages[i_min : i_max + 1, j_min : j_max + 1, k_min : k_max + 1] = (
                    voltage
                )

        return self.electrode_voltages

    def check_electrode_overlap(self) -> None:
        """電極の重複をチェック

        重複がある場合はエラーを出す

        Raises
        ------
        ValueError
            電極が重複している場合
        """
        # 各z位置での電極数をカウント
        electrode_count = np.zeros((self.nx, self.ny, self.nz), dtype=int)

        for electrode in self.electrodes:
            shape = electrode.get("shape", "rectangle")

            if shape == "rectangle":
                x_range = electrode.get("x_range", [0, 0])
                y_range = electrode.get("y_range", [0, 0])
                z_position = electrode.get("z_position", 0)

                # インデックスに変換
                i_min = int(x_range[0] / self.h)
                i_max = int(x_range[1] / self.h)
                j_min = int(y_range[0] / self.h)
                j_max = int(y_range[1] / self.h)
                k = int(z_position / self.h)

                # 範囲チェック
                i_min = max(0, min(i_min, self.nx - 1))
                i_max = max(0, min(i_max, self.nx - 1))
                j_min = max(0, min(j_min, self.ny - 1))
                j_max = max(0, min(j_max, self.ny - 1))
                k = max(0, min(k, self.nz - 1))

                # カウントを増やす
                electrode_count[i_min : i_max + 1, j_min : j_max + 1, k] += 1

        # 重複チェック
        if np.any(electrode_count > 1):
            overlapping_positions = np.where(electrode_count > 1)
            raise ValueError(
                f"Electrode overlap detected at {len(overlapping_positions[0])} positions. "
                f"First overlap at grid indices: "
                f"({overlapping_positions[0][0]}, {overlapping_positions[1][0]}, {overlapping_positions[2][0]})"
            )

    def set_charge_density(self, rho: np.ndarray) -> None:
        """電荷密度分布を設定

        Parameters
        ----------
        rho : np.ndarray
            電荷密度分布 (C/m^3), shape=(nx, ny, nz)
        """
        if rho.shape != (self.nx, self.ny, self.nz):
            raise ValueError(
                f"Charge density shape {rho.shape} does not match "
                f"grid size ({self.nx}, {self.ny}, {self.nz})"
            )

        self.charge_density = rho.copy()

    def get_grid_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """グリッド座標を取得

        Returns
        -------
        x, y, z : np.ndarray
            各方向の座標配列 (m)
        """
        x = np.arange(self.nx) * self.dx
        y = np.arange(self.ny) * self.dy
        z = np.arange(self.nz) * self.dz

        return x, y, z

    def get_summary(self) -> str:
        """構造の要約を文字列で取得

        Returns
        -------
        summary : str
            構造の要約
        """
        summary = []
        summary.append("=== Structure Summary ===")
        summary.append(f"Grid size: ({self.nx}, {self.ny}, {self.nz})")
        summary.append(
            f"Grid spacing: ({self.dx * 1e9:.2f}, {self.dy * 1e9:.2f}, {self.dz * 1e9:.2f}) nm"
        )
        summary.append(f"Number of layers: {len(self.layers)}")
        summary.append(f"Number of electrodes: {len(self.electrodes)}")

        summary.append("\n--- Layers ---")
        for i, layer in enumerate(self.layers):
            material = layer.get("material", "Unknown")
            z_range = layer.get("z_range", [0, 0])
            epsilon_r = layer.get("epsilon_r", 1.0)
            summary.append(
                f"  {i + 1}. {material}: "
                f"z=[{z_range[0] * 1e9:.1f}, {z_range[1] * 1e9:.1f}] nm, "
                f"εr={epsilon_r:.2f}"
            )

        summary.append("\n--- Electrodes ---")
        for i, electrode in enumerate(self.electrodes):
            name = electrode.get("name", f"electrode_{i + 1}")
            voltage = electrode.get("voltage", 0.0)
            shape = electrode.get("shape", "rectangle")
            summary.append(f"  {i + 1}. {name}: V={voltage:.3f} V, shape={shape}")

        return "\n".join(summary)
