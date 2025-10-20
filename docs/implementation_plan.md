# SOR法による2次元半導体ポアソンソルバの実装計画

## 概要
半導体ヘテロ構造と電極構造から2DEGのポテンシャルを計算するSOR法ベースのポアソンソルバを実装します。

## 重要な制限事項

### 等方格子のみサポート
**本ソルバは等方格子（dx = dy = dz = h）のみをサポートします。**

- **理由:** SOR法は格子の異方性に対して数値的に不安定であり、異方性比が大きい場合（例: dx/dz > 3）にoverflow/NaNが発生します。
- **設定方法:** YAMLファイルで `grid_spacing` をスカラー値として指定します。
  ```yaml
  domain:
    size: [100e-9, 100e-9, 50e-9]
    grid_spacing: 5e-9  # スカラー値（単位: m）
  ```
- **推奨:** 等方的な格子間隔（dx = dy = dz）を使用してください。

## ディレクトリ構成

```
SOR/
├── docs/                            # ドキュメント
│   ├── implementation_plan.md       # 実装計画書（このファイル）
│   └── CLAUDE.md                    # その他のドキュメント
│
├── src/                             # ソースコード
│   ├── poisson_solver.py            # メインのSORソルバクラス
│   ├── structure_manager.py         # 半導体構造管理クラス
│   └── visualizer.py                # 可視化モジュール
│
├── configs/                         # 設定ファイル
│   └── example.yaml                 # 設定ファイルの例（フィンガーゲート構造）
│
├── tests/                           # テストコード
│   └── test_solver.py               # テストケース
│
├── examples/                        # 実行例
│   └── main.py                      # 実行スクリプト例
│
└── results/                         # 結果保存用ディレクトリ（実行時に自動作成）
    ├── potential_*.npz              # ポテンシャル分布データ
    └── figures/                     # 可視化結果の画像
        ├── potential_slice_*.png    # ポテンシャル分布のスライス画像
        ├── electrode_pattern.png    # 電極パターンの可視化
        └── convergence.png          # 収束履歴のグラフ
```

## ファイル構成

### 1. `poisson_solver.py` - メインのSORソルバクラス
**主な機能:**
- 誘電率が不均一な3Dポアソン方程式 −∇⋅(ε∇ϕ)=ρ を解く
- SOR法の反復計算（緩和パラメータω設定可能）
- 柔軟な境界条件設定（Dirichlet/Neumann/周期境界）
- 表面での混合境界条件対応（電極部分はDirichlet、非電極部分はNeumann）
- 収束判定機能（残差のしきい値、最大反復回数）

**クラス構成:**
- `PoissonSolver`: SORアルゴリズムの実装
  - `solve()`: メインの求解メソッド
  - `apply_boundary_conditions()`: 境界条件の適用
  - `apply_surface_boundary()`: 表面での混合境界条件の適用
  - `compute_residual()`: 残差計算

### 2. `structure_manager.py` - 半導体構造管理クラス
**主な機能:**
- YAMLファイルから構造定義を読み込み
- 誘電率分布の3Dアレイ生成
- 電極構造の定義と配置（矩形電極、将来的にマスクファイル対応）
- 電荷密度分布の管理
- 電極重複チェック

**クラス構成:**
- `StructureManager`: 構造定義と管理
  - `load_from_yaml()`: YAML読み込み
  - `generate_epsilon_array()`: 誘電率分布生成
  - `generate_electrode_mask()`: 電極マスク（3Dブーリアン配列）の生成
  - `get_electrode_voltages()`: 各グリッド点での電極電圧を取得
  - `check_electrode_overlap()`: 電極の重複チェック（重複時はエラー）
  - `set_charge_density()`: 電荷密度設定
  - `load_electrode_from_file()`: マスクファイルから電極パターン読み込み（将来実装）

**保持するデータ:**
```python
self.epsilon_array: np.ndarray  # 誘電率分布 (nx, ny, nz)
self.electrode_mask: np.ndarray  # 電極マスク (nx, ny, nz)
self.electrode_voltages: np.ndarray  # 電極電圧 (nx, ny, nz)
self.electrodes: List[Dict]  # 電極定義のリスト
self.layers: List[Dict]  # 層構造定義のリスト
```

### 3. `visualizer.py` - 可視化モジュール
**主な機能:**
- ポテンシャル分布の2Dカラーマップ表示
- 異なる深さ（z方向）のスライス表示
- 電極位置の可視化（電極領域をオーバーレイ表示）
- 電界分布の可視化（オプション）
- 結果のファイル保存（npz形式）

**関数構成:**
- `plot_potential_slice()`: 2Dスライスのプロット（電極位置も表示）
- `plot_electrode_pattern()`: 電極パターンの可視化
- `plot_potential_3d()`: 3D表示（オプション）
- `save_results()`: 結果保存
- `plot_convergence()`: 収束履歴のプロット

### 4. `config_example.yaml` - 構造定義の例
**含まれる設定（フィンガーゲート構造の例）:**
```yaml
# 計算領域
domain:
  size: [100e-9, 100e-9, 100e-9]  # [x, y, z] in meters
  grid_spacing: [10e-9, 10e-9, 5e-9]  # z方向を細かく

# 材料層構造
layers:
  - material: "Si"
    z_range: [0, 50e-9]
    epsilon_r: 11.7

  - material: "SiO2"
    z_range: [50e-9, 100e-9]
    epsilon_r: 3.9

# 電極構造（複数のフィンガーゲート）
electrodes:
  - name: "finger_gate_1"
    shape: "rectangle"
    x_range: [10e-9, 30e-9]
    y_range: [0, 100e-9]
    z_position: 100e-9  # 表面
    voltage: -0.5  # V

  - name: "finger_gate_2"
    shape: "rectangle"
    x_range: [40e-9, 60e-9]
    y_range: [0, 100e-9]
    z_position: 100e-9
    voltage: -1.0  # V

  - name: "finger_gate_3"
    shape: "rectangle"
    x_range: [70e-9, 90e-9]
    y_range: [0, 100e-9]
    z_position: 100e-9
    voltage: -0.5  # V

# 将来的なマスクファイル対応の例（コメントアウト）
# electrodes:
#   - name: "complex_pattern"
#     shape: "from_file"
#     mask_file: "electrode_mask.npy"
#     z_position: 100e-9
#     voltage: -0.5

# SORパラメータ
solver:
  omega: 1.8  # 緩和パラメータ
  max_iterations: 10000
  tolerance: 1e-6

# 境界条件
boundary_conditions:
  z_top:
    type: "mixed"  # 電極部分はDirichlet、その他はNeumann
    default_neumann_value: 0.0  # ∂ϕ/∂z = 0

  z_bottom:
    type: "neumann"
    value: 0.0

  x_sides:
    type: "neumann"  # or "periodic"
    value: 0.0

  y_sides:
    type: "neumann"  # or "periodic"
    value: 0.0
```

### 5. `test_solver.py` - テストケース
**テスト内容:**
- 一様誘電率での平行平板コンデンサ（解析解と比較）
- 単一電極での動作確認
- 複数電極（フィンガーゲート）での動作確認
- 電極重複検出のテスト
- 混合境界条件の正しい実装の検証
- 収束性のテスト
- 非均一誘電率での動作確認

### 6. `main.py` - 実行スクリプト例
**実装内容:**
- 典型的な使用例のデモ
- YAML読み込みから結果可視化までの一連の流れ

## 実装の詳細

### SOR法のアルゴリズム
誘電率が空間的に変化する場合の有限差分式：
```
ϕ[i,j,k]^(n+1) = (1-ω)ϕ[i,j,k]^(n) + (ω/A) * (B - ρ[i,j,k]/ε₀)
```
ここで、
- A, B: 隣接点の誘電率を考慮した係数
- ω: 緩和パラメータ（1 < ω < 2）
- ε₀: 真空の誘電率

### 格子サイズの柔軟性
- デフォルト: 10nm等間隔
- z方向を細かく設定可能（例: 5nm）
- YAMLで各方向独立に設定

### 表面電極構造の実装詳細

#### 電極の形状定義
**現在の実装: 矩形電極**
```yaml
electrodes:
  - name: "finger_gate_1"
    shape: "rectangle"
    x_range: [10e-9, 30e-9]
    y_range: [0, 100e-9]
    z_position: 100e-9
    voltage: -0.5  # V
```

**将来的な拡張: マスクファイル対応**
```yaml
electrodes:
  - name: "complex_pattern"
    shape: "from_file"
    mask_file: "electrode_mask.npy"  # 2D boolean array
    z_position: 100e-9
    voltage: -0.5
```

#### 表面境界条件の実装
表面（z=top）での境界条件：
- **電極がある領域**: ディリクレ境界条件
  - 電圧を指定値に固定: `ϕ = V_applied`
- **電極がない領域**: ノイマン境界条件
  - ∂ϕ/∂z = 0（真空との境界）

#### 電極の層構造
電極は境界条件としてのみ扱う（厚みゼロ）：
```
[表面 z=100nm] = 電極位置（部分的にDirichlet境界）
    ↑
SiO2層（誘電体、計算領域内）
    ↑
Si基板
```

#### 複数電極の管理
- **重複チェック**: 電極が重なる場合はエラーを出す
- **電極マスク**: 3Dブーリアン配列で電極位置を管理
- **電圧配列**: 各電極位置での電圧を保持

### 拡張性の確保
将来的な機能追加に備えた設計：
- **電極形状の拡張**: 円形、多角形、マスクファイル対応
- **ドーピング分布の追加**
- **自己無撞着計算**（Poisson-Schrödinger連成）
- **マルチグリッド法への拡張**
- **他の材料系**（GaAs/AlGaAs等）への対応

## 実装順序
1. **基本的なSORソルバの実装** (`poisson_solver.py`)
   - 一様境界条件での動作確認

2. **構造管理クラスの基本実装** (`structure_manager.py`)
   - 層構造の定義
   - 誘電率分布の生成

3. **電極構造の実装** (`structure_manager.py`に追加)
   - 矩形電極の配置機能
   - 電極マスクの生成
   - 重複チェック

4. **混合境界条件の実装** (`poisson_solver.py`に追加)
   - 表面での電極/非電極領域の区別
   - ディリクレ/ノイマン境界の適切な適用

5. **YAML設定ファイルの作成** (`config_example.yaml`)
   - フィンガーゲート構造の例

6. **可視化機能の実装** (`visualizer.py`)
   - 電極位置の可視化も追加

7. **テストケースの作成と検証** (`test_solver.py`)
   - 単一電極でのテスト
   - 複数電極でのテスト
   - 電極重複検出のテスト

8. **実行例の作成** (`main.py`)

## 依存ライブラリ
- numpy: 数値計算
- matplotlib: 可視化
- pyyaml: YAML読み込み
- scipy: （オプション）高度な数値計算

## 物理的パラメータ

### 材料定数
- **Si (シリコン)**
  - 比誘電率: εᵣ = 11.7
  - 真空の誘電率: ε₀ = 8.854 × 10⁻¹² F/m

- **SiO2 (酸化シリコン)**
  - 比誘電率: εᵣ = 3.9

### 計算領域
- デフォルトサイズ: 100nm × 100nm × 100nm
- デフォルト格子間隔: 10nm (調整可能)
- z方向の推奨格子間隔: 5nm (より細かい分解能)

### 境界条件の物理的意味
- **トップゲート (z=100nm)**: ディリクレ境界条件
  - 電極電位を固定（例: -1.0V）

- **基板底面 (z=0)**: ノイマン境界条件
  - 電場のz成分を指定（通常は0）

- **側面 (x, y方向)**: ノイマンまたは周期境界条件
  - ノイマン: 構造が十分に広い場合
  - 周期: 周期的な構造を想定する場合
