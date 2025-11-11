# SOR法による3D半導体ポアソンソルバの実装計画

## 概要
半導体ヘテロ構造と電極構造から電位分布とバンド構造を計算するSOR法ベースのポアソンソルバを実装します。

**主な機能:**
1. 3D静電ポテンシャル計算（SOR法）
2. **ヘテロ構造バンド曲がり計算** ← NEW (Phase 1実装中)
3. 複雑な電極構造のサポート
4. 柔軟な境界条件設定

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

## 座標系とインデックス規則

### 座標系
- **z = 0nm**: 表面（電極が配置される位置）
- **z軸の向き**: 負の方向に層が伸びる
  - 例: SiO2層 (0nm → -10nm)、Si基板 (-10nm → -50nm)
- **電極**: z = 0nm の表面に配置され、負の方向に厚みを持つ

### 配列のインデックス順序
- **配列shape**: `(nz, nx, ny)` - z軸が最初の次元
- **インデックス表記**: `array[k, i, j]`
  - `k`: z方向インデックス（k=0 が表面 z=0nm、k増加で z減少）
  - `i`: x方向インデックス
  - `j`: y方向インデックス
- **ループ順序**: z軸が最も外側
  ```python
  for k in range(nz):      # z方向（最も外側）
      for i in range(nx):  # x方向
          for j in range(ny):  # y方向
  ```

### 境界条件の対応
- **z_top** (k=0): 表面 (z=0nm)、電極がある場合はDirichlet BC
- **z_bottom** (k=nz-1): 底面、通常はNeumann BC

## ディレクトリ構成

```
SOR/
├── docs/                            # ドキュメント
│   ├── implementation_plan.md       # 実装計画書（このファイル）
│   └── optimization_plan.md         # 最適化計画
│
├── src/                             # ソースコード
│   ├── poisson_solver.py            # メインのSORソルバクラス
│   ├── structure_manager.py         # 半導体構造管理クラス
│   ├── materials.py                 # 材料データベース ← NEW
│   ├── solver_result.py             # 計算結果コンテナ ← NEW
│   └── visualizer.py                # 可視化モジュール
│
├── configs/                         # 設定ファイル
│   └── example.yaml                 # 設定ファイルの例（Si/SiO2フィンガーゲート）
│
├── tests/                           # テストコード
│   ├── test_solver.py               # ソルバテスト
│   ├── test_structure_manager.py    # 構造マネージャテスト
│   └── test_materials.py            # 材料データベーステスト ← NEW
│
├── examples/                        # 実行例
│   └── example.py                   # 実行スクリプト例
│
└── results/                         # 結果保存用ディレクトリ（実行時に自動作成）
    ├── potential_distribution.npz   # ポテンシャル分布データ
    ├── band_bending_result.npz      # バンド構造計算結果 ← NEW
    └── figures/                     # 可視化結果の画像
        ├── potential_slices.png     # ポテンシャル分布のスライス画像
        ├── electrode_pattern.png    # 電極パターンの可視化
        ├── convergence_history.png  # 収束履歴のグラフ
        └── band_diagram_1d.png      # 1Dバンドダイアグラム ← NEW
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

**重要な変更点（新しい座標系）:**
- **z軸**: z = 0nm (表面) → z = -size_z (底面)
- **z_range**: `[z_max, z_min]` の形式で **左側が大きい**
- **z_position**: 電極底面の位置（負の値、z=0から負の方向）

```yaml
# 計算領域
domain:
  size: [100e-9, 100e-9, 50e-9]  # [x, y, z] in meters
  grid_spacing: 5e-9  # 等方格子（単位: m）

# 材料層構造（z = 0 (表面) → z = -50nm (底面)）
# z_range は [z_max, z_min] の形式（左が大きい）
layers:
  - material: "SiO2"
    z_range: [0, -10e-9]  # 0nm → -10nm (表面側)
    epsilon_r: 3.9

  - material: "Si"
    z_range: [-10e-9, -50e-9]  # -10nm → -50nm (底面側)
    epsilon_r: 11.7

# 電極構造（複数のフィンガーゲート）
# z_position は電極底面の位置（負の値）
electrodes:
  - name: "finger_gate_1"
    shape: "rectangle"
    x_range: [10e-9, 30e-9]
    y_range: [0, 100e-9]
    z_position: -5e-9  # 電極底面（z=0 → z=-5nm が電極）
    voltage: -0.5  # V

  - name: "finger_gate_2"
    shape: "rectangle"
    x_range: [40e-9, 60e-9]
    y_range: [0, 100e-9]
    z_position: -5e-9
    voltage: -1.0  # V

  - name: "finger_gate_3"
    shape: "rectangle"
    x_range: [70e-9, 90e-9]
    y_range: [0, 100e-9]
    z_position: -5e-9
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
- **トップゲート (z=0nm, 表面)**: ディリクレ境界条件
  - 電極電位を固定（例: -1.0V）

- **基板底面 (z=-size_z)**: ノイマン境界条件
  - 電場のz成分を指定（通常は0）

- **側面 (x, y方向)**: ノイマンまたは周期境界条件
  - ノイマン: 構造が十分に広い場合
  - 周期: 周期的な構造を想定する場合

---

## Phase 1: ヘテロ構造バンド曲がり計算 (実装中)

### 実装状況

#### ✓ 完了した実装

##### 1. 材料データベースシステム (`materials.py`)
**実装内容:**
- `Material` dataclass
  - `name`, `epsilon_r`, `electron_affinity`, `band_gap`
  - `effective_mass_e`, `effective_mass_h` (将来使用)
- `MATERIAL_DATABASE`: Si, SiO2の物性値（4K）
- `get_material(name, overrides)`: ハイブリッド取得
  - データベースから自動取得
  - YAMLで個別上書き可能

**材料パラメータ (4K):**
| Material | εr   | χ (eV) | Eg (eV) | me* | mh* |
|----------|------|--------|---------|-----|-----|
| Si       | 11.7 | 4.05   | 1.12    | 0.26| 0.36|
| SiO2     | 3.9  | 0.9    | 9.0     | -   | -   |

**テスト:** `test_materials.py` - 8テスト、全てパス ✓

##### 2. SolverResultコンテナ (`solver_result.py`)
**メモリ効率設計:**
- 保持: `phi`, `x`, `y`, `z`, `materials` (list), `info`
- `Ec`, `Ev`は計算時のみ生成（保存しない）
- `materials`: z層ごとのリスト (3D配列ではない)

**バンドエッジ計算式:**
```
Ec(r) = -φ(r) - χ(z)  [eV]
Ev(r) = Ec(r) - Eg(z) [eV]
```
φ [V] は数値的にq·φ [eV]と等価

**主要メソッド:**
- `compute_Ec()`, `compute_Ev()`: バンドエッジ計算
- `get_band_diagram_1d(x_idx, y_idx)`: z方向プロファイル抽出
- `save()`, `load()`: 永続化

**特徴:**
- `materials=None`をサポート（後方互換性）
- 初期化時に配列形状を検証
- プロパティ: `nz`, `nx`, `ny`

##### 3. StructureManager拡張
**追加機能:**
- `materials_list`: 各z層のMaterialオブジェクト
- `generate_materials_list()`: YAMLから材料リスト生成
  - 材料名でデータベースから取得
  - YAML上書きを適用
  - 各z格子点に1つのMaterialオブジェクト

**`params`プロパティ更新:**
```python
{
    "epsilon": epsilon_array,
    "grid_spacing": h,
    "boundary_conditions": boundary_conditions,
    "electrode_mask": electrode_mask,
    "electrode_voltages": electrode_voltages,
    "materials_list": materials_list,  # NEW
}
```

**テスト:** 既存テスト全てパス ✓

##### 4. PoissonSolver拡張
**変更点:**
- `params`で`materials_list`を受け取り
- `solve()`が`SolverResult`を返す（`(phi, info)`ではない）
- `_create_solver_result()`: 結果オブジェクト生成ヘルパー

**後方互換性:** なし（要求に応じて削除）

**テスト:** `test_solver.py` - 4テスト、全てパス ✓
- テストは`materials=None`で動作（バンド計算は不要）

#### ⏳ 残りのタスク (Phase 1)

##### 5. 1Dバンドダイアグラム可視化 (`visualizer.py`)
**関数:** `plot_band_diagram_1d(result, x_idx, y_idx, save_path)`

**機能:**
- Ec(z), Ev(z)を実線でプロット
- -φ(z)を破線でプロット（フェルミ準位のシフト）
- 材料境界を縦線でマーク
- 軸: エネルギー (eV) vs z位置 (nm)
- 凡例、グリッド、ラベル

##### 6. バンド曲がりテスト (`test_solver.py`)
**テスト:** `test_band_bending_si_sio2()`

**検証項目:**
1. Si/SiO2構造をStructureManagerで作成
2. ポアソン方程式を解く
3. 以下を確認:
   - `Ec`, `Ev`配列の形状
   - 全点で`Ec - Ev = Eg`
   - Si領域とSiO2領域で異なるバンドパラメータ
   - Si/SiO2界面でのバンドオフセット（χの不連続）
   - `SolverResult`オブジェクトの正常生成

##### 7. exampleスクリプト更新
**ファイル:** `examples/example.py`

**更新内容:**
- `solve()`から`SolverResult`を受け取る
- `plot_band_diagram_1d()`を呼び出し
- `result.save()`で結果保存（Ec, Ev計算含む）

**出力:**
- `results/figures/band_diagram_1d.png`
- `results/band_bending_result.npz`

##### 8. ドキュメント更新
**ファイル:** `README.md`

**新規セクション:** バンド構造計算

**内容:**
- 機能説明
- 材料パラメータ設定（データベース vs YAML上書き）
- 使用例
- 1Dバンドダイアグラムの解釈
- 利用可能な材料リスト

---

## Phase 2以降の計画

### Phase 2: 拡張可視化と材料データベース (計画中)

#### 目標
1. 材料データベース拡張
2. 2D/3Dバンド構造可視化
3. 温度依存性パラメータ

#### タスク

**1. 材料データベース拡張**
- SiGe（各種Ge組成）
- GaAs, InP, AlGaAs, InGaAs
- 各材料に4Kと300Kのパラメータ
- 合金材料の組成依存性

**2. 2Dバンドダイアグラム**
- 関数: `plot_band_diagram_2d(result, z_index, save_path)`
- 固定zでのEcまたはEvの2Dヒートマップ
- 電極パターンのオーバーレイ
- 横方向のバンド曲がり表示

**3. 3Dバンド可視化**
- 関数: `plot_band_diagram_3d(result, component, save_path)`
- 3D等値面またはボリュームレンダリング
- component: 'Ec', 'Ev', 'Eg'
- インタラクティブプロット (plotly)

**4. 温度依存性**
- ソルバ初期化時に温度パラメータ追加
- 材料データベースにEg(T), χ(T)
- Vardeman近似でEg(T)を計算

### Phase 3: 自己無撞着計算 (将来)

#### 目標
1. ドーピングによる空間電荷考慮
2. ポアソン-キャリア密度連成
3. 自己無撞着反復

#### タスク

**1. ドーピングプロファイル**
StructureManagerに追加:
```yaml
layers:
  - material: "Si"
    z_range: [-20e-9, -100e-9]
    doping_type: "n"  # or "p"
    doping_density: 1e18  # cm^-3
```

電荷密度:
```
ρ(r) = q(p - n + Nd - Na)
```

**2. キャリア密度計算**

*Boltzmann近似（古典、高温）:*
```python
n(r) = Nc * exp((Ef - Ec(r)) / kT)
p(r) = Nv * exp((Ev(r) - Ef) / kT)
```

*Fermi-Dirac統計（量子、低温）:*
- Fermi-Dirac積分を実装
- 極低温でより正確

**3. 自己無撞着ループ**
```
1. φ(r) = 0で初期化
2. φ(r)からEc(r), Ev(r)を計算
3. Ec(r), Ev(r), EfからN(r), p(r)を計算
4. n(r), p(r), ドーピングからρ(r)を計算
5. ポアソン方程式を解く: -∇⋅(ε∇φ) = ρ/ε0
6. |φ_new - φ_old| < tolerance なら収束
7. そうでなければ φ = φ_new, ステップ2へ
```

実装:
- 新メソッド: `PoissonSolver.solve_self_consistent()`
- φとn/p両方の収束を追跡
- 安定性のためのダンピング/ミキシング

**4. フェルミ準位計算**

*大域平衡:*
- 電荷中性からEfを計算
- 非平衡では空間的に変化

*準フェルミ準位（発展）:*
- 電子と正孔で別々のEf
- バイアスまたは照明下

### Phase 4: 量子効果 (将来)

#### 目標
1. 量子井戸/量子細線計算
2. サブバンド構造
3. トンネル電流

#### タスク

**1. Schrödinger-Poisson連成**

*1D Schrödinger方程式（z方向）:*
```
[-ħ²/2m* d²/dz² + Ec(z)]ψ(z) = E·ψ(z)
```

*自己無撞着ループ:*
```
1. PoissonでEc(z)を解く
2. Schrödingerでψ(z), Eを解く
3. |ψ(z)|²からキャリア密度を計算
4. 更新されたρ(z)でPoissonを解く
5. 収束まで繰り返し
```

**2. サブバンド構造**

*量子井戸:*
- エネルギー準位En
- 波動関数ψn(z)
- 2D状態密度

*可視化:*
- Ec(z)にψn(z)を重ねてプロット
- エネルギー準位図

**3. トンネル計算**

*WKB近似:*
- 障壁を通過するトンネル確率
- 量子ドット、共鳴トンネルダイオードに関連

---

## データフロー概要

```
YAML設定
    ↓
StructureManager
    ├→ epsilon_array (nz, nx, ny)
    ├→ materials_list [Material × nz]
    ├→ electrode_mask, electrode_voltages
    └→ boundary_conditions
    ↓
PoissonSolver.params
    ↓
PoissonSolver.solve()
    ├→ φ(r) via SOR iteration
    └→ SolverResult
           ├─ phi, x, y, z
           ├─ materials
           ├─ info
           ├─ compute_Ec() → Ec(r)
           ├─ compute_Ev() → Ev(r)
           └─ get_band_diagram_1d() → z, Ec, Ev, phi
    ↓
Visualizer
    ├→ plot_band_diagram_1d()
    ├→ plot_band_diagram_2d() [Phase 2]
    └→ plot_band_diagram_3d() [Phase 2]
```

---

## 設計原則

### メモリ効率
- **計算可能なものは保存しない**: Ec, Evはオンデマンドで計算
- **z層材料リスト**: nz個のMaterialリスト、(nz,nx,ny)配列ではない
- **遅延評価**: バンドエッジは必要時のみ

### モジュール性
- **材料データベース**: 集中管理、拡張容易
- **ハイブリッドパラメータシステム**: データベースデフォルト + YAML上書き
- **SolverResult**: 結果アクセスのクリーンなインターフェース

### テスト戦略
- **単体テスト**: 各モジュールを独立してテスト
- **統合テスト**: 全ワークフロー（YAML → solve → band diagram）
- **検証テスト**: 可能な場合は解析解と比較

---

## パフォーマンス考慮事項

### 現在のボトルネック
1. **SOR反復**: JITコンパイル済み、既に最適化
2. **バンドエッジ計算**: O(nz·nx·ny)、現在の格子サイズでは許容範囲

### 将来の最適化（必要に応じて）
1. **ベクトル化**: バンドエッジ計算で既に実装
2. **キャッシング**: 複数回アクセス時はEc, Evをキャッシュ
3. **疎格子**: 適応的メッシュ細分化
4. **GPU加速**: 非常に大きな3Dグリッド用

---

## 参考

### バンド構造形式
- 真空準位を基準 (E = 0)
- Ec(r) = -q·φ(r) - χ(r)
- Ev(r) = Ec(r) - Eg(r)
- 界面でのバンドオフセット: ΔEc = χ₁ - χ₂

### 材料パラメータ (4K)
**Si:**
- χ = 4.05 eV (電子親和力)
- Eg = 1.12 eV (間接遷移)
- εr = 11.7

**SiO2:**
- χ = 0.9 eV
- Eg = 9.0 eV (絶縁体)
- εr = 3.9

**バンドオフセット (Si/SiO2):**
- ΔEc ≈ 3.15 eV (伝導帯)
- ΔEv ≈ 4.73 eV (価電子帯)
