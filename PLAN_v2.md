# Implementation Plan v2.1 — 最终修订版

> **Live progress tracker** (updated as batches land)
>
> | Batch | Status | Commit | Notes |
> |---|---|---|---|
> | Step 0 — merge master, extend GMM K range, rerun precompute | ✅ Done | `ab4a1f0` | GMM full best K=44 (interior), GMM diag best K=49 (still near upper bound, accepted) |
> | Batch 1 — recommend_from_playlist + new app.py + drop page_playlist | ✅ Done | _this commit_ | `recommend_from_playlist` arrived from master merge (no work). App.py registers 4 pages, journey + evaluation are stubs ("Coming soon") |
> | Batch 2 — derive_scenario_mappings.py | ⏳ Next | | |
> | Batch 3 — src/journey.py + src/visualization.py | ⏳ Pending | | |
> | Batch 4 — app/page_journey.py | ⏳ Pending | | Replaces stub; absorbs old playlist functionality |
> | Batch 5 — scripts/evaluate_recommendations.py + app/page_evaluation.py | ⏳ Pending | | Replaces stub |
> | Batch 6 — page_clusters.py overlap heatmap + pyproject + README + .gitignore | ⏳ Pending | | |
>
> **Decisions locked in for execution:**
> - Trajectory feature scale: raw 4D, energy/valence/danceability as-is (already [0,1]), tempo clipped to [50,200] BPM then `(tempo-50)/150` to map onto [0,1].
> - GMM tuning K range: K-Means stays [5,30], GMM full + diag at [5,50]. Best K post-rerun: full=44, diag=49.
> - `recommend_from_playlist` is provided by master's `src/recommend.py` — no new method needed in `recommend.py`.
> - Reranking from master is preserved in `page_recommend.py`.

## 0. 项目目标 & Beyond Baseline 叙事

Baseline: PCA + clustering (K-Means / GMM) + cosine-similarity recommendation

Beyond baseline 的四个支柱：

1. **方法论比较的深度**: 自实现 K-Means + sklearn GMM (full + diag 两种 covariance) 三个 clustering 算法的系统对比
2. **Recommendation evaluation 的严谨度**: 6 个方法 × 2 个 metric 的对比（详见 §4）
3. **Genre coverage as a new metric**: 直接呼应项目 motivation
4. **Sequential recommendation extension**: Music Journey 页面（trajectory playlist + radio mode + Russell 情绪空间可视化），呼应教材 Ch 11

---

## 1. 当前 Repo 状态 Audit (基于最新代码)

| 模块 | 状态 | 备注 |
|---|---|---|
| `app/page_recommend.py` | ✅ 完成 | 三种推荐 mode + Spotify API + load artifacts |
| `app/page_clusters.py` | ✅ 完成 | 已重构为 load artifacts 版本，无 UMAP |
| `app/page_playlist.py` | ❌ 未在 fork | 上游 main 有，fork 没拉，且代码有 bug |
| `app/app.py` | ⚠️ 不完整 | 没注册 playlist page |
| `src/clustering.py` | ✅ 完成 | tune_kmeans 用 sklearn, fit_kmeans 用 CustomKMeans |
| `src/custom_kmeans.py` | ✅ 完成 | scratch 实现 |
| `src/evaluate.py` | ✅ 完成 | 已有 genre_hit_rate |
| `src/recommend.py` | ⚠️ 缺方法 | 缺 `recommend_from_playlist` |
| `src/explain.py` | ✅ 完成 | radar charts |
| `src/features.py` | ✅ 完成 | feature engineering |
| `scripts/precompute.py` | ✅ 完成 | 三个算法 artifacts 已生成 |
| `pyproject.toml` | ⚠️ 待清理 | umap-learn 依赖需删 |
| `README.md` | ⚠️ 过时 | 状态表过时, 没提 precompute 流程 |

---

## 2. 待完成工作（按依赖顺序，标 P0–P3 优先级）

### P0 — Bug 修复 & playlist 整合

**Task 2.1**: `recommend.py` 加 `recommend_from_playlist` 方法

逻辑：
- 输入 `seed_indices: list[int], top_k: int`
- 取 `self.feature_matrix[seed_indices]` 算 mean → `aggregate_query`（12 维）
- 在全 dataset 上算 `aggregate_query` 的 cosine similarity（用 `self.nn.kneighbors`）
- 排除 seed 本身和同名歌
- 返回 top-K dataframe（同 `recommend()` 的格式）

**Task 2.2**: 砍掉独立的 `page_playlist.py`，整合进 Music Journey

决策（Q5 选 c）：playlist 输入作为 Music Journey 页面的"用 playlist 定义起点"模式。
具体：在 Music Journey 页面的"自定义模式"里加一个子选项 — 用户可以通过 (a) slider 直接调 feature 值 / (b) 从 playlist 中聚合 feature 值 来定义起点和终点。
`recommend_from_playlist` 还是要加到 `recommend.py`，因为 Music Journey 内部要复用这个 aggregate 逻辑。

**Task 2.3**: 更新 `app/app.py`

只注册三个 page：

- `page_recommend.py` — Song Search & Recommend
- `page_clusters.py` — Cluster Explorer
- `page_journey.py` (新) — Music Journey
- `page_evaluation.py` (新) — Recommendation Evaluation

砍掉 `page_playlist.py`。

### P1 — Music Journey 新页面（核心 beyond baseline）

**Task 2.4**: 新建 `src/journey.py`

需要实现的函数：

```python
# 4 个 trajectory features
TRAJECTORY_FEATURES = ["energy", "valence", "danceability", "tempo_normalized"]

def normalize_trajectory_input(raw: dict) -> np.ndarray:
    """从 {energy: 0.7, valence: 0.5, ...} 字典生成 4 维向量。
    tempo 输入是 BPM，归一化到 tempo / 200 -> [0, 1] 范围。"""

def generate_trajectory(start: np.ndarray, end: np.ndarray, n_steps: int, mode: str = "linear") -> list[np.ndarray]:
    """生成 n_steps 个 target points。mode='linear' 或 'sigmoid'(extension)。"""

def select_song_at_target(
    target: np.ndarray,
    trajectory_feature_matrix: np.ndarray,  # 89578 x 4 (subset of full feature matrix)
    df: pd.DataFrame,
    already_selected: set[int],
    popularity_weight: float = 0.1,
    diversity_artist_penalty: float = 1.0,
) -> int:
    """对单个 target point 选一首歌。
    score = -L2_distance(song_features, target) + popularity_weight * pop_normalized
            - diversity_artist_penalty * (1 if artist already in selected else 0)
    """

def build_journey_playlist(
    start: np.ndarray, end: np.ndarray,
    trajectory_feature_matrix: np.ndarray, df: pd.DataFrame,
    n_songs: int = 10, popularity_weight: float = 0.1,
) -> pd.DataFrame:
    """主入口。返回 dataframe 包含 song_index, track_name, artists, target_position(in trajectory), actual_features。"""

def radio_next(
    center: np.ndarray,
    trajectory_feature_matrix: np.ndarray, df: pd.DataFrame,
    already_played: set[int],
    drift: float = 0.05,
) -> int:
    """Radio 模式：在 center 附近小范围漂移找下一首。
    具体：candidate 区间 [center - drift, center + drift]，从中找最近的、未播放的、未同 artist 的歌。"""
```

**Task 2.5**: 新建 `src/visualization.py`

```python
def plot_journey(
    trajectory_targets: list[np.ndarray],   # n_steps x 2 (energy, valence)
    selected_songs: pd.DataFrame,            # 必须有 energy, valence, danceability, tempo, track_name, track_genre
    background_df: pd.DataFrame,             # 抽样 3000 首作为背景
    start: np.ndarray, end: np.ndarray,
) -> go.Figure:
    """Russell 情绪空间 2D 可视化。
    Layer 1: 背景散点 (灰色, opacity=0.2, color by genre 但低 saturation)
    Layer 2: Trajectory 虚线 + target points (空心圆)
    Layer 3: 推荐歌 (实心圆, size = danceability, color saturation = tempo)
    Layer 4: 起点 (五角星 ★) + 终点 (方块 ■)
    四象限标注: 右上=Excited/Happy, 左上=Angry/Tense, 左下=Sad/Depressed, 右下=Calm/Peaceful
    """

def plot_radio_history(
    history_songs: pd.DataFrame,
    background_df: pd.DataFrame,
    drift_radius: float,
) -> go.Figure:
    """Radio 模式可视化 - 显示已播放歌曲在 Russell 空间里的轨迹（实线连接），加 drift 范围圈。"""
```

**Task 2.6**: 新建 `app/page_journey.py`

页面结构：

```
# Music Journey 页面

## Tab 1: 场景预设 (Scenario Presets)
- 6 个场景卡片 (运动 / 学习 / 睡前 / 派对 / 通勤 / 雨夜)
- 用户点选 → 自动填入起点终点 + 生成 trajectory playlist
- 显示 plot_journey 可视化 + 推荐歌曲列表 (含 Spotify play button)

## Tab 2: 自定义模式 (Custom Mode)
- 子模式选择: (a) slider 模式 / (b) playlist 聚合模式

### (a) Slider 模式
- 起点 4 个 slider: energy, valence, danceability, tempo
- 终点 4 个 slider
- Toggle: "起点=终点（生成电台）" 或 "起点≠终点（生成 trajectory playlist）"
- 如果是 trajectory: slider for n_songs (5-15)
- 如果是 radio: 显示已播放的歌 + "Next" 按钮

### (b) Playlist 聚合模式
- 复用 page_playlist.py 的 search & multi-select UI
- 用户选若干歌 → 算 aggregate feature → 作为起点
- 用户可以再选若干歌 → aggregate → 作为终点
- 或者只选一组 → 自动进入 radio 模式
```

**Task 2.7**: 6 个预设场景的具体定义

Q-followup: 场景 mapping 我之前推荐"数据驱动 EDA"但你没明确选。我在这里默认用数据驱动方法，并在 plan 里写明：

具体做法（写一个一次性 EDA 脚本 `scripts/derive_scenario_mappings.py`）：
- 对每个场景，找出 dataset 里语义最相关的 `track_genre`（如 workout → `'workout'` / `'electro'`；focus → `'classical'` / `'piano'`；wind down → `'sleep'` / `'ambient'`）
- 算这些歌的 audio feature 中位数 → 作为该场景的 "centroid"
- 起点 = centroid 偏低能量/快感的方向；终点 = centroid 偏高能量/快感的方向（对应"渐入场景"）
- 把这些 mapping 存成 `artifacts/scenario_mappings.joblib`

这样 mapping 是数据驱动的，不是凭空定义的，report 里好讲。

### P1 — Recommendation Evaluation（Phase 4 from PLAN）

**Task 2.8**: 新建 `scripts/evaluate_recommendations.py`

目的（写在 report 里的核心论述）：

> 我们的推荐算法基于 audio feature similarity 而非 collaborative filtering。一个根本问题是：**这种基于 feature 的推荐真的比 trivial baselines 好吗？** 如果只是随机推或推热门歌也能达到类似效果，那 PCA+clustering 这一整套就没价值。
> 我们设计 3 个 baseline 与 3 个我们的算法做对比，并定义 2 个 evaluation metric。Genre Coverage 是我们提出的新 metric，用来从相反方向支撑项目动机：feature-based 推荐应该比 genre-match baseline 有更高的 coverage（即跨 genre 找到风格相近的歌），同时 hit rate 不能太低。

**3 个 Baseline 的具体定义:**

| Baseline | 定义 |
|---|---|
| `random` | 从 dataset 随机抽 K 首（不重复，不含 query 本身） |
| `popularity` | 全 dataset 按 popularity 降序的 top-K（对所有 query 都一样 — 体现 cold start 时的常见 fallback） |
| `genre_match` | 从和 query 同 `track_genre` 的歌中随机抽 K 首 |

**3 个我们的算法:**

| Algorithm | 实现位置 |
|---|---|
| KNN | `RecommendationEngine.recommend()` (Phase 1) |
| K-Means cluster | `RecommendationEngine.recommend_by_cluster()` (Phase 2) |
| GMM posterior | `RecommendationEngine.recommend_by_gmm()` (Phase 2，用 GMM-full) |

**2 个 Evaluation Metrics:**

**Metric A — Genre Hit Rate** (已实现 `evaluate.py:genre_hit_rate`)
- 推荐 K 首中，和 query 共享至少一个 `all_genres` 标签的比例
- 范围 [0, 1]，越高越好（"推荐和 query 风格相符"）
- 期望: `genre_match` ≈ 1.0; `random` ≈ 0.05; `ours` > 0.5

**Metric B — Genre Coverage** (新做)
- 推荐 K 首中出现的 distinct `track_genre` 数量
- 范围 [1, K]，两面性：太低 = 困在单一 genre，太高 = 杂乱无章
- 关键论点: feature-based 推荐应该比 `genre_match` baseline 有更高 coverage —— 这正是 "genre labels are heterogeneous" 的实证支撑

**Script 输出:**

`artifacts/recommendation_eval.joblib` 是一个 dict:

```python
{
    "n_queries": 500,
    "top_k": 10,
    "results": {
        "random": {"hit_rate_mean": ..., "hit_rate_std": ..., "coverage_mean": ..., "coverage_std": ...},
        "popularity": {...},
        "genre_match": {...},
        "knn": {...},
        "kmeans_cluster": {...},
        "gmm_posterior": {...},
    },
    "raw_scores": {  # 给画 box plot 用
        "random": {"hit_rates": [500 floats], "coverages": [500 ints]},
        ...
    }
}
```

Sample size 500（已确认 Q3）, top_k=10。固定 random_state=42。

**Task 2.9**: 新建 `app/page_evaluation.py`

布局：

```
# Recommendation Evaluation 页面

## Section 1: Methodology
- 简短说明 6 种方法 + 2 个 metrics
- 一个 expander 详细解释每个 metric

## Section 2: Comparison Table
- 6 行 × 2 metric 对比表 (mean ± std)
- 高亮 best per metric

## Section 3: Box plot
- Genre Hit Rate 的 6 方法 box plot (绿色)
- Genre Coverage 的 6 方法 box plot (蓝色)
- 同时画一个散点 plot: x=hit_rate, y=coverage, 6 个 method 各一个点 (with error bar)
  - 这个 scatter 是核心 takeaway: "在 hit_rate vs coverage 的 trade-off 空间里，我们的方法在哪？"

## Section 4: Discussion (markdown)
- 解释 finding (留空 templates，自己填数字)
```

### P2 — Cluster comparison analysis（基本免费）

**Task 2.10**: 在 `page_clusters.py` Tab 4 加深度解读

precompute 已经把三个算法的 metrics 都存好了，需要：
1. 在已有的对比表下面加一段 markdown 模板（自己填数字）：
   - "K-Means achieved silhouette=X but NMI=Y (lowest). GMM-full has K_full=A, NMI=B; GMM-diag has K_diag=C, NMI=D. The fact that GMM-diag picked K=C while GMM-full picked K=A illustrates that..."
2. 加一个 cluster overlap heatmap: K-Means cluster vs GMM-full cluster 的 contingency matrix。如果两个算法找到了"同样的分组只是 label 不同"，对角线（适当 reorder 后）会很亮；如果找到了不同分组，整个矩阵会比较 diffuse。

具体实现：

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(km_labels, gmm_labels)  # NxN where N = max(K_km, K_gmm)
# Optional: reorder rows/cols to make diagonal brightest (Hungarian algorithm)
fig = px.imshow(cm, ...)
```

### P3 — Extensions（如果还有时间）

按优先级排（你选做几个就做几个）：

**Ext 1: LLM free-text mood**
- 用户输入自然语言（"a sad rainy night with coffee"）
- 调 OpenAI / Anthropic API → 翻译成 4 维 feature 目标值
- 注入 Music Journey 作为新模式
- 风险：API key, 网络依赖, demo 稳定性

**Ext 2: Trajectory sigmoid 模式**
- `generate_trajectory(mode='sigmoid', steepness=k)`
- 适合"运动"等需要"前慢中快后稳"曲线的场景
- 工作量小（在 journey.py 里加 10 行）

**Ext 3: Animation**
- Music Journey 可视化加 Play button
- marker 沿 trajectory 依次高亮，模拟"播放进度"
- 用 Plotly 的 animation_frame 做

**Ext 4: Diversity slider**
- Music Journey 页面加 λ slider 让用户调 popularity_weight 和 diversity_penalty
- 让用户实时看到推荐的变化

---

## 3. 文件改动清单（最终版）

| 文件 | 状态 | 改动 |
|---|---|---|
| `app/app.py` | 改 | 注册 4 个 page (砍 playlist, 加 journey + evaluation) |
| `app/page_recommend.py` | 不动 | 已经是好状态 |
| `app/page_clusters.py` | 改 | Tab 4 加 cluster overlap heatmap + 解读 |
| `app/page_playlist.py` | 删 | 功能合并进 journey |
| `app/page_journey.py` | 新建 | 场景预设 + 自定义 (slider / playlist) + radio |
| `app/page_evaluation.py` | 新建 | 6 方法 × 2 metric 对比 |
| `src/recommend.py` | 改 | 加 `recommend_from_playlist` |
| `src/journey.py` | 新建 | trajectory + radio 算法 |
| `src/visualization.py` | 新建 | Russell plot |
| `src/clustering.py` | 不动 | OK |
| `src/custom_kmeans.py` | 不动 | OK |
| `src/evaluate.py` | 不动 | `genre_hit_rate` 已实现 |
| `src/explain.py` | 不动 | OK |
| `src/features.py` | 不动 | OK |
| `scripts/precompute.py` | 不动 | OK |
| `scripts/derive_scenario_mappings.py` | 新建 | 一次性 EDA, 生成场景 mapping |
| `scripts/evaluate_recommendations.py` | 新建 | 跑 6 方法 × 500 query × 2 metric |
| `pyproject.toml` | 改 | 删 umap-learn |
| `README.md` | 改 | 更新 status, precompute 流程, Music Journey 描述 |
| `.gitignore` | 改 | 加 `artifacts/` |

---

## 4. 实施顺序（依赖图）

```
[P0] Task 2.1: recommend.py 加 recommend_from_playlist
        ↓
[P0] Task 2.2 + 2.3: 砍 page_playlist, 改 app.py 注册
        ↓
[P1] Task 2.7: derive_scenario_mappings.py + 跑一次 → scenario_mappings.joblib
        ↓
[P1] Task 2.4 + 2.5: src/journey.py + src/visualization.py
        ↓
[P1] Task 2.6: page_journey.py 整合
        ↓
[P1] Task 2.8: evaluate_recommendations.py
        ↓
[P1] 跑 evaluate_recommendations.py (~5-15 分钟) → recommendation_eval.joblib
        ↓
[P1] Task 2.9: page_evaluation.py
        ↓
[P2] Task 2.10: page_clusters.py Tab 4 加 overlap heatmap
        ↓
[Polish] 改 pyproject.toml + README + .gitignore
        ↓
[P3] Extensions (有时间就做)
```

---

## 5. Open Questions（这次只剩 2 个，比上次少多了）

| # | 问题 | 我的 default |
|---|---|---|
| Q-A | 场景预设 6 个 (workout/focus/wind down/party/commute/rainy night) 名单 OK 吗？要加/换吗？ | 这 6 个 |
| Q-B | 场景 mapping 用数据驱动 EDA (`derive_scenario_mappings.py`) 还是我直接给手动 default？ | 数据驱动 |

---

## 准备工作

回答这 2 个问题后，我会逐个文件输出代码。代码量比较大，所以我会分批给你：

**Batch 1**（修 bug + 砍 playlist + 改 app.py）：
- `recommend.py` 加 `recommend_from_playlist`
- 新 `app.py`
- 删除 `page_playlist.py`

**Batch 2**（场景 mapping）：
- `scripts/derive_scenario_mappings.py`

**Batch 3**（核心算法）：
- `src/journey.py`
- `src/visualization.py`

**Batch 4**（Music Journey 页面）：
- `app/page_journey.py`

**Batch 5**（Evaluation）：
- `scripts/evaluate_recommendations.py`
- `app/page_evaluation.py`

**Batch 6**（cluster overlap + polish）：
- `page_clusters.py` 改动
- `pyproject.toml` 改动
- `README.md` 重写
- `.gitignore` 改动

每个 batch 我会确认你跑通了再继续下一个。
