"""Music Journey: trajectory playlist generation and GMM-cluster roaming.

Trajectory playlist: greedy sequential nearest-neighbor with penalty scoring
in a 4D trajectory feature space (energy, valence, danceability, tempo_norm).

Endless mode: GMM-cluster roaming with Bayesian belief update over the GMM
posterior distribution — each new song shifts the belief vector, producing
a probabilistic drift through the feature space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRAJECTORY_FEATURES = ["energy", "valence", "danceability", "tempo_norm"]


# ── Shared helpers ───────────────────────────────────────────────────────────


def normalize_tempo(tempo: pd.Series) -> pd.Series:
    """Clip tempo to [50, 200] BPM then scale to [0, 1] via (tempo - 50) / 150."""
    return (tempo.clip(lower=50, upper=200) - 50) / 150


def build_trajectory_features(df_encoded: pd.DataFrame) -> np.ndarray:
    """Project the catalog onto the 4-D trajectory feature space.

    Returns an (N, 4) float array in column order matching TRAJECTORY_FEATURES.
    """
    return np.column_stack([
        df_encoded["energy"].astype(float).values,
        df_encoded["valence"].astype(float).values,
        df_encoded["danceability"].astype(float).values,
        normalize_tempo(df_encoded["tempo"].astype(float)).values,
    ])


# ── Trajectory generation ───────────────────────────────────────────────────


def generate_trajectory(
    start: np.ndarray,
    end: np.ndarray,
    n_steps: int,
) -> list[np.ndarray]:
    """Generate *n_steps* linearly-interpolated waypoints from *start* to *end*.

    Each waypoint is a 4-D vector in [0, 1]^4.
    """
    if n_steps < 2:
        return [start.copy()]
    return [start + (end - start) * t / (n_steps - 1) for t in range(n_steps)]


def select_song_at_target(
    target: np.ndarray,
    traj_matrix: np.ndarray,
    df: pd.DataFrame,
    excluded: set[int],
    excluded_artists: set[str],
    excluded_names: set[str],
    popularity_weight: float = 0.5,
    artist_penalty: float = 1.0,
    name_penalty: float = 1.0,
) -> int:
    """Select the best song for a single trajectory waypoint.

    Scoring function (maximized):
        score = -L2(features, target)
              + popularity_weight * popularity_normalized
              - artist_penalty * (1 if artist already used)
              - name_penalty   * (1 if track_name already used)

    Returns the DataFrame integer index of the chosen song.
    """
    # L2 distance to target
    dists = np.linalg.norm(traj_matrix - target, axis=1)

    # Popularity in [0, 1]
    pop = df["popularity"].values.astype(float) / 100.0

    # Artist + name repeat penalties (vectorized via pandas isin)
    artist_repeat = df["artists"].isin(excluded_artists).values.astype(float)
    name_repeat = df["track_name"].isin(excluded_names).values.astype(float)

    scores = (
        -dists
        + popularity_weight * pop
        - artist_penalty * artist_repeat
        - name_penalty * name_repeat
    )

    # Mask excluded indices
    if excluded:
        scores[np.fromiter(excluded, dtype=int, count=len(excluded))] = -np.inf

    return int(np.argmax(scores))


def build_journey_playlist(
    start: np.ndarray,
    end: np.ndarray,
    traj_matrix: np.ndarray,
    df: pd.DataFrame,
    n_songs: int = 10,
    popularity_weight: float = 0.5,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build a trajectory playlist using greedy sequential selection.

    For each waypoint along the start→end trajectory, greedily select the
    highest-scoring song (composite of feature proximity, popularity, and
    artist + track-name diversity).  Selected songs, artists, and track
    names are excluded from future steps so the playlist contains 10
    visibly distinct songs even when the catalog has the same track on
    multiple track_ids (re-releases / explicit/clean variants).

    *jitter*: if positive, perturb start and end by an independent uniform
    sample in [-jitter, +jitter] per dimension and clip to [0, 1].  This
    breaks determinism so each "Regenerate" returns a different playlist
    while staying near the scenario's intended audio character.  Pass a
    fresh `rng` per call to actually see different results.
    """
    if jitter > 0:
        if rng is None:
            rng = np.random.default_rng()
        start = np.clip(start + rng.uniform(-jitter, jitter, size=start.shape), 0.0, 1.0)
        end = np.clip(end + rng.uniform(-jitter, jitter, size=end.shape), 0.0, 1.0)

    waypoints = generate_trajectory(start, end, n_songs)
    excluded: set[int] = set()
    excluded_artists: set[str] = set()
    excluded_names: set[str] = set()
    rows: list[dict] = []

    for step, target in enumerate(waypoints):
        idx = select_song_at_target(
            target, traj_matrix, df, excluded, excluded_artists, excluded_names,
            popularity_weight=popularity_weight,
        )
        excluded.add(idx)
        excluded_artists.add(df.iloc[idx]["artists"])
        excluded_names.add(df.iloc[idx]["track_name"])

        song = df.iloc[idx]
        actual = traj_matrix[idx]
        rows.append({
            "step": step,
            "song_index": idx,
            "track_id": song.get("track_id"),
            "track_name": song["track_name"],
            "artists": song["artists"],
            "album_name": song.get("album_name", ""),
            "track_genre": song["track_genre"],
            "popularity": int(song["popularity"]),
            # Targets
            "target_energy": target[0],
            "target_valence": target[1],
            "target_danceability": target[2],
            "target_tempo_norm": target[3],
            # Actuals
            "actual_energy": actual[0],
            "actual_valence": actual[1],
            "actual_danceability": actual[2],
            "actual_tempo_norm": actual[3],
        })

    return pd.DataFrame(rows)


# ── GMM-cluster roaming (endless mode) ──────────────────────────────────────


_AUDIO_DIST_MAX = float(np.sqrt(4))  # max possible L2 in [0,1]^4


def gmm_endless_next(
    belief: np.ndarray,
    gmm_probs: np.ndarray,
    traj_matrix: np.ndarray,
    prev_traj_features: np.ndarray | None,
    df: pd.DataFrame,
    excluded: set[int],
    excluded_names: set[str] | None = None,
    eta: float = 0.3,
    temperature: float = 0.5,
    top_k: int = 15,
    rng: np.random.Generator | None = None,
) -> tuple[int, np.ndarray]:
    """Pick the next song using GMM-cluster roaming with audio continuity.

    Both terms are *distances* in [0, 1] so η linearly mixes their cost:

        cost(c) =      η  · TVD(belief, posterior(c))
                  + (1−η) · ‖prev_audio − audio(c)‖₂ / √4

    where TVD is total variation distance (½·L1 between probability
    vectors).  We minimize cost (== maximize −cost).

    Why TVD instead of cosine: cosine similarity between *probability
    vectors* saturates at 1.0 for hundreds of songs whose dominant GMM
    component matches the belief's, leaving the audio term as the only
    discriminator regardless of η.  TVD keeps all 44 cluster components
    in play so η actually changes the ranking.

    η semantics — single drift-rate knob:
      η → 0  : audio continuity dominates → tight neighborhood radio.
      η = ½  : equal weight.
      η → 1  : posterior similarity dominates → wider cluster-roam.

    On the seed step *prev_traj_features* is None, so the audio term is
    skipped (just rank by belief similarity).

    Belief update is unchanged: EMA `belief ← (1−η)·old + η·picked`,
    renormalized.

    Sampling:
    1. Mask excluded indices and track-names.
    2. Take top-*top_k* by score (= −cost).
    3. Rescale the top scores (subtract pool min) so temperature has
       visible effect even when the raw range is narrow.
    4. Softmax-sample with *temperature*.

    Returns (song_index, updated_belief).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Total variation distance between belief and each posterior, in [0, 1]
    belief_dist = 0.5 * np.abs(gmm_probs - belief).sum(axis=1)

    if prev_traj_features is not None:
        audio_dist = np.linalg.norm(traj_matrix - prev_traj_features, axis=1) / _AUDIO_DIST_MAX
        cost = eta * belief_dist + (1.0 - eta) * audio_dist
    else:
        cost = belief_dist

    score = -cost

    # Mask excluded indices and excluded track-names
    if excluded:
        score[np.fromiter(excluded, dtype=int, count=len(excluded))] = -np.inf
    if excluded_names:
        name_mask = df["track_name"].isin(excluded_names).values
        score[name_mask] = -np.inf

    # Top-K candidates
    top_indices = np.argsort(score)[::-1][:top_k]
    top_scores = score[top_indices]
    valid = top_scores > -np.inf
    if not valid.any():
        available = np.flatnonzero(np.isfinite(score))
        if len(available) == 0:
            raise ValueError("No songs available")
        chosen = int(rng.choice(available))
        return chosen, belief.copy()

    top_indices = top_indices[valid]
    top_scores = top_scores[valid]

    # Score rescaling: stretch contrast so temperature has visible effect
    rescaled = top_scores - top_scores.min()
    logits = rescaled / max(temperature, 1e-8)
    logits -= logits.max()  # numerical stability
    probs = np.exp(logits)
    probs /= probs.sum()

    chosen = int(top_indices[rng.choice(len(top_indices), p=probs)])

    # Belief update (EMA over GMM posterior space)
    new_belief = (1 - eta) * belief + eta * gmm_probs[chosen]
    total = new_belief.sum()
    if total > 0:
        new_belief /= total

    return chosen, new_belief
