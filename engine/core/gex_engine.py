"""
core/gex_engine.py

GEX計算エンジン。gamma.pyのロジックを継承・昇華。

変更点（gamma.pyからの主な修正）:
  - CSV手動読み込み → OptionChainオブジェクトを受け取る設計
  - 行ループ → NumPy完全vectorized演算
  - Call/Put式の不統一 → Call式に統一（理論上同値）
  - T=0 → 最小1分（1/1440）を保証
  - OI → effective_oi（当日Volume補正済み）で計算

依存関係:
  - models/option_chain.py  → OptionChain
  - models/gex_snapshot.py  → GEXSnapshot, WallLevel, GammaCondition, WallStrength
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.stats import norm

from models.option_chain import OptionChain
from models.gex_snapshot import (
    GammaCondition,
    GEXSnapshot,
    WallLevel,
    WallStrength,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 設定定数（config.yamlに外出し予定。現時点はここで管理）
# ------------------------------------------------------------------

# OI補正係数: effective_oi = OI + ALPHA * daily_volume
OI_CORRECTION_ALPHA: float = 0.5

# GEXプロファイル計算のストライクレンジ（スポット比）
GEX_PROFILE_RANGE_PCT: float = 0.10  # ±10%

# GEXプロファイルの分解能（レベル数）
GEX_PROFILE_LEVELS: int = 200

# ZeroGamma付近のNEUTRAL判定幅（スポット比）
ZERO_GAMMA_NEUTRAL_BAND_PCT: float = 0.005  # ±0.5%

# Dissonanceスコアの平滑化ウィンドウ
DISSONANCE_SMOOTH_WINDOW: int = 5

# 無リスク金利（年率）
RISK_FREE_RATE: float = 0.05

# 配当利回り（SPY: 約1.3%）
DIVIDEND_YIELD: float = 0.013

# WallStrength判定閾値
WALL_STRONG_THRESHOLD: float = 0.30   # 全体GEXの30%以上
WALL_MEDIUM_THRESHOLD: float = 0.15   # 全体GEXの15%以上


class GEXEngine:
    """
    OptionChainを受け取り、GEXSnapshotを生成する計算エンジン。

    使用例:
        engine = GEXEngine()
        chain  = await saxo_client.get_option_chain("SPY")
        snap   = engine.calc_gex_profile(chain)
        print(snap.summary())
    """

    def __init__(
        self,
        risk_free_rate: float = RISK_FREE_RATE,
        dividend_yield: float = DIVIDEND_YIELD,
        oi_alpha: float = OI_CORRECTION_ALPHA,
    ):
        self.r = risk_free_rate
        self.q = dividend_yield
        self.oi_alpha = oi_alpha

    # ==================================================================
    # PUBLIC: メインエントリーポイント
    # ==================================================================

    def calc_gex_profile(
        self,
        chain: OptionChain,
        profile_range_pct: float = GEX_PROFILE_RANGE_PCT,
        profile_levels: int = GEX_PROFILE_LEVELS,
    ) -> GEXSnapshot:
        """
        OptionChain → GEXSnapshot の完全計算。

        処理順:
          1. OI補正（effective_oi = OI + alpha * volume）
          2. GreeksフォールバックBS計算（取引所提供がない場合）
          3. Spot GEX（現在価格でのGEX分布）
          4. GEXプロファイル（価格レンジ全体でのGEX曲線）
          5. ZeroGamma / Call Wall / Put Wall の検出
          6. 0DTE比率・Dissonanceスコアの計算
          7. GEXSnapshot生成
        """
        if not chain.legs:
            raise ValueError(f"OptionChain for {chain.ticker} has no legs")

        logger.info(
            f"[GEXEngine] {chain.ticker} spot={chain.spot:.2f} "
            f"legs={len(chain.legs)} @ {chain.timestamp}"
        )

        # --- Step 1: DataFrame変換 + OI補正 ---
        df = chain.to_dataframe()
        df = self._apply_oi_correction(df)

        # --- Step 2: Greeksフォールバック ---
        df = self._fill_greeks_fallback(df, chain.spot)
        self._last_df = df  # ZeroGamma拡張探索用に保持

        # --- Step 3: Spot GEX計算 ---
        df = self._calc_spot_gex(df, chain.spot)

        # --- Step 4: GEXプロファイル計算 ---
        levels = np.linspace(
            chain.spot * (1 - profile_range_pct),
            chain.spot * (1 + profile_range_pct),
            profile_levels,
        )
        gex_profile_arr = self._calc_gex_at_levels(df, levels)
        gex_profile = {
            round(float(lv), 2): round(float(gx), 6)
            for lv, gx in zip(levels, gex_profile_arr)
        }

        # --- Step 5: キーレベル検出 ---
        zero_gamma = self._find_zero_gamma(levels, gex_profile_arr)
        call_wall  = self._find_call_wall(df, chain.spot)
        put_wall   = self._find_put_wall(df, chain.spot)

        # --- Step 6: 補助指標 ---
        gex_0dte, gex_0dte_ratio = self._calc_0dte_ratio(df, chain)
        dissonance_score, dissonance_dir = self._calc_dissonance(df, chain.spot)
        gamma_condition = self._judge_gamma_condition(
            chain.spot, zero_gamma
        )

        # --- Step 7: GEXSnapshot生成 ---
        total_gex   = float(df["call_gex"].sum() + df["put_gex"].sum()) / 1e9
        call_gex_total = float(df["call_gex"].sum()) / 1e9
        put_gex_total  = abs(float(df["put_gex"].sum())) / 1e9

        snap = GEXSnapshot(
            ticker=chain.ticker,
            timestamp=chain.timestamp,
            spot=chain.spot,
            zero_gamma=zero_gamma,
            call_wall=call_wall,
            put_wall=put_wall,
            gamma_condition=gamma_condition,
            total_gex=total_gex,
            call_gex_total=call_gex_total,
            put_gex_total=put_gex_total,
            gex_0dte=gex_0dte,
            gex_0dte_ratio=gex_0dte_ratio,
            dissonance_score=dissonance_score,
            dissonance_direction=dissonance_dir,
            gex_profile=gex_profile,
            iv_rank=chain.iv_rank,
            iv_percentile=chain.iv_percentile,
        )

        logger.info(f"[GEXEngine] {snap.summary()}")
        return snap

    # ==================================================================
    # PRIVATE: 計算ステップ
    # ==================================================================

    def _apply_oi_correction(self, df):
        """
        OI補正: effective_oi = OI + alpha * daily_volume
        OIはT+1更新のため、当日Volumeで日中ポジション増加を近似補正。
        """
        df = df.copy()
        df["call_effective_oi"] = (
            df["call_oi"] + self.oi_alpha * df["call_volume"]
        ).clip(lower=0)
        df["put_effective_oi"] = (
            df["put_oi"] + self.oi_alpha * df["put_volume"]
        ).clip(lower=0)
        return df

    def _fill_greeks_fallback(self, df, spot: float):
        """
        取引所提供のGammaがNoneの行をBS式で補完。
        gamma.pyのcalcGammaExのCall式をvectorized化。
        """
        df = df.copy()

        # Greeks提供済みの行はスキップ
        needs_calc = ~df["has_exchange_greeks"]
        if not needs_calc.any():
            logger.debug("[GEXEngine] All Greeks provided by exchange")
            return df

        sub = df[needs_calc].copy()
        S   = spot
        K   = sub["strike"].values
        T   = sub["days_to_expiry"].values
        T   = np.maximum(T, 1 / 1440)  # 最小1分を保証

        # Call Gamma（BS）
        iv_call = sub["call_iv"].values
        valid_call = iv_call > 0
        gamma_call = np.zeros(len(sub))
        if valid_call.any():
            gamma_call[valid_call] = self._bs_gamma(
                S, K[valid_call], iv_call[valid_call], T[valid_call]
            )

        # Put Gamma（理論上Callと同値。IVが異なる場合のみ別計算）
        iv_put = sub["put_iv"].values
        valid_put = iv_put > 0
        gamma_put = np.zeros(len(sub))
        if valid_put.any():
            gamma_put[valid_put] = self._bs_gamma(
                S, K[valid_put], iv_put[valid_put], T[valid_put]
            )

        df.loc[needs_calc, "call_gamma"] = gamma_call
        df.loc[needs_calc, "put_gamma"]  = gamma_put

        n_filled = needs_calc.sum()
        logger.debug(f"[GEXEngine] BS fallback applied to {n_filled} legs")
        return df

    def _calc_spot_gex(self, df, spot: float):
        """
        現在のスポット価格でのGEX計算。
        gamma.py L117-118の式を継承。

        GEX = Gamma * effective_OI * 100 * S^2 * 0.01
            = Gamma * effective_OI * S^2  （契約サイズ100 * 1%移動）
        """
        df = df.copy()
        S = spot

        df["call_gex"] = (
            df["call_gamma"].fillna(0)
            * df["call_effective_oi"]
            * 100
            * S * S
            * 0.01
        )
        # PutGEXは符号を反転（ディーラーはPutをShort→負のGamma）
        df["put_gex"] = (
            df["put_gamma"].fillna(0)
            * df["put_effective_oi"]
            * 100
            * S * S
            * 0.01
            * -1
        )
        df["total_gex"] = df["call_gex"] + df["put_gex"]
        return df

    def _calc_gex_at_levels(
        self, df, levels: np.ndarray
    ) -> np.ndarray:
        """
        各価格レベルでのGEXプロファイルをvectorized計算。
        gamma.py のforループをNumPy行列演算に置換。

        計算量: O(levels × legs) → 200 × 100 = 20,000演算（高速）
        """
        S_arr = levels[:, np.newaxis]          # (200, 1)
        K_arr = df["strike"].values[np.newaxis, :]      # (1, N)
        T_arr = df["days_to_expiry"].values[np.newaxis, :]
        T_arr = np.maximum(T_arr, 1 / 1440)

        iv_call = df["call_iv"].values[np.newaxis, :]
        iv_put  = df["put_iv"].values[np.newaxis, :]
        oi_call = df["call_effective_oi"].values[np.newaxis, :]
        oi_put  = df["put_effective_oi"].values[np.newaxis, :]

        # ブロードキャストで全レベル×全ストライクを一括計算
        gamma_call = np.where(
            iv_call > 0,
            self._bs_gamma(S_arr, K_arr, iv_call, T_arr),
            0.0,
        )
        gamma_put = np.where(
            iv_put > 0,
            self._bs_gamma(S_arr, K_arr, iv_put, T_arr),
            0.0,
        )

        gex_call = gamma_call * oi_call * 100 * S_arr * S_arr * 0.01
        gex_put  = gamma_put  * oi_put  * 100 * S_arr * S_arr * 0.01 * -1

        total = (gex_call.sum(axis=1) + gex_put.sum(axis=1)) / 1e9
        return total  # shape: (200,)

    def _find_zero_gamma(
        self, levels: np.ndarray, gex_arr: np.ndarray
    ) -> float:
        """
        Gamma Flip Pointの線形補間。
        gamma.py L188-196のロジックを関数化。
        符号が反転するゼロクロス点を検出。

        修正（v2）:
          ±10%レンジ内にゼロクロスが見つからない場合、
          ±25%の拡張レンジで再計算する。
          それでも見つからない場合は「完全Positive/Negative環境」と判断し、
          GEXが最もゼロに近いレベルをGamma Flipの推定値として返す。
        """
        zero_cross_idx = np.where(np.diff(np.sign(gex_arr)))[0]

        if len(zero_cross_idx) == 0:
            logger.warning(
                "[GEXEngine] No zero gamma crossing in ±10% range. "
                "Expanding search to ±25%..."
            )
            spot = float(levels[len(levels) // 2])  # レンジ中央 ≈ spot
            wide_levels = np.linspace(spot * 0.75, spot * 1.25, 500)
            wide_gex = self._calc_gex_at_levels(self._last_df, wide_levels)
            wide_cross = np.where(np.diff(np.sign(wide_gex)))[0]

            if len(wide_cross) == 0:
                # 完全にPositive or Negative → GEXが最小絶対値の点を返す
                logger.warning(
                    "[GEXEngine] No zero gamma found in ±25% range. "
                    "Market is in strong directional gamma environment."
                )
                return float(wide_levels[np.argmin(np.abs(wide_gex))])

            levels    = wide_levels
            gex_arr   = wide_gex
            zero_cross_idx = wide_cross

        # スポットに最も近いゼロクロスを採用
        spot_approx = float(levels[len(levels) // 2])
        closest_idx = min(
            zero_cross_idx,
            key=lambda i: abs(levels[i] - spot_approx)
        )
        idx = closest_idx

        neg_gamma  = gex_arr[idx]
        pos_gamma  = gex_arr[idx + 1]
        neg_strike = levels[idx]
        pos_strike = levels[idx + 1]

        # 線形補間
        zero_gamma = pos_strike - (
            (pos_strike - neg_strike) * pos_gamma / (pos_gamma - neg_gamma)
        )
        return float(zero_gamma)

    def _calc_notional_gex(self, df):
        """
        各ストライクの「潜在GEX（Notional GEX）」を計算。

        S = K（その価格に達した時のガンマ）で計算するのが業界標準。
        SpotGammaなど主要GEXプロバイダーと同一の手法。

        スポット価格ベース（Spot GEX）との違い:
          Spot GEX    → 「今の価格での各ストライクの影響」
          Notional GEX → 「その価格に達した時の潜在的な壁の強さ」
          → Call Wall / Put Wall検出にはNotional GEXが正しい
        """
        K   = df["strike"].values
        T   = np.maximum(df["days_to_expiry"].values, 1 / 1440)

        iv_call = df["call_iv"].values
        iv_put  = df["put_iv"].values
        oi_call = df["call_effective_oi"].values
        oi_put  = df["put_effective_oi"].values

        # S=K でガンマを計算（ATMガンマが最大になる）
        gamma_call = np.where(
            iv_call > 0, self._bs_gamma(K, K, iv_call, T), 0.0
        )
        gamma_put = np.where(
            iv_put > 0, self._bs_gamma(K, K, iv_put, T), 0.0
        )

        call_notional = gamma_call * oi_call * 100 * K * K * 0.01
        put_notional  = gamma_put  * oi_put  * 100 * K * K * 0.01 * -1

        return call_notional, put_notional

    def _find_call_wall(self, df, spot: float) -> WallLevel:
        """
        最大Call Notional GEX集積ストライク = Call Wall（赤・抵抗）

        修正（v2）: S=K のNotional GEXを使用。
        スポット上方のみを対象。
        """
        call_notional, _ = self._calc_notional_gex(df)
        df = df.copy()
        df["call_notional"] = call_notional

        above = df[df["strike"] >= spot]
        if above.empty:
            above = df

        by_strike = above.groupby("strike")["call_notional"].sum()
        total_call = df["call_notional"].sum()

        if by_strike.empty or total_call == 0:
            return WallLevel(spot, 0.0, 0.0, WallStrength.WEAK, "CALL")

        max_strike = float(by_strike.idxmax())
        max_val    = float(by_strike.max()) / 1e9
        ratio      = float(by_strike.max()) / total_call

        strength = self._classify_wall_strength(ratio)
        return WallLevel(max_strike, max_val, ratio, strength, "CALL")

    def _find_put_wall(self, df, spot: float) -> WallLevel:
        """
        最大Put Notional GEX集積ストライク = Put Wall（緑・支持）

        修正（v2）: S=K のNotional GEXを使用。
        スポット下方のみを対象。
        """
        _, put_notional = self._calc_notional_gex(df)
        df = df.copy()
        df["put_notional"] = put_notional

        below = df[df["strike"] <= spot]
        if below.empty:
            below = df

        by_strike = below.groupby("strike")["put_notional"].sum().abs()
        total_put = abs(df["put_notional"].sum())

        if by_strike.empty or total_put == 0:
            return WallLevel(spot, 0.0, 0.0, WallStrength.WEAK, "PUT")

        max_strike = float(by_strike.idxmax())
        max_val    = float(by_strike.max()) / 1e9
        ratio      = float(by_strike.max()) / total_put

        strength = self._classify_wall_strength(ratio)
        return WallLevel(max_strike, max_val, ratio, strength, "PUT")

    def _calc_0dte_ratio(self, df, chain: OptionChain):
        """
        0DTEオプション（当日満期）のGEXが全体に占める比率。
        比率が高いほど当日の値動きへの影響が大きい。
        """
        from datetime import date
        today = date.today()

        today_mask = df["expiry"].apply(
            lambda x: x.date() == today if hasattr(x, "date") else False
        )
        gex_0dte = float(
            df.loc[today_mask, "total_gex"].sum()
        ) / 1e9

        total_gex = float(df["total_gex"].sum()) / 1e9
        ratio = abs(gex_0dte / total_gex) if total_gex != 0 else 0.0

        return round(gex_0dte, 4), round(min(ratio, 1.0), 4)

    def _calc_dissonance(self, df, spot: float):
        """
        プレミアム乖離スコア（Dissonance）。

        定義:
          Put/Call プレミアム比率と現在のIVスキューを比較。
          「現物が落ち着いているのにPutプレミアムが異常に高い」
          = 機関がヘッジを積んでいる兆候 → Gold表示のトリガー

        スコア: 0.0（乖離なし）〜 1.0（最大乖離）
        """
        atm_mask = (
            (df["strike"] >= spot * 0.98) &
            (df["strike"] <= spot * 1.02)
        )
        atm = df[atm_mask]

        if atm.empty:
            return 0.0, "NEUTRAL"

        call_premium = (atm["call_mid"] * atm["call_effective_oi"]).sum()
        put_premium  = (atm["put_mid"]  * atm["put_effective_oi"]).sum()

        total = call_premium + put_premium
        if total == 0:
            return 0.0, "NEUTRAL"

        put_ratio  = put_premium / total
        call_ratio = call_premium / total

        # 0.5からの乖離をスコア化（0.5 = 完全均衡）
        score = abs(put_ratio - 0.5) * 2  # 0〜1にスケール

        if put_ratio > 0.6:
            direction = "PUT_PREMIUM"   # ダウンサイドヘッジ集積
        elif call_ratio > 0.6:
            direction = "CALL_PREMIUM"  # アップサイド投機集積
        else:
            direction = "NEUTRAL"

        return round(float(score), 4), direction

    def _judge_gamma_condition(
        self, spot: float, zero_gamma: float
    ) -> str:
        """
        スポットとZeroGammaの位置関係でガンマ環境を判定。

        spot > zero_gamma → POSITIVE（ディーラーLong Gamma → 往来相場）
        spot < zero_gamma → NEGATIVE（ディーラーShort Gamma → トレンド加速）
        """
        diff_pct = (spot - zero_gamma) / zero_gamma

        if abs(diff_pct) <= ZERO_GAMMA_NEUTRAL_BAND_PCT:
            return GammaCondition.NEUTRAL
        elif spot > zero_gamma:
            return GammaCondition.POSITIVE
        else:
            return GammaCondition.NEGATIVE

    # ==================================================================
    # PRIVATE: 数式コア（gamma.pyから継承・修正）
    # ==================================================================

    def _bs_gamma(
        self,
        S:   np.ndarray | float,
        K:   np.ndarray | float,
        vol: np.ndarray | float,
        T:   np.ndarray | float,
    ) -> np.ndarray | float:
        """
        Black-Scholesガンマ（Call式に統一）。
        gamma.py L34のCall式を継承。

        修正点:
          - Call/Putで異なる式を使っていたバグを修正（L34 vs L37）
          - CallとPutのガンマは理論上同値なのでCall式に統一
          - r（無リスク金利）とq（配当利回り）を明示的に使用
          - T=0防止はこの関数の外（呼び出し元）で保証

        引数:
          S   : 現物価格（スカラーまたはndarrayでブロードキャスト可）
          K   : ストライク
          vol : インプライドボラティリティ（小数。例: 0.15 = 15%）
          T   : 残存期間（年）

        戻り値:
          ガンマ値（スカラーまたはndarray）
        """
        dp = (
            np.log(S / K) + (self.r - self.q + 0.5 * vol ** 2) * T
        ) / (vol * np.sqrt(T))

        gamma = (
            np.exp(-self.q * T)
            * norm.pdf(dp)
            / (S * vol * np.sqrt(T))
        )
        return gamma

    # ==================================================================
    # PRIVATE: ユーティリティ
    # ==================================================================

    @staticmethod
    def _classify_wall_strength(ratio: float) -> str:
        if ratio >= WALL_STRONG_THRESHOLD:
            return WallStrength.STRONG
        elif ratio >= WALL_MEDIUM_THRESHOLD:
            return WallStrength.MEDIUM
        else:
            return WallStrength.WEAK
