"""
models/gex_snapshot.py

GEXEngineが計算した結果を格納するデータクラス。
このオブジェクトがVercel配信・Discord通知・TradingViewの
すべての出力の「単一の情報源」となる。

依存関係: なし（モデル層）
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


# ------------------------------------------------------------------
# GEX環境の定義
# ------------------------------------------------------------------

class GammaCondition:
    """
    ディーラーのガンマポジション環境。
    
    POSITIVE: ディーラーがLong Gamma
              → 価格上昇時に売り・下落時に買いでヘッジ
              → 市場はMean Reversion（壁に挟まれた往来相場）
              
    NEGATIVE: ディーラーがShort Gamma
              → 価格上昇時に買い・下落時に売りでヘッジ
              → 市場はTrend Following（動いた方向に加速）
    """
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL  = "NEUTRAL"   # ZeroGamma付近（±0.5%圏内）


class WallStrength:
    """GEXの壁の強度分類"""
    STRONG  = "STRONG"   # 市場全体のGEXの30%以上が集中
    MEDIUM  = "MEDIUM"   # 15〜30%
    WEAK    = "WEAK"     # 15%未満（ハリボテ候補）


# ------------------------------------------------------------------
# メインデータクラス
# ------------------------------------------------------------------

@dataclass
class WallLevel:
    """
    個別のGEX壁レベル（Call Wall / Put Wall / Secondary）。
    ConfluenceDetectorがPAシグナルと照合する単位。
    """
    strike: float
    gex_value: float            # その価格レベルの合計GEX（$B）
    gex_pct: float              # 市場全体GEXに対する比率（0〜1）
    strength: str               # WallStrength定数
    wall_type: str              # "CALL" / "PUT" / "ZERO_GAMMA"

    @property
    def is_strong(self) -> bool:
        return self.strength == WallStrength.STRONG

    @property
    def distance_pct(self) -> float:
        """スポット価格からの距離%（呼び出し元がspotを渡して計算）"""
        # GEXSnapshotのメソッド経由で使う設計
        return 0.0


@dataclass
class GEXSnapshot:
    """
    1回のGEX計算結果の完全なスナップショット。
    
    GEXEngine.calc_gex_profile() が生成し、
    以降の全処理（ConfluenceDetector / 出力層）が
    このオブジェクトを参照する。
    """

    # --- 基本情報 ---
    ticker: str
    timestamp: datetime
    spot: float

    # --- キーレベル ---
    zero_gamma: float               # Gamma Flip Point（最重要）
    call_wall: WallLevel            # 最大抵抗（赤）
    put_wall: WallLevel             # 最大支持（緑）

    # --- ガンマ環境 ---
    gamma_condition: str            # GammaCondition定数

    # --- GEX総量 ---
    total_gex: float                # 市場全体のGEX（$B）
    call_gex_total: float           # Call側合計
    put_gex_total: float            # Put側合計（正値で表記）

    # --- 0DTE専用指標 ---
    gex_0dte: float                 # 0DTEオプションのGEX（$B）
    gex_0dte_ratio: float           # 0DTE GEX / Total GEX（0〜1）

    # --- プレミアム乖離（Dissonance）---
    dissonance_score: float         # 0〜1。高いほど潜在的な大幅動意あり
    dissonance_direction: str       # "CALL_PREMIUM" / "PUT_PREMIUM" / "NEUTRAL"

    # --- ストライク別GEXプロファイル ---
    # {strike: total_gex_at_level} → TradingViewのヒートマップ描画に使用
    gex_profile: dict[float, float] = field(default_factory=dict)

    # --- セカンダリレベル（壁の候補リスト）---
    secondary_walls: list[WallLevel] = field(default_factory=list)

    # --- IV関連 ---
    iv_rank: Optional[float] = None         # 52週IVランク（0〜100）
    iv_percentile: Optional[float] = None

    # ------------------------------------------------------------------
    # 派生プロパティ
    # ------------------------------------------------------------------

    @property
    def is_positive_gamma(self) -> bool:
        return self.gamma_condition == GammaCondition.POSITIVE

    @property
    def is_negative_gamma(self) -> bool:
        return self.gamma_condition == GammaCondition.NEGATIVE

    @property
    def spot_vs_zero_gamma_pct(self) -> float:
        """スポットとZeroGammaの乖離率（正=スポット上、負=スポット下）"""
        return (self.spot - self.zero_gamma) / self.zero_gamma * 100

    @property
    def call_wall_distance_pct(self) -> float:
        """スポットからCall Wallまでの距離%（正値）"""
        return (self.call_wall.strike - self.spot) / self.spot * 100

    @property
    def put_wall_distance_pct(self) -> float:
        """スポットからPut Wallまでの距離%（正値）"""
        return (self.spot - self.put_wall.strike) / self.spot * 100

    @property
    def range_width_pct(self) -> float:
        """Put Wall〜Call Wall の幅（%）= 期待レンジ"""
        return (self.call_wall.strike - self.put_wall.strike) / self.spot * 100

    @property
    def is_near_call_wall(self, threshold_pct: float = 0.3) -> bool:
        """スポットがCall Wallから0.3%以内にいるか"""
        return self.call_wall_distance_pct <= threshold_pct

    @property
    def is_near_put_wall(self, threshold_pct: float = 0.3) -> bool:
        """スポットがPut Wallから0.3%以内にいるか"""
        return self.put_wall_distance_pct <= threshold_pct

    @property
    def is_near_zero_gamma(self, threshold_pct: float = 0.5) -> bool:
        """スポットがZeroGammaから0.5%以内にいるか"""
        return abs(self.spot_vs_zero_gamma_pct) <= threshold_pct

    # ------------------------------------------------------------------
    # 0DTE/1DTE 推奨ロジック（暫定版 / ConfluenceDetectorで上書き）
    # ------------------------------------------------------------------

    @property
    def iv_rank_regime(self) -> str:
        """
        IV Rankに基づく戦略レジーム。
        LOW  → プレミアム買い（Debit）優位
        HIGH → プレミアム売り（Credit）優位
        """
        if self.iv_rank is None:
            return "UNKNOWN"
        if self.iv_rank < 30:
            return "LOW"    # Buy Call / Buy Put
        if self.iv_rank > 60:
            return "HIGH"   # Sell Put Spread / Sell Call Spread
        return "NEUTRAL"

    # ------------------------------------------------------------------
    # シリアライズ
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        """
        Vercel配信・GitHub保存・Discord通知用。
        TradingView Pine Scriptが読む形式に合わせる。
        """
        return {
            "ticker":    self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "spot":      round(self.spot, 2),

            # キーレベル（TradingViewが水平線として描画）
            "levels": {
                "zero_gamma":  round(self.zero_gamma, 2),
                "call_wall":   round(self.call_wall.strike, 2),
                "put_wall":    round(self.put_wall.strike, 2),
            },

            # ガンマ環境
            "gamma_condition": self.gamma_condition,
            "total_gex":       round(self.total_gex, 3),

            # 0DTE
            "gex_0dte":        round(self.gex_0dte, 3),
            "gex_0dte_ratio":  round(self.gex_0dte_ratio, 3),

            # Dissonance（Gold表示のトリガー）
            "dissonance_score":     round(self.dissonance_score, 3),
            "dissonance_direction": self.dissonance_direction,

            # IV
            "iv_rank":       self.iv_rank,
            "iv_rank_regime": self.iv_rank_regime,

            # 壁の強度
            "call_wall_strength": self.call_wall.strength,
            "put_wall_strength":  self.put_wall.strength,

            # GEXヒートマップ（ストライク→GEX値）
            "gex_profile": {
                str(k): round(v, 4)
                for k, v in self.gex_profile.items()
            },
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False)

    # ------------------------------------------------------------------
    # 人間向け表示
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Discord通知・ログ用の1行サマリー"""
        condition_emoji = {
            GammaCondition.POSITIVE: "🟢",
            GammaCondition.NEGATIVE: "🔴",
            GammaCondition.NEUTRAL:  "⚪",
        }.get(self.gamma_condition, "❓")

        dissonance_emoji = "🟡" if self.dissonance_score > 0.6 else ""

        return (
            f"{condition_emoji} {self.ticker} @{self.spot:.2f} | "
            f"GammaFlip:{self.zero_gamma:.0f} | "
            f"CW:{self.call_wall.strike:.0f}({self.call_wall_distance_pct:+.1f}%) "
            f"PW:{self.put_wall.strike:.0f}({self.put_wall_distance_pct:+.1f}%) | "
            f"0DTE:{self.gex_0dte_ratio:.0%} "
            f"{dissonance_emoji}"
        )

    def __repr__(self) -> str:
        return (
            f"GEXSnapshot({self.ticker} spot={self.spot:.2f} "
            f"ZG={self.zero_gamma:.2f} "
            f"CW={self.call_wall.strike:.2f} "
            f"PW={self.put_wall.strike:.2f} "
            f"[{self.gamma_condition}])"
        )
