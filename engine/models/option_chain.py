"""
models/option_chain.py

Saxo OpenAPI の Options Chain レスポンスを受け取るデータクラス。
全フィールドはSaxo APIドキュメントの実フィールド名に準拠。

依存関係: なし（最下層モデル）
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
import pandas as pd


@dataclass
class OptionLeg:
    """
    1ストライク分のCall/Put両サイドのデータ。
    Saxo Options Chain APIの "Strikes" 配列の1要素に対応。

    Greeksフィールドはオプショナル:
      - 取引所がGreeksを提供する場合 → Saxoから直接取得
      - 提供しない場合 → GEXEngineがBS式でフォールバック計算
    """

    # --- 基本情報 ---
    strike: float                       # ストライク価格
    expiry: datetime                    # 満期日時（NY時間 16:00を付加）

    # --- Call サイド ---
    call_uic: Optional[int] = None      # Saxo内部識別子（注文時に使用）
    call_bid: float = 0.0
    call_ask: float = 0.0
    call_bid_size: float = 0.0
    call_ask_size: float = 0.0
    call_volume: int = 0                # 当日出来高（リアルタイム）
    call_oi: int = 0                    # Open Interest（T+1更新）
    call_iv: float = 0.0               # MidVolatilityPct（小数表記: 0.15 = 15%）
    call_delta: Optional[float] = None  # DeltaPct（Saxo提供時のみ）
    call_gamma: Optional[float] = None  # Greeks.Gamma（Saxo提供時のみ）
    call_last: float = 0.0
    call_high: float = 0.0
    call_low: float = 0.0
    call_open: float = 0.0
    call_close: float = 0.0            # 前日終値

    # --- Put サイド ---
    put_uic: Optional[int] = None
    put_bid: float = 0.0
    put_ask: float = 0.0
    put_bid_size: float = 0.0
    put_ask_size: float = 0.0
    put_volume: int = 0
    put_oi: int = 0
    put_iv: float = 0.0
    put_delta: Optional[float] = None
    put_gamma: Optional[float] = None
    put_last: float = 0.0
    put_high: float = 0.0
    put_low: float = 0.0
    put_open: float = 0.0
    put_close: float = 0.0

    # --- OI補正済み実効OI（GEXEngine が書き込む）---
    # effective_oi = oi + alpha * daily_volume
    call_effective_oi: float = 0.0
    put_effective_oi: float = 0.0

    def __post_init__(self):
        """基本バリデーション"""
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.call_iv < 0 or self.put_iv < 0:
            raise ValueError("IV cannot be negative")

    @property
    def mid_iv(self) -> float:
        """Call/Put IVの平均（スマイル対称性の簡易チェック用）"""
        if self.call_iv > 0 and self.put_iv > 0:
            return (self.call_iv + self.put_iv) / 2
        return self.call_iv or self.put_iv

    @property
    def call_mid(self) -> float:
        """Callのミッドプライス"""
        return (self.call_bid + self.call_ask) / 2 if self.call_ask > 0 else 0.0

    @property
    def put_mid(self) -> float:
        """Putのミッドプライス"""
        return (self.put_bid + self.put_ask) / 2 if self.put_ask > 0 else 0.0

    @property
    def days_to_expiry(self) -> float:
        """
        残存日数（営業日ベース）。
        0DTE当日は市場残り時間を分単位で換算。
        最小値: 1/1440（1分）→ T=0によるBS計算エラーを防止。
        """
        now = datetime.utcnow()
        delta_seconds = (self.expiry - now).total_seconds()
        if delta_seconds <= 0:
            return 1 / 1440
        # 年換算（252営業日ベース）
        dte = delta_seconds / (252 * 24 * 3600)
        return max(dte, 1 / 1440)

    @property
    def has_exchange_greeks(self) -> bool:
        """取引所提供のGreeksが存在するか"""
        return self.call_gamma is not None and self.put_gamma is not None


@dataclass
class OptionChain:
    """
    1銘柄・1取得時刻のオプションチェーン全体。
    Saxo Options Chain サブスクリプションの Snapshot に対応。

    用途:
      - GEXEngine.calc_gex_profile() に渡すメインの入力オブジェクト
      - filter_by_expiry() で0DTE/1DTEを分離して渡す
    """

    ticker: str                         # 例: "SPY"
    uic: int                            # Saxo内部のOption Root ID
    spot: float                         # 現物価格（取得時点）
    timestamp: datetime                 # データ取得時刻（UTC）
    legs: list[OptionLeg] = field(default_factory=list)

    # --- 補助情報（別途取得）---
    iv_rank: Optional[float] = None     # 52週IVランク（0〜100）
    iv_percentile: Optional[float] = None

    def __post_init__(self):
        if self.spot <= 0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if not self.legs:
            # 空チェーンは警告のみ（初期化直後は空でOK）
            pass

    # ------------------------------------------------------------------
    # フィルタリング
    # ------------------------------------------------------------------

    def filter_by_expiry(self, target_date: date) -> OptionChain:
        """指定日のストライクのみを含む新しいOptionChainを返す"""
        filtered = [
            leg for leg in self.legs
            if leg.expiry.date() == target_date
        ]
        return OptionChain(
            ticker=self.ticker,
            uic=self.uic,
            spot=self.spot,
            timestamp=self.timestamp,
            legs=filtered,
            iv_rank=self.iv_rank,
            iv_percentile=self.iv_percentile,
        )

    def get_0dte(self) -> OptionChain:
        """当日満期のストライクのみを返す"""
        today = datetime.utcnow().date()
        return self.filter_by_expiry(today)

    def get_1dte(self) -> OptionChain:
        """翌営業日満期のストライクのみを返す"""
        from datetime import timedelta
        # 簡易実装: 翌日。週末・祝日は呼び出し元で制御
        tomorrow = (datetime.utcnow() + timedelta(days=1)).date()
        return self.filter_by_expiry(tomorrow)

    def filter_near_spot(self, pct: float = 0.10) -> OptionChain:
        """
        スポット価格の±pct%圏内のストライクのみを返す。
        デフォルト: ±10%（GEXプロファイル計算の標準レンジ）
        """
        lower = self.spot * (1 - pct)
        upper = self.spot * (1 + pct)
        filtered = [
            leg for leg in self.legs
            if lower <= leg.strike <= upper
        ]
        return OptionChain(
            ticker=self.ticker,
            uic=self.uic,
            spot=self.spot,
            timestamp=self.timestamp,
            legs=filtered,
            iv_rank=self.iv_rank,
            iv_percentile=self.iv_percentile,
        )

    # ------------------------------------------------------------------
    # 変換
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        GEXEngineでのNumPy演算に渡すためのDataFrame。
        各行が1ストライク。
        """
        if not self.legs:
            return pd.DataFrame()

        records = []
        for leg in self.legs:
            records.append({
                "strike":             leg.strike,
                "expiry":             leg.expiry,
                "days_to_expiry":     leg.days_to_expiry,
                "call_iv":            leg.call_iv,
                "call_oi":            leg.call_oi,
                "call_volume":        leg.call_volume,
                "call_effective_oi":  leg.call_effective_oi,
                "call_gamma":         leg.call_gamma,      # Noneの場合はBS計算
                "call_delta":         leg.call_delta,
                "call_bid":           leg.call_bid,
                "call_ask":           leg.call_ask,
                "call_mid":           leg.call_mid,
                "put_iv":             leg.put_iv,
                "put_oi":             leg.put_oi,
                "put_volume":         leg.put_volume,
                "put_effective_oi":   leg.put_effective_oi,
                "put_gamma":          leg.put_gamma,
                "put_delta":          leg.put_delta,
                "put_bid":            leg.put_bid,
                "put_ask":            leg.put_ask,
                "put_mid":            leg.put_mid,
                "has_exchange_greeks": leg.has_exchange_greeks,
            })

        df = pd.DataFrame(records)
        df.sort_values("strike", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def to_json(self) -> dict:
        """Vercel配信・GitHub保存用のシリアライズ"""
        return {
            "ticker":    self.ticker,
            "uic":       self.uic,
            "spot":      self.spot,
            "timestamp": self.timestamp.isoformat(),
            "iv_rank":   self.iv_rank,
            "leg_count": len(self.legs),
            # legsの詳細は容量が大きいためGEXSnapshotで別途出力
        }

    # ------------------------------------------------------------------
    # 統計（簡易）
    # ------------------------------------------------------------------

    @property
    def atm_leg(self) -> Optional[OptionLeg]:
        """スポットに最も近いストライクのLegを返す"""
        if not self.legs:
            return None
        return min(self.legs, key=lambda leg: abs(leg.strike - self.spot))

    @property
    def total_call_oi(self) -> int:
        return sum(leg.call_oi for leg in self.legs)

    @property
    def total_put_oi(self) -> int:
        return sum(leg.put_oi for leg in self.legs)

    @property
    def put_call_oi_ratio(self) -> float:
        """Put/Call OI比率。1.0以上でプット優勢（弱気ヘッジ多め）"""
        if self.total_call_oi == 0:
            return 0.0
        return self.total_put_oi / self.total_call_oi

    def __repr__(self) -> str:
        return (
            f"OptionChain(ticker={self.ticker}, spot={self.spot:.2f}, "
            f"legs={len(self.legs)}, timestamp={self.timestamp.strftime('%H:%M:%S')})"
        )
