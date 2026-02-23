# Institutional GEX-PA Professional System
## 設計・実装ログ

> 最終更新：2026-02-21  
> ステータス：**Phase 1 進行中 — Step 3完了**

---

## 目次
1. [ミッション](#1-ミッション)
2. [確定事項（設計上の決定）](#2-確定事項設計上の決定)
3. [棄却事項（採用しないと決めたもの）](#3-棄却事項採用しないと決めたもの)
4. [未決事項（今後確認が必要なもの）](#4-未決事項今後確認が必要なもの)
5. [システムアーキテクチャ](#5-システムアーキテクチャ)
6. [クラス構造](#6-クラス構造)
7. [実装ロードマップ](#7-実装ロードマップ)
8. [進捗トラッカー](#8-進捗トラッカー)
9. [既知の技術的制約とその対策](#9-既知の技術的制約とその対策)
10. [重要な設計判断の根拠ログ](#10-重要な設計判断の根拠ログ)

---

## 1. ミッション

Saxo Bank OpenAPI（OPRAリアルタイムデータ）を基盤とし、機関投資家のプライスアクション（PA）とオプション市場の物理的需給（GEX）を融合させた、**0DTE/1DTE専用・SPY単独スタート**のプロ仕様オプション分析システムを構築する。

将来的なSaaS（サブスクリプション）展開を前提とした疎結合設計。

---

## 2. 確定事項（設計上の決定）

### 2-1. データソース
| データ種別 | ソース | 根拠 |
|-----------|--------|------|
| オプションチェーン（Bid/Ask/OI/Volume/IV） | **Saxo OpenAPI** | 口座保有済み・追加コストゼロ |
| 現物価格（リアルタイムストリーム） | **Saxo OpenAPI WebSocket** | 同上 |
| Greeks（Gamma/Delta） | **Saxo提供 → フォールバックでBS自計算** | 取引所提供条件付き。BS計算は実装コスト小 |
| 外部有料データ（Massive.com等） | **採用しない（現時点）** | Saxo単独で0DTE/1DTE GEXに必要なデータが揃う |

### 2-2. 対象銘柄・商品
- **初期スコープ：SPYのみ**（StockOption）
- 将来：クラス設計で動的ティッカー追加に対応済みにしておく

### 2-3. コアロジック

**GEX計算式（gamma.pyから継承・修正）**
- `calcGammaEx`のCall式に統一（CallとPutのガンマは理論上同値）
- 全計算はNumPy vectorized演算（行ループ禁止）
- T=0対策：最小残存時間を `1/1440`（1分）に設定
- OI補正式：`effective_oi = OI + 0.5 * daily_volume`（alphaは設定ファイルで調整可）

**フラクタルPAの層構造（確定）**
```
Context層  : W1, D1, H4   → 「今どのフェーズか」。これがなければTrigger無効
Trigger層  : H1, M30, M15 → 「セットアップが整ったか」
Entry層    : M5, M1        → 「今入るか」

スコア計算 : ContextScore × TriggerScore × EntryScore（乗算モデル）
             ※いずれかが0なら全体が0になる。加算モデルは採用しない。
```

**Volume-GEX判定閾値（確定）**
- Phase 1（現在）：固定閾値（20日平均の1.5倍）。`config.yaml`に外出し
- Phase 2（データ30日蓄積後）：銘柄別動的計算に移行
- フォールバック：当該銘柄データが20営業日未満の場合は固定に自動切替

### 2-4. アーキテクチャ（確定）

```
Mac（演算エンジン）
    │
    ├─→ [メモリキャッシュ] → FastAPIローカルサーバー（port 8000）  ← 即時（ms）
    │
    ├─→ [Vercel Webhook]  → スマホダッシュボード               ← 結果確定後即時POST
    │
    └─→ [GitHub Push]     → 履歴・バックアップ（5分毎バッチ）  ← 遅延許容
```

**重要**：Vercelへの配信はGitHubを経由しない。MacからVercel APIエンドポイントに直接POST。GitHubは履歴管理のみ。

### 2-5. Saxo Options Chain APIの仕様（確認済み）

- エンドポイント：`POST /trade/v1/optionschain/subscriptions`
- 最大ストライク数：**100本/リクエスト**（ウィンドウ方式）
- 最小リフレッシュレート：**2000ms**
- 取得フィールド：Bid, Ask, Volume, OpenInterest, MidVolatilityPct, Greeks（条件付き）, BidSize, AskSize, High/Low/Open/Close

### 2-6. 100ストライク制限の評価（問題なしと確定）

- SPY $580基準・$1刻みで**ATM±$50（±8.6%）**をカバー
- 0DTEのガンマはATMから±5%圏外で実質ゼロに収束 → カバー範囲は十分
- **唯一の実装上の注意点**：毎朝NY時間9:30前に`ResetATM`エンドポイントを自動実行してウィンドウ再センタリング（ギャップ対応）

---

## 3. 棄却事項（採用しないと決めたもの）

| 項目 | 棄却理由 |
|------|---------|
| TD Ameritrade API | 2023年Schwabに統合・廃止済み |
| CSVファイル手動読み込み | gamma.pyの旧設計。API連携で完全置換 |
| Massive.com（旧Polygon.io）Options Advanced | $199/月。Saxo単独で代替可能 |
| GitHub経由のVercelデータ配信 | Push遅延（最悪数十秒）が0DTE判断に致命的 |
| GEXスコアの加算モデル | 「1分足10個 = 日足1個」の問題。乗算モデルに変更 |
| Dashフロントエンド（app.pyのUI） | ロジックとUIが密結合。疎結合設計に移行 |

---

## 4. 未決事項（今後確認が必要なもの）

| # | 確認事項 | 優先度 | 確認方法 |
|---|---------|--------|---------|
| U-1 | Saxo Live環境のOAuth認証フロー（Simulation vs Live） | 🔴 高 | Developer Portal + 実接続テスト |
| U-2 | SPYのOption Root ID（Saxo内部識別子） | 🔴 高 | `GET /ref/v1/instruments?Keywords=SPY` |
| U-3 | SPYでGreeksが実際に返ってくるか | 🔴 高 | Simulation環境での実接続確認 |
| U-4 | Saxo OpenAPIのレート制限（詳細） | 🟡 中 | ドキュメント確認 + 実測 |
| U-5 | OHLCヒストリカルデータ（`/chart/v1/charts`）の取得可能期間 | 🟡 中 | 実接続確認。取れなければシステム稼働後に蓄積 |
| U-6 | ConfluenceDetectorのスコア重み（要バックテスト） | 🟢 低 | システム稼働後にデータ蓄積して調整 |
| U-7 | ダークプールデータの統合（補助ファクター） | 🟢 低 | Phase 4以降。Massive.com Stocks Developer $79/月。ConfluenceDetectorのWEIGHTSに予約枠のみ追加済み |

---

## 5. システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                    Mac（母艦）                            │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐                  │
│  │ SaxoAPIClient │    │  GEXEngine   │                  │
│  │  WebSocket   │───▶│  (NumPy)     │                  │
│  │  REST Polling│    └──────┬───────┘                  │
│  └──────────────┘           │                          │
│                             ▼                          │
│                    ┌──────────────┐                    │
│                    │FractalPAEngine│                    │
│                    │  9層並列     │                    │
│                    └──────┬───────┘                    │
│                           │                            │
│                           ▼                            │
│                  ┌─────────────────┐                   │
│                  │ConfluenceDetector│                   │
│                  │ (乗算スコア)    │                   │
│                  └──────┬──────────┘                   │
│                         │                              │
│          ┌──────────────┼──────────────┐               │
│          ▼              ▼              ▼               │
│  [メモリキャッシュ]  [Vercel直POST]  [GitHub Batch]    │
│  FastAPI :8000      即時配信         5分毎バックアップ  │
└─────────────────────────────────────────────────────────┘
          │                │
          ▼                ▼
   TradingView        スマホ
   Pine Script        ダッシュボード
   (可視化)           (Discord通知)
```

---

## 6. クラス構造

```
engine/
├── core/
│   ├── saxo_client.py        # SaxoAPIClient     ← Step 1で実装
│   ├── gex_engine.py         # GEXEngine         ← Step 2で実装（最初）
│   ├── pa_engine.py          # FractalPAEngine   ← Step 3で実装
│   ├── confluence.py         # ConfluenceDetector← Step 4で実装
│   └── scheduler.py          # MainOrchestrator  ← Step 5で統合
├── models/
│   ├── option_chain.py       # ✅ OptionChain, OptionLeg（dataclass）
│   ├── gex_snapshot.py       # ✅ GEXSnapshot（dataclass）
│   └── pa_signal.py          # PASignal, PALayer, Timeframe, PatternType
├── output/
│   ├── github_publisher.py   # 5分毎バッチPush
│   ├── discord_notifier.py   # 高スコア時アラート
│   └── vercel_bridge.py      # 直接POST（非GitHub経由）
└── config/
    └── settings.py           # 全設定値の集中管理（ハードコード禁止）
```

### 主要クラスの責務一覧

| クラス | 責務 | 依存関係 |
|--------|------|---------|
| `SaxoAPIClient` | Saxo API通信・OAuthトークン管理・WebSocket | なし（最下層） |
| `GEXEngine` | BS計算・GEXプロファイル・ZeroGamma/Wall算出 | OptionChain |
| `FractalPAEngine` | 9層OHLC取得・パターン検知・並列スキャン | SaxoAPIClient |
| `ConfluenceDetector` | GEX×PA統合スコア・方向性・限月推奨 | GEXSnapshot, PASignal |
| `MainOrchestrator` | asyncioイベントループ・全エンジン制御 | 全クラス |

---

## 7. 実装ロードマップ

### Phase 0：設計確定（✅ 完了）
- [x] 既存コード（gamma.py / app.py）の問題点洗い出し
- [x] Saxo OpenAPI オプションエンドポイント仕様確認
- [x] データソース選定（Saxo単独で完結と確定）
- [x] アーキテクチャ確定（GitHub迂回・Vercel直接POST）
- [x] クラス構造設計
- [x] PAの多層コンフルエンス設計（乗算モデル）
- [x] 100ストライク制限の影響評価

### Phase 1：GEXエンジン単体（🔄 進行中）
- [x] **Step 1**：`models/option_chain.py` — OptionChain / OptionLeg dataclass定義
- [x] **Step 2**：`models/gex_snapshot.py` — 計算結果クラス定義
- [x] **Step 3**：`core/gex_engine.py` — BS計算・GEXプロファイル・Wall検出
- [ ] **Step 4**：GEXEngineの単体テスト（モックデータで動作確認）

### Phase 2：Saxo API接続
- [ ] **Step 5**：`core/saxo_client.py` — OAuth認証フロー実装
- [ ] **Step 6**：Simulation環境でSPY Option Chain取得テスト
- [ ] **Step 7**：Greeksの提供有無確認 → フォールバック確認
- [ ] **Step 8**：ResetATM自動実行（9:30前）の実装

### Phase 3：PAエンジン
- [ ] **Step 9**：`models/pa_signal.py` — PAシグナルデータクラス
- [ ] **Step 10**：`core/pa_engine.py` — QM検知ロジック
- [ ] **Step 11**：Fakeout V1/V2/V3 検知
- [ ] **Step 12**：Compression + Supply/Demand Zone検知
- [ ] **Step 13**：9層並列スキャン（asyncio.gather）

### Phase 4：統合・出力
- [ ] **Step 14**：`core/confluence.py` — 乗算スコアモデル実装
- [ ] **Step 15**：`output/discord_notifier.py` — 高スコア時アラート
- [ ] **Step 16**：`output/vercel_bridge.py` — 直接POST
- [ ] **Step 17**：`output/github_publisher.py` — 5分毎バッチ
- [ ] **Step 18**：`core/scheduler.py` — MainOrchestrator統合

### Phase 5：フロントエンド・商用化準備
- [ ] **Step 19**：TradingView Pine Script（GEXヒートマップ・Walls）
- [ ] **Step 20**：スマホダッシュボード（Vercel）
- [ ] **Step 21**：バックテスト基盤
- [ ] **Step 22**：Volume動的閾値への移行（データ30日蓄積後）

---

## 8. 進捗トラッカー

| Phase | ステータス | 完了日 | メモ |
|-------|-----------|--------|------|
| Phase 0：設計確定 | ✅ 完了 | 2026-02-21 | |
| Phase 1：GEXエンジン単体 | 🔄 進行中 | — | Step 3完了 / Step 4（単体テスト）着手待ち |

### Step完了ログ
| Step | ファイル | 完了日 | 備考 |
|------|---------|--------|------|
| Step 1 | `engine/models/option_chain.py` | 2026-02-21 | OptionChain / OptionLeg dataclass。BSフォールバック対応済み |
| Step 2 | `engine/models/gex_snapshot.py` | 2026-02-22 | GEXSnapshot / WallLevel / GammaCondition。to_json()でTradingView配信形式に対応 |
| Step 3 | `engine/core/gex_engine.py`     | 2026-02-22 | BS計算vectorized化・OI補正・ZeroGamma/Wall検出・Dissonanceスコア実装 |
| Phase 2：Saxo API接続 | 🔲 未着手 | — | U-1〜U-3の確認が必要 |
| Phase 3：PAエンジン | 🔲 未着手 | — | |
| Phase 4：統合・出力 | 🔲 未着手 | — | |
| Phase 5：フロントエンド | 🔲 未着手 | — | |

---

## 9. 既知の技術的制約とその対策

| 制約 | 影響度 | 対策 |
|------|--------|------|
| OIはT+1更新 | 中 | `effective_oi = OI + 0.5 * volume`で日中補正 |
| 100ストライク制限 | 低（0DTE/1DTE用途では問題なし） | 毎朝9:30前にResetATM自動実行 |
| 最小リフレッシュ2秒 | 低 | 0DTE/1DTE判断に2秒は許容範囲 |
| Greeksは条件付き | 低 | BSフォールバックを実装（簡単） |
| GitHub Push遅延 | 解決済み | Vercelへ直接POST。GitHubは履歴のみ |
| PAの誤検知 | 中 | 乗算スコアモデル。Context層ゼロならEntry無効 |

---

## 10. 重要な設計判断の根拠ログ

### 2026-02-21：GEXスコア乗算モデルの採用

**判断**：PAのコンフルエンススコアを加算ではなく乗算で計算する。

**根拠**：加算モデルだと「1分足パターンが10個 ≒ 日足が1個」という数式上の等価が発生し、上位足の文脈なしで下位足シグナルだけでエントリーする事故を防げない。乗算にすることでContext層（W1/D1/H4）がゼロなら最終スコアも強制的にゼロになる。

### 2026-02-21：Massive.com（旧Polygon.io）の不採用

**判断**：現時点での外部有料データソース契約なし。

**根拠**：Saxo OpenAPIのOptions Chain APIでOI・Volume・IV・Greeksが取得可能と確認。0DTE/1DTE・SPY専用スコープでは100ストライク制限も問題なし。収益化後に必要なら再検討。

### 2026-02-21：GitHub経由Vercel配信の廃止

**判断**：MacからVercel APIエンドポイントに直接POSTする。GitHubは5分毎バッチのバックアップ専用。

**根拠**：GitHub Push → Vercel同期のレイテンシが最悪数十秒。0DTEの局面では致命的な遅延になりうる。

### 2026-02-21：Volume閾値の段階的移行戦略

**判断**：Phase 1は固定閾値（20日平均×1.5倍）。Phase 2でデータ30日蓄積後に動的計算へ移行。

**根拠**：固定は実装コストが低く即日稼働可。動的計算はヒストリカルデータが必要。移行条件（20営業日以上）を下回る場合は自動でフォールバックし、安全性を確保。

---

---

## 11. 参考コード・技術ナレッジ

### 11-1. Saty Pivot Ribbon（TradingView Pine Script V5）
**出典**: Saty Mahajan, 2022-2025  
**用途**: Step 19（Pine Script実装）時の技術参照

#### ① Time Warp（`request.security`でマルチタイムフレーム取得）
```javascript
// 現在チャートと異なる時間足のデータを取得するパターン
price = request.security(ticker, timeframe_func(), close, 
        gaps=barmerge.gaps_off, lookahead=barmerge.lookahead_on)
```
**転用先**: PA9層のTradingView側補完。GEXキーレベルは別足で計算したデータを受け取る際に使用。

#### ② Wall Touchアラートの実装パターン（Conviction Arrowsより）
```javascript
// クロスの瞬間だけを検出（再描画しない）
bullish_confirmed = condition[0] == true and condition[1] == false

// GEX Wall Touchへの応用例
near_call_wall  = math.abs(close - call_wall_level) / close < 0.003
touch_call_wall = near_call_wall[0] == true and near_call_wall[1] == false
plotshape(touch_call_wall, style=shape.triangledown, 
          color=color.red, location=location.abovebar)
```
**転用先**: Call Wall / Put Wall / ZeroGammaのタッチ瞬間を検出するアラートロジック。

#### ③ リアルタイムバー再描画防止
```javascript
// リアルタイムバーでは1本前の確定値を使う
ta.ema(price, length)[barstate.isrealtime ? 1 : 0]
```
**転用先**: Macから送られるGEXレベルをプロットする際に必須。これを怠るとアラートが再描画される。

#### ④ GEX環境によるローソク色変え（Candle Biasより）
```javascript
// GEX POSITIVE/NEGATIVEでローソク色を変える実装例
gex_candle_color =
    gamma_positive and close > open ? color.green :
    gamma_positive and close < open ? color.new(color.green, 60) :
    gamma_negative and close > open ? color.new(color.red, 60) :
    gamma_negative and close < open ? color.red :
    color.gray
plotcandle(open, high, low, close, color=gex_candle_color,
           bordercolor=gex_candle_color, wickcolor=gex_candle_color)
```
**転用先**: Step 19でGEX環境の視覚化に直接使用。

---

*このファイルはセッション開始時に必ず参照し、決定事項を追記・更新していく。*
