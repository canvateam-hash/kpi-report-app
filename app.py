# -*- coding: utf-8 -*-
"""
KPIレポート作成システム（単一ファイル版）
ANALYSIS_KPI.md / KPI分析_完全引き継ぎプロンプト v3 のロジックに準拠。

このファイル1つだけで動作します（modulesフォルダは不要）。
サイドバーで全データを一括アップロード → 上部タブで①〜⑫を自由に切り替え表示。
実行方法: streamlit run app.py
"""
import io
import datetime
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ===== from modules/scoring.py =====
# -*- coding: utf-8 -*-
"""
売上金額の正常/異常（与信NG等）判定ロジック。

KPI分析_完全引き継ぎプロンプト v3 で検証済みの「対応状況・決済状況スコア方式」を採用。
キーワードベース判定より正確なため、こちらを標準ロジックとする。
"""

# 対応状況スコア（正常=1 / 異常=0）
TAIOU_SCORE = {
    "注文確定": 1, "出荷準備中": 1, "発送完了": 1, "配送完了": 1,
    "キャンセル": 0, "支払エラー": 0, "報酬否認": 0, "返品": 0,
}

# 決済状況スコア（正常=1 / 異常=0）
KESSAI_SCORE = {
    "与信審査完了": 1, "仮売上完了": 1, "売上完了": 1,
    "与信審査エラー": 0, "仮売上失敗": 0, "取消完了": 0,
    "取引修正失敗": 0, "取引登録失敗": 0, "与信保留": 0,
}


def compute_amount_for_kpi(
    df: pd.DataFrame,
    taiou_col: str = "対応状況",
    kessai_col: str = "決済状況",
    amount_col: str = "合計",
) -> pd.DataFrame:
    """
    「合計（計算用）」列を追加して返す。
    対応状況スコア + 決済状況スコア = 2（両方正常）のときのみ売上を計上、
    それ以外（与信NG・キャンセル等のシステム的キャンセル）は0とする。

    未知のステータス値（マップに存在しない値）は0点として扱う（v3ロジック踏襲）。
    ただし未知値が存在する場合は quality_notes に記録し、勝手な判断で無視しないようにする。
    ・スコア列（0/1/2）も同時に付与する。用途に応じて使い分ける：
        - スコア=2のみ「完全購入」→ 合計（計算用）はこの基準（LTV等の金額系KPI用）
        - スコア>=1／<=1 のような閾値判定は、呼び出し側でスコア列を直接使うこと（①初回離脱率など）
    """
    df = df.copy()
    quality_notes = []

    bp = df[taiou_col].map(TAIOU_SCORE)
    bq = df[kessai_col].map(KESSAI_SCORE)

    unknown_taiou = sorted(set(df.loc[bp.isna(), taiou_col].dropna().unique().tolist()))
    unknown_kessai = sorted(set(df.loc[bq.isna(), kessai_col].dropna().unique().tolist()))
    if unknown_taiou:
        quality_notes.append(
            f"「{taiou_col}」列に未定義の値があります（0点=異常として扱いました）: {unknown_taiou}"
        )
    if unknown_kessai:
        quality_notes.append(
            f"「{kessai_col}」列に未定義の値があります（0点=異常として扱いました）: {unknown_kessai}"
        )

    bp = bp.fillna(0)
    bq = bq.fillna(0)
    df["スコア"] = (bp + bq).round(0).astype(int)  # 0=両方異常 / 1=片方正常 / 2=両方正常（完全購入）

    # 合計（計算用）：LTV等の金額系KPIで使用。スコア=2（完全購入）のときのみ金額を計上。
    df["合計（計算用）"] = df.apply(
        lambda r: r[amount_col] if r["スコア"] == 2 else 0, axis=1
    )

    df.attrs["quality_notes"] = df.attrs.get("quality_notes", []) + quality_notes
    return df


# ===== from modules/data_loader.py =====
# -*- coding: utf-8 -*-
"""
CSVファイルの読み込みユーティリティ。
ECフォース出力のCSVは cp932（Shift-JIS系）エンコーディングが標準。
"""

ENCODING_CANDIDATES = ["cp932", "shift_jis", "utf-8-sig", "utf-8"]


def read_csv_flexible(file) -> pd.DataFrame:
    """複数エンコーディングを順に試して読み込む。"""
    last_err = None
    for enc in ENCODING_CANDIDATES:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            return df
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    raise ValueError(f"CSVの読み込みに失敗しました（対応エンコーディング: {ENCODING_CANDIDATES}）: {last_err}")


def load_multiple_csv(files, required_columns=None, dedup_key=None) -> tuple[pd.DataFrame, list[str]]:
    """
    複数ファイルを読み込み結合する。将来的に月次でファイルを追加アップロードしていく運用を想定。

    Returns:
        (結合済みDataFrame, 注記リスト)
    """
    notes = []
    frames = []
    for f in files:
        try:
            df = read_csv_flexible(f)
        except Exception as e:
            notes.append(f"❌ 「{f.name}」の読み込みに失敗しました: {e}")
            continue

        if required_columns:
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                notes.append(
                    f"❌ 「{f.name}」に必須列が見つかりません: {missing}。このファイルはスキップしました。"
                )
                continue

        df = df.copy()
        df["_source_file"] = f.name
        frames.append(df)

    if not frames:
        return pd.DataFrame(), notes

    combined = pd.concat(frames, ignore_index=True)

    if dedup_key and dedup_key in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset=[dedup_key], keep="last")
        after = len(combined)
        if before != after:
            notes.append(
                f"ℹ️ 重複データを{before - after}件除去しました（キー: {dedup_key}、複数ファイルで同一IDが存在）。"
            )

    return combined, notes


# ===== from modules/customer_master.py =====
# -*- coding: utf-8 -*-
"""
顧客マスタ（コホート起点）の生成。
ANALYSIS_KPI.md STEP1/STEP2 に対応。

「初回購入日（登録月）」＝ 顧客ごとの受注データ全体における最も古い受注日を年月に変換したもの。
このモジュールは受注データ（sales_data系）を対象とする。
"""


def build_customer_master(
    df_orders: pd.DataFrame,
    customer_col: str = "顧客番号",
    date_col: str = "受注日",
) -> tuple[pd.DataFrame, list[str]]:
    """
    顧客マスタを生成する。

    Returns:
        (customer_master, notes)
        customer_master columns: [顧客番号, 初回購入日, 登録月, 最終購入日]
    """
    notes = []
    df = df_orders.copy()

    df["_受注日_dt"] = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    n_invalid = df["_受注日_dt"].isna().sum()
    if n_invalid > 0:
        notes.append(
            f"⚠️ 「{date_col}」列が日付として解釈できない行が{n_invalid}件あります（顧客マスタ集計から除外）。"
        )

    valid = df.dropna(subset=["_受注日_dt", customer_col])

    grp = valid.groupby(customer_col)["_受注日_dt"].agg(["min", "max"]).reset_index()
    grp = grp.rename(columns={"min": "初回購入日", "max": "最終購入日"})
    grp["登録月"] = grp["初回購入日"].dt.to_period("M").astype(str)  # 例: 2026-05

    return grp, notes


def attach_registration_month(
    df_orders: pd.DataFrame,
    customer_master: pd.DataFrame,
    customer_col: str = "顧客番号",
) -> pd.DataFrame:
    """
    STEP2: 全ての受注行に「登録月」列を紐付ける（VLOOKUP相当）。
    """
    df = df_orders.copy()
    lookup = customer_master.set_index(customer_col)["登録月"]
    df["登録月"] = df[customer_col].map(lookup)
    return df


def add_order_month(
    df_orders: pd.DataFrame,
    date_col: str = "受注日",
) -> tuple[pd.DataFrame, list[str]]:
    """
    受注日から「受注月」列（YYYY-MM形式の文字列）を付与する。
    複数月のデータが1ファイルに混在しているケースに対応するため、
    「登録月（コホート月）」とは別に、各行そのものが実際にいつの受注かを判定できるようにする。
    """
    notes = []
    df = df_orders.copy()
    dt = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    n_invalid = dt.isna().sum()
    if n_invalid > 0:
        notes.append(f"⚠️ 「{date_col}」列が日付として解釈できない行が{n_invalid}件あります。")
    df["受注月"] = dt.dt.to_period("M").astype(str)
    df.loc[dt.isna(), "受注月"] = None
    return df, notes


def summarize_order_months(df_orders: pd.DataFrame, order_month_col: str = "受注月") -> pd.DataFrame:
    """アップロードデータに含まれる受注月ごとの件数を集計する（データ範囲の確認用）。"""
    if order_month_col not in df_orders.columns:
        return pd.DataFrame()
    counts = df_orders[order_month_col].value_counts(dropna=True).sort_index()
    return counts.rename_axis("受注月").reset_index(name="件数")


def get_valid_first_orders(
    df: pd.DataFrame,
    teiki_kaisu_col: str = "定期回数",
    order_month_col: str = "受注月",
) -> tuple[pd.DataFrame, list[str]]:
    """
    「真の初回購入」行を抽出する共通ロジック（①初回離脱率・②期間別解約率など、
    定期回数=1を対象とする全KPIで共有する）。

    - 定期回数 = 1 の行のみを対象
    - 登録月が特定できない行は除外
    - 受注月が登録月と一致しない行（再定期などで定期回数がリセットされたケース）は
      真の初回購入ではないため除外し、件数を注記する

    Returns:
        (first_orders_df, notes)
    """
    notes = []

    if teiki_kaisu_col not in df.columns:
        notes.append(f"❌ 「{teiki_kaisu_col}」列が見つかりません。")
        return pd.DataFrame(), notes

    if "登録月" not in df.columns:
        notes.append("❌ 「登録月」列が見つかりません。顧客マスタとの結合を先に行ってください。")
        return pd.DataFrame(), notes

    n_no_cohort = df["登録月"].isna().sum()
    if n_no_cohort > 0:
        notes.append(
            f"⚠️ 登録月が特定できない受注が{n_no_cohort}件あります（顧客番号の不一致等）。集計から除外しました。"
        )

    first_orders = df[(df[teiki_kaisu_col] == 1) & (df["登録月"].notna())].copy()

    if first_orders.empty:
        notes.append("⚠️ 定期回数=1の行が見つかりませんでした。列名・データ内容をご確認ください。")
        return pd.DataFrame(), notes

    if order_month_col in first_orders.columns:
        mismatch = first_orders[
            first_orders[order_month_col].notna()
            & (first_orders[order_month_col] != first_orders["登録月"])
        ]
        if len(mismatch) > 0:
            notes.append(
                f"⚠️ 「定期回数=1」だが受注月が登録月と一致しない行が{len(mismatch)}件あります"
                f"（再定期など、真の初回購入ではない可能性があるため集計から除外しました）。"
                f" 該当登録月: {sorted(mismatch['登録月'].unique().tolist())}"
            )
        first_orders = first_orders[
            first_orders[order_month_col].isna()
            | (first_orders[order_month_col] == first_orders["登録月"])
        ]
    else:
        notes.append(
            f"ℹ️ 「{order_month_col}」列がないため、受注月と登録月の整合性チェックは行っていません。"
        )

    return first_orders, notes


# ===== from modules/teiki_loader.py =====
# -*- coding: utf-8 -*-
"""
定期受注データ（sales_teiki系CSV）の読み込み。
⑧顧客推移データ・⑨稼働顧客数（アクティブ判定）・⑪解約理由 で使用する。
"""


REQUIRED_TEIKI_COLUMNS = ["顧客番号", "ステータス", "作成日", "定期回数"]


def load_teiki_data(files) -> tuple[pd.DataFrame, list[str]]:
    """定期受注データCSV（複数月分）を読み込み結合する。"""
    df, notes = load_multiple_csv(
        files,
        required_columns=REQUIRED_TEIKI_COLUMNS,
        dedup_key="定期受注ID",
    )
    return df, notes


# ===== from modules/continuation_data.py =====
# -*- coding: utf-8 -*-
"""
定期継続率データ（全件・縛りあり・縛りなし）の読み込み。

EC フォース_分析管理_定期継続率分析からダウンロードしたCSVをそのまま使用する。
CSV内の数値は再計算せず、そのまま使う（ANALYSIS_KPI.md ④の定義に準拠）。

このデータは③F2転換率（n~n+1回＝"0～1"の行）と④定期継続率（全行）の両方の元データになる。
"""


REQUIRED_COLUMNS = [
    "定期受注作成日（年月）", "n~n+1回", "継続率 (%)", "継続予定率 (%)",
    "合計 (件)", "売上済 (件)", "売上前 (件)", "待機中 (件)", "離脱 (件)",
]

SEGMENT_LABELS = {
    "all": "全体",
    "ari": "定期縛りあり",
    "nashi": "定期縛りなし",
}


def load_continuation_segment(files, segment_key: str) -> tuple[pd.DataFrame, list[str]]:
    """
    1つのセグメント（全体／縛りあり／縛りなし）分のファイル群を読み込み結合する。
    複数月分をまとめてアップロードした場合はすべて結合する。
    """
    notes = []
    frames = []
    for f in files:
        try:
            df = read_csv_flexible(f)
        except Exception as e:
            notes.append(f"❌ 「{f.name}」の読み込みに失敗しました: {e}")
            continue

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            notes.append(f"❌ 「{f.name}」に必須列が見つかりません: {missing}。このファイルはスキップしました。")
            continue

        df = df.copy()
        df["_source_file"] = f.name
        frames.append(df)

    if not frames:
        return pd.DataFrame(), notes

    combined = pd.concat(frames, ignore_index=True)
    combined["区分"] = SEGMENT_LABELS[segment_key]

    # 登録月の正規化（例: '2026/05' -> '2026-05'）
    combined["登録月"] = (
        combined["定期受注作成日（年月）"].astype(str).str.replace("/", "-", regex=False)
    )
    combined["登録月"] = combined["登録月"].apply(_normalize_month)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["登録月", "n~n+1回"], keep="last")
    after = len(combined)
    if before != after:
        notes.append(
            f"ℹ️ {SEGMENT_LABELS[segment_key]}：重複データを{before - after}件除去しました"
            f"（同一の登録月×n~n+1回が複数ファイルに存在）。"
        )

    return combined, notes


def _normalize_month(month_str: str) -> str:
    """'2026-05' や '2026-5' を '2026-05' に正規化する。"""
    try:
        year, month = month_str.split("-")
        return f"{int(year):04d}-{int(month):02d}"
    except (ValueError, AttributeError):
        return month_str


def load_all_continuation_segments(
    files_all, files_ari, files_nashi
) -> tuple[pd.DataFrame, list[str]]:
    """3セグメント分をまとめて読み込み、1つのDataFrameに結合する。"""
    all_notes = []
    frames = []

    for files, key in [(files_all, "all"), (files_ari, "ari"), (files_nashi, "nashi")]:
        if not files:
            continue
        df, notes = load_continuation_segment(files, key)
        all_notes.extend(notes)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(), all_notes

    combined = pd.concat(frames, ignore_index=True)
    return combined, all_notes


# ===== from modules/kpi01_churn.py =====
# -*- coding: utf-8 -*-
"""
① 初回離脱率

定義：登録月ごとの初回購入者のうち、1回目の購入後に定期を「停止」または「キャンセル」した顧客の割合。
除外ルール：与信NG・決済エラー等のシステム的キャンセルは、離脱者（分子）・初回購入者（分母）双方から除外。

ロジック（ユーザー確定版）：
- スコア = 対応状況スコア（0/1）＋ 決済状況スコア（0/1） → 0・1・2のいずれか
- 定期回数 = 1 の行を対象（真の初回購入のみ。customer_master.get_valid_first_orders で抽出）
- 分母（初回購入者）＝ スコア >= 1 の件数（片方でも正常なら「購入した」とみなす）
- 分子（初回離脱者）＝ スコア <= 1 の件数（完全に成立していなければ「離脱」扱い）
  → スコア=1の行は分母・分子の両方にカウントされる
  → スコア=2（両方正常＝完全に購入）のみが「離脱していない」ことになる
- 離脱率 = 分子 ÷ 分母
- 集計軸：登録月（顧客の初回購入月）
"""



def calculate_first_time_churn(
    df_orders_with_cohort: pd.DataFrame,
    teiki_kaisu_col: str = "定期回数",
    order_month_col: str = "受注月",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Args:
        df_orders_with_cohort: 受注データに「登録月」「受注月」列が付与されたDataFrame
                                （customer_master.attach_registration_month / add_order_month の出力）

    Returns:
        (result_df, notes)
        result_df columns: [登録月, 初回購入者数, 離脱者数, 初回離脱率]
    """
    notes = []
    df = compute_amount_for_kpi(df_orders_with_cohort)
    notes.extend(df.attrs.get("quality_notes", []))

    first_orders, fo_notes = get_valid_first_orders(df, teiki_kaisu_col, order_month_col)
    notes.extend(fo_notes)

    if first_orders.empty:
        return pd.DataFrame(), notes

    rows = []
    for month, sub in first_orders.groupby("登録月"):
        denom = int((sub["スコア"] >= 1).sum())  # 初回購入者：片方でも正常
        numer = int((sub["スコア"] <= 1).sum())  # 初回離脱者：完全成立していない
        rate = numer / denom if denom > 0 else None
        rows.append({
            "登録月": month,
            "初回購入者数": denom,
            "離脱者数": numer,
            "初回離脱率": rate,
        })

    result = pd.DataFrame(rows).sort_values("登録月").reset_index(drop=True)
    return result, notes


# ===== from modules/kpi02_cancellation.py =====
# -*- coding: utf-8 -*-
"""
② 期間別解約率

対象：定期回数=1（真の初回購入のみ）

ロジック（ユーザー確定版）：
- 出荷件数：「発送日」に実データが入っている件数（＝実際に発送された件数）
- キャンセル件数：「対応状況」＝キャンセル の件数
  （※実データ上、出荷済とキャンセルは完全に排他。キャンセルは発送前に確定する業務フローのため）
- 決済保留・エラー件数：「対応状況」はキャンセルでなくても、「決済状況」が
  与信保留／与信審査エラー／仮売上失敗／取引修正失敗 のいずれかに該当する件数
  （キャンセル件数とは別枠で並列集計する。ユーザー確認済み）
- 期間別解約：キャンセルとなった行について、「発送予定日」を出荷日とみなし、
  「更新日」との差分日数を計算。7日以内・14日以内の件数と、
  キャンセル合計に対する割合を算出する。
- 集計軸：登録月（顧客の初回購入月）
"""


PAYMENT_ERROR_STATUSES = ["与信保留", "与信審査エラー", "仮売上失敗", "取引修正失敗"]


def calculate_period_cancellation(
    df_orders_with_cohort: pd.DataFrame,
    df_teiki: pd.DataFrame = None,
    customer_master: pd.DataFrame = None,
    teiki_kaisu_col: str = "定期回数",
    order_month_col: str = "受注月",
    taiou_col: str = "対応状況",
    kessai_col: str = "決済状況",
    shipped_date_col: str = "発送日",
    schedule_date_col: str = "発送予定日",
    update_date_col: str = "更新日",
    customer_col: str = "顧客番号",
    teiki_status_col: str = "ステータス",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns:
        (result_df, notes)
        result_df columns:
            登録月, 出荷件数, キャンセル件数, 停止件数, 決済保留エラー件数,
            7日以内解約件数, 7日以内解約率, 14日以内解約件数, 14日以内解約率

    停止件数：定期受注データ（df_teiki）の「ステータス＝停止」件数（定期回数=1のみ）。
    df_teiki・customer_masterが渡されない場合は算出せず、Noneで埋める（注記あり）。
    """
    notes = []
    df = df_orders_with_cohort.copy()

    required = [taiou_col, kessai_col, shipped_date_col, schedule_date_col, update_date_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        notes.append(f"❌ 必須列が見つかりません: {missing}。KPI②は計算できません。")
        return pd.DataFrame(), notes

    first_orders, fo_notes = get_valid_first_orders(df, teiki_kaisu_col, order_month_col)
    notes.extend(fo_notes)

    if first_orders.empty:
        return pd.DataFrame(), notes

    first_orders["_発送予定日_dt"] = pd.to_datetime(
        first_orders[schedule_date_col], errors="coerce", format="mixed"
    )
    first_orders["_更新日_dt"] = pd.to_datetime(
        first_orders[update_date_col], errors="coerce", format="mixed"
    )
    first_orders["_経過日数"] = (
        first_orders["_更新日_dt"] - first_orders["_発送予定日_dt"]
    ).dt.days

    # 停止件数（定期受注データから、定期回数=1×ステータス=停止を登録月別に集計）
    stopped_by_month = None
    if df_teiki is not None and not df_teiki.empty and customer_master is not None and not customer_master.empty:
        required_teiki = [customer_col, teiki_status_col, teiki_kaisu_col]
        missing_teiki = [c for c in required_teiki if c not in df_teiki.columns]
        if missing_teiki:
            notes.append(f"⚠️ 定期受注データに必須列がありません: {missing_teiki}。停止件数は計算できません。")
        else:
            teiki_first = df_teiki[df_teiki[teiki_kaisu_col] == 1].copy()
            teiki_first = teiki_first.merge(
                customer_master[[customer_col, "登録月"]], on=customer_col, how="left"
            )
            n_no_cohort = teiki_first["登録月"].isna().sum()
            if n_no_cohort > 0:
                notes.append(
                    f"⚠️ 定期受注データのうち{n_no_cohort}件は、登録月が特定できず"
                    f"（受注データに同一顧客番号が見つからない）停止件数の集計から除外しました。"
                )
            stopped = teiki_first[
                (teiki_first[teiki_status_col] == "停止") & teiki_first["登録月"].notna()
            ]
            stopped_by_month = stopped.groupby("登録月").size()
    else:
        notes.append(
            "ℹ️ 定期受注データがアップロードされていないため、「停止件数」は集計していません"
            "（受注データのみでは判定できない項目のため）。"
        )

    rows = []
    for month, sub in first_orders.groupby("登録月"):
        shipped_count = int(sub[shipped_date_col].notna().sum())
        cancelled = sub[sub[taiou_col] == "キャンセル"].copy()
        cancelled_count = len(cancelled)
        stopped_count = int(stopped_by_month.get(month, 0)) if stopped_by_month is not None else None
        payment_error_count = int(sub[kessai_col].isin(PAYMENT_ERROR_STATUSES).sum())

        n_missing_date = cancelled["_経過日数"].isna().sum()
        if n_missing_date > 0:
            notes.append(
                f"⚠️ 登録月{month}：キャンセル行のうち{n_missing_date}件は"
                f"「{schedule_date_col}」または「{update_date_col}」が日付として解釈できず、経過日数を計算できませんでした。"
            )

        valid_cancelled = cancelled.dropna(subset=["_経過日数"])
        within_7 = int((valid_cancelled["_経過日数"] <= 7).sum())
        within_14 = int((valid_cancelled["_経過日数"] <= 14).sum())
        denom_for_rate = len(valid_cancelled)

        rate_7 = within_7 / denom_for_rate if denom_for_rate > 0 else None
        rate_14 = within_14 / denom_for_rate if denom_for_rate > 0 else None

        rows.append({
            "登録月": month,
            "出荷件数": shipped_count,
            "キャンセル件数": cancelled_count,
            "停止件数": stopped_count,
            "決済保留エラー件数": payment_error_count,
            "7日以内解約件数": within_7,
            "7日以内解約率": rate_7,
            "14日以内解約件数": within_14,
            "14日以内解約率": rate_14,
        })

    result = pd.DataFrame(rows).sort_values("登録月").reset_index(drop=True)
    return result, notes


# ===== from modules/kpi03_f2.py =====
# -*- coding: utf-8 -*-
"""
③ F2転換率

定義：1回目購入者のうち、2回目の購入データが存在する割合。
継続率データCSV（全体／縛りあり／縛りなし）の「n~n+1回＝0～1」の行がそのままF2転換率に相当するため、
CSVの数値をそのまま使用する（再計算しない）。

- 確定のみ　　　＝「継続率 (%)」（売上済のみを反映）
- 未確定含む　　＝「継続予定率 (%)」（売上済＋売上前＋待機中を反映）
"""


def calculate_f2_conversion(df_continuation: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Args:
        df_continuation: continuation_data.load_all_continuation_segments の出力
                          （全体／縛りあり／縛りなしが「区分」列で区別された結合済みDataFrame）

    Returns:
        (result_df, notes)
        result_df columns:
            登録月, 区分, 初回購入者数, F2転換数（確定）, F2転換率（確定）, F2転換率（未確定含む）
    """
    notes = []

    if df_continuation.empty:
        notes.append("⚠️ 継続率データが読み込まれていません。")
        return pd.DataFrame(), notes

    f2 = df_continuation[df_continuation["n~n+1回"] == "0～1"].copy()

    if f2.empty:
        notes.append("⚠️ 「n~n+1回＝0～1」の行が見つかりませんでした。CSVの内容をご確認ください。")
        return pd.DataFrame(), notes

    result = f2[[
        "登録月", "区分", "合計 (件)", "売上済 (件)", "継続率 (%)", "継続予定率 (%)",
    ]].rename(columns={
        "合計 (件)": "初回購入者数",
        "売上済 (件)": "F2転換数（確定）",
        "継続率 (%)": "F2転換率（確定）",
        "継続予定率 (%)": "F2転換率（未確定含む）",
    })

    result = result.sort_values(["登録月", "区分"]).reset_index(drop=True)
    return result, notes


# ===== from modules/kpi04_continuation.py =====
# -*- coding: utf-8 -*-
"""
④ 定期継続率

定義：n回目→n+1回目への継続割合。
継続率データCSV（全体／縛りあり／縛りなし）の数値をそのまま使用する（再計算しない）。

- 確定のみ　　　＝「継続率 (%)」（売上済のみを反映）
- 未確定含む　　＝「継続予定率 (%)」（売上済＋売上前＋待機中を反映）
"""


def build_continuation_table(df_continuation: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Args:
        df_continuation: continuation_data.load_all_continuation_segments の出力

    Returns:
        (result_df, notes)
        result_df columns:
            登録月, 区分, n~n+1回, 合計(件), 売上済(件), 売上前(件), 待機中(件), 離脱(件),
            継続率（確定）, 継続率（未確定含む）
    """
    notes = []

    if df_continuation.empty:
        notes.append("⚠️ 継続率データが読み込まれていません。")
        return pd.DataFrame(), notes

    result = df_continuation[[
        "登録月", "区分", "n~n+1回", "合計 (件)", "売上済 (件)", "売上前 (件)",
        "待機中 (件)", "離脱 (件)", "継続率 (%)", "継続予定率 (%)",
    ]].rename(columns={
        "合計 (件)": "合計（件）",
        "売上済 (件)": "売上済（件）",
        "売上前 (件)": "売上前（件）",
        "待機中 (件)": "待機中（件）",
        "離脱 (件)": "離脱（件）",
        "継続率 (%)": "継続率（確定）",
        "継続予定率 (%)": "継続率（未確定含む）",
    })

    result = result.sort_values(["登録月", "区分", "n~n+1回"]).reset_index(drop=True)
    return result, notes


# ===== from modules/kpi05_customer_value.py =====
# -*- coding: utf-8 -*-
"""
⑤ LTV（平均顧客生涯価値）／⑥ 平均購入回数／⑦ アップセル率

【共通の分母（ユーザー確定版）】
登録月ごとに、KPI①と同じ「初回注文（定期回数=1）のスコア≥1」の顧客のみを対象とする
（初回注文が完全失敗＝スコア0の顧客は、そもそも実質的な購入者ではないため分母から除外）。

⑤ LTV
- 分子：対象顧客の全注文における「合計（計算用）」（スコア=2＝完全購入のみ計上）の合計
- 分母：登録月の対象顧客数
- LTV = 分子 ÷ 分母

⑥ 平均購入回数
- 分子：対象顧客ごとの MAX(定期回数) の合計
- 分母：⑤と同じ
- 平均購入回数 = 分子 ÷ 分母

⑦ アップセル率（ユーザー確定版：お約束回数（定期）＞0 を基準とする）
- SKUコードの静的リストではなく、「お約束回数（定期）」列を直接参照する
  （今後SKUが追加されても自動的に対応でき、F2転換率の縛りあり/なし判定とも一貫性が取れるため）
- 分子：対象顧客のうち、いずれかの注文で「お約束回数（定期）＞0」だった顧客数
- 分母：⑤と同じ
- アップセル率 = 分子 ÷ 分母
"""



def calculate_ltv_purchase_upsell(
    df_orders_with_cohort: pd.DataFrame,
    customer_col: str = "顧客番号",
    teiki_kaisu_col: str = "定期回数",
    order_month_col: str = "受注月",
    promise_count_col: str = "お約束回数（定期）",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns:
        (result_df, notes)
        result_df columns:
            登録月, 対象顧客数, LTV, 平均購入回数, アップセル顧客数, アップセル率
    """
    notes = []

    if promise_count_col not in df_orders_with_cohort.columns:
        notes.append(f"❌ 「{promise_count_col}」列が見つかりません。⑦アップセル率は計算できません。")

    df = compute_amount_for_kpi(df_orders_with_cohort)
    notes.extend(df.attrs.get("quality_notes", []))

    # STEP1: 分母となる対象顧客（登録月ごと・初回注文スコア≥1）を確定
    first_orders, fo_notes = get_valid_first_orders(df, teiki_kaisu_col, order_month_col)
    notes.extend(fo_notes)

    if first_orders.empty:
        return pd.DataFrame(), notes

    target_customers = first_orders[first_orders["スコア"] >= 1][[customer_col, "登録月"]].drop_duplicates()

    if target_customers.empty:
        notes.append("⚠️ 対象顧客（初回注文スコア≥1）が0件でした。")
        return pd.DataFrame(), notes

    n_dup_rows = len(first_orders[first_orders["スコア"] >= 1]) - len(target_customers)
    if n_dup_rows > 0:
        notes.append(
            f"ℹ️ 同一顧客が複数の「定期回数=1」注文を持つケースが{n_dup_rows}件ありました"
            f"（二重送信等の可能性）。⑤⑥⑦は顧客単位で集計するため重複排除済みです"
            f"（①の注文単位カウントとは母数が異なる場合があります）。"
        )

    # STEP2: 対象顧客の全注文から、顧客単位の集計値を作る
    target_ids = set(target_customers[customer_col])
    df_target_orders = df[df[customer_col].isin(target_ids)].copy()

    agg_dict = {
        "合計（計算用）_SUM": (df_target_orders.groupby(customer_col)["合計（計算用）"].sum()),
        "定期回数_MAX": (df_target_orders.groupby(customer_col)[teiki_kaisu_col].max()),
    }
    if promise_count_col in df_target_orders.columns:
        promise_max = df_target_orders.groupby(customer_col)[promise_count_col].max()
        agg_dict["アップセルフラグ"] = (promise_max > 0)
    else:
        agg_dict["アップセルフラグ"] = pd.Series(dtype=bool)

    customer_summary = pd.DataFrame(agg_dict).reset_index().rename(columns={"index": customer_col})
    customer_summary = customer_summary.merge(target_customers, on=customer_col, how="right")

    # STEP3: 登録月ごとに集計
    rows = []
    for month, sub in customer_summary.groupby("登録月"):
        n = len(sub)
        ltv = sub["合計（計算用）_SUM"].sum() / n if n > 0 else None
        avg_purchase = sub["定期回数_MAX"].sum() / n if n > 0 else None
        upsell_count = int(sub["アップセルフラグ"].fillna(False).sum())
        upsell_rate = upsell_count / n if n > 0 else None

        rows.append({
            "登録月": month,
            "対象顧客数": n,
            "LTV": ltv,
            "平均購入回数": avg_purchase,
            "アップセル顧客数": upsell_count,
            "アップセル率": upsell_rate,
        })

    result = pd.DataFrame(rows).sort_values("登録月").reset_index(drop=True)
    return result, notes


# ===== from modules/kpi08_customer_trend.py =====
# -*- coding: utf-8 -*-
"""
⑧ 顧客推移データ

定義：月ごとに全顧客を「アクティブ/非アクティブ」×「初回購入/2回目購入/継続/優良」に分類し、
人数と構成比を算出する。

【アクティブ/非アクティブの判定（ユーザー確定版）】
定期受注データ（sales_teiki）の各定期契約について、対象月末時点で
- 作成日 <= 対象月末
- （停止日が空 or 対象月末より後）かつ（キャンセル日が空 or 対象月末より後）
を満たす契約を1件以上持つ顧客を「アクティブ」、それ以外を「非アクティブ」とする。
定期受注データに一度も登場しない顧客は、判定不能のため注記の上「非アクティブ」として扱う。

【属性区分の判定（ユーザー確定版）】
受注データ（sales_data）から、対象月末時点までの受注（受注日<=対象月末）に絞り、
顧客ごとの MAX(定期回数) で分類：
- 初回購入顧客：MAX = 1
- 2回目購入顧客：MAX = 2
- 継続顧客：MAX = 3
- 優良顧客：MAX >= 4
"""

ATTRIBUTE_ORDER = ["初回購入顧客", "2回目購入顧客", "継続顧客", "優良顧客"]
ACTIVE_ORDER = ["アクティブ", "非アクティブ"]


def _classify_attribute(max_teiki_kaisu) -> str:
    if pd.isna(max_teiki_kaisu):
        return None
    if max_teiki_kaisu == 1:
        return "初回購入顧客"
    elif max_teiki_kaisu == 2:
        return "2回目購入顧客"
    elif max_teiki_kaisu == 3:
        return "継続顧客"
    else:
        return "優良顧客"


def _is_active_asof(sub_teiki: pd.DataFrame, month_end: pd.Timestamp) -> bool:
    created_ok = sub_teiki["_作成日_dt"] <= month_end
    not_stopped = sub_teiki["_停止日_dt"].isna() | (sub_teiki["_停止日_dt"] > month_end)
    not_cancelled = sub_teiki["_キャンセル日_dt"].isna() | (sub_teiki["_キャンセル日_dt"] > month_end)
    return bool((created_ok & not_stopped & not_cancelled).any())


def build_customer_trend(
    df_orders: pd.DataFrame,
    df_teiki: pd.DataFrame,
    target_months: list,
    customer_col: str = "顧客番号",
    teiki_kaisu_col: str = "定期回数",
    order_date_col: str = "受注日",
    teiki_created_col: str = "作成日",
    teiki_stop_col: str = "停止日",
    teiki_cancel_col: str = "キャンセル日",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Args:
        df_orders: 受注データ（顧客番号・定期回数・受注日を含む）
        df_teiki: 定期受注データ（顧客番号・作成日・停止日・キャンセル日を含む）
        target_months: 対象月のリスト（例: ["2026-05", "2026-06"]）

    Returns:
        (result_df, notes)
        result_df columns: [対象月, 状態, 属性, 人数, 構成比]
    """
    notes = []

    required_order_cols = [customer_col, teiki_kaisu_col, order_date_col]
    missing_order = [c for c in required_order_cols if c not in df_orders.columns]
    if missing_order:
        notes.append(f"❌ 受注データに必須列がありません: {missing_order}")
        return pd.DataFrame(), notes

    required_teiki_cols = [customer_col, teiki_created_col]
    missing_teiki = [c for c in required_teiki_cols if c not in df_teiki.columns]
    if missing_teiki:
        notes.append(f"❌ 定期受注データに必須列がありません: {missing_teiki}")
        return pd.DataFrame(), notes

    orders = df_orders.copy()
    orders["_受注日_dt"] = pd.to_datetime(orders[order_date_col], errors="coerce", format="mixed")

    teiki = df_teiki.copy()
    teiki["_作成日_dt"] = pd.to_datetime(teiki[teiki_created_col], errors="coerce", format="mixed")
    teiki["_停止日_dt"] = (
        pd.to_datetime(teiki[teiki_stop_col], errors="coerce", format="mixed")
        if teiki_stop_col in teiki.columns else pd.NaT
    )
    teiki["_キャンセル日_dt"] = (
        pd.to_datetime(teiki[teiki_cancel_col], errors="coerce", format="mixed")
        if teiki_cancel_col in teiki.columns else pd.NaT
    )

    all_customers = pd.Index(orders[customer_col].dropna().unique(), name=customer_col)

    customers_without_teiki = set(all_customers) - set(teiki[customer_col].dropna().unique())
    if customers_without_teiki:
        notes.append(
            f"⚠️ 受注データにはあるが定期受注データに存在しない顧客が{len(customers_without_teiki)}件あります。"
            f"定期受注データは「作成日」がその期間内の契約のみを含むため、"
            f"それより前に定期契約を開始した顧客（2回目以降の購入者）は、その契約作成月のデータが"
            f"アップロードされていないと判定できません（現状「非アクティブ」として扱っています）。"
            f"複数月分の定期受注データを蓄積アップロードすると精度が上がります。"
        )

    teiki_by_customer = {cust: sub for cust, sub in teiki.groupby(customer_col)}

    rows = []
    for month in target_months:
        month_end = pd.Period(month, freq="M").end_time

        # 属性区分：対象月末までの受注に基づくMAX定期回数
        orders_asof = orders[orders["_受注日_dt"] <= month_end]
        max_kaisu = orders_asof.groupby(customer_col)[teiki_kaisu_col].max()

        month_customers = pd.DataFrame(index=all_customers)
        month_customers["定期回数_MAX"] = max_kaisu
        # その月末までに1件も受注がない顧客は対象外（まだ登録前）
        month_customers = month_customers.dropna(subset=["定期回数_MAX"])
        if month_customers.empty:
            continue

        month_customers["属性"] = month_customers["定期回数_MAX"].apply(_classify_attribute)

        active_flags = {}
        for cust in month_customers.index:
            sub = teiki_by_customer.get(cust)
            active_flags[cust] = _is_active_asof(sub, month_end) if sub is not None else False
        month_customers["状態"] = month_customers.index.map(
            lambda c: "アクティブ" if active_flags.get(c, False) else "非アクティブ"
        )

        total = len(month_customers)
        grouped = month_customers.groupby(["状態", "属性"]).size().reset_index(name="人数")

        for state in ACTIVE_ORDER:
            for attr in ATTRIBUTE_ORDER:
                match = grouped[(grouped["状態"] == state) & (grouped["属性"] == attr)]
                n = int(match["人数"].iloc[0]) if not match.empty else 0
                rows.append({
                    "対象月": month,
                    "状態": state,
                    "属性": attr,
                    "人数": n,
                    "構成比": n / total if total > 0 else None,
                })

    result = pd.DataFrame(rows)
    return result, notes


# ===== from modules/kpi09_active_customers.py =====
# -*- coding: utf-8 -*-
"""
⑨ 稼働顧客数

定義：一定期間内に活動がある顧客の割合。
稼働顧客の条件（KPI分析_完全引き継ぎプロンプト v3準拠）：
「3ヶ月以内購入者 ＋ 当月購入者 ＋ 翌月発送予定者」の合算
＝ 次回発送予定日が [基準日, 基準日+2ヶ月] の範囲内　かつ　最終購入日が [基準日-3ヶ月, 基準日] の範囲内

- 対象：合計（計算用）＞0（＝スコア2、完全購入）の実績を持つ顧客のみ（全体母数）
- 顧客単位で判定：次回発送予定日は「その顧客の完全購入注文の中で最も新しいもの」、
  最終購入日は顧客マスタの最終購入日を使用
- 稼働率 = 稼働顧客数 ÷ 全体顧客数（全コホート合算がメイン数値。登録月別は参考情報）
"""



def calculate_active_customers(
    df_orders_with_cohort: pd.DataFrame,
    customer_master: pd.DataFrame,
    asof_date: pd.Timestamp,
    customer_col: str = "顧客番号",
    next_ship_col: str = "次回発送予定日",
) -> tuple[dict, pd.DataFrame, list[str]]:
    """
    Args:
        df_orders_with_cohort: 受注データ（登録月を含む）
        customer_master: customer_master.build_customer_master の出力（最終購入日を含む）
        asof_date: 基準日

    Returns:
        (overall_summary, by_cohort_df, notes)
        overall_summary: {"全体顧客数":, "稼働顧客数":, "稼働率":}
        by_cohort_df columns: [登録月, 全体顧客数, 稼働顧客数, 稼働率]
    """
    notes = []

    if next_ship_col not in df_orders_with_cohort.columns:
        notes.append(f"❌ 「{next_ship_col}」列が見つかりません。⑨は計算できません。")
        return {}, pd.DataFrame(), notes

    df = compute_amount_for_kpi(df_orders_with_cohort)
    notes.extend(df.attrs.get("quality_notes", []))

    valid = df[df["合計（計算用）"] > 0].copy()  # スコア2＝完全購入のみ
    if valid.empty:
        notes.append("⚠️ 完全購入（スコア=2）の注文が見つかりませんでした。")
        return {}, pd.DataFrame(), notes

    valid["_次回発送予定日_dt"] = pd.to_datetime(valid[next_ship_col], errors="coerce", format="mixed")

    # 顧客単位に集約：次回発送予定日は最新のものを採用
    ship_summary = valid.groupby(customer_col).agg(
        次回発送予定日=("_次回発送予定日_dt", "max"),
        登録月=("登録月", "first"),
    ).reset_index()

    # 最終購入日は顧客マスタ（全注文ベース）から結合
    customer_summary = ship_summary.merge(
        customer_master[[customer_col, "最終購入日"]], on=customer_col, how="left"
    )
    customer_summary["最終購入日"] = pd.to_datetime(customer_summary["最終購入日"], errors="coerce")

    delivery_low = asof_date
    delivery_high = asof_date + pd.DateOffset(months=2)
    three_months_ago = asof_date - pd.DateOffset(months=3)

    cond = (
        (customer_summary["次回発送予定日"] >= delivery_low)
        & (customer_summary["次回発送予定日"] <= delivery_high)
        & (customer_summary["最終購入日"] >= three_months_ago)
        & (customer_summary["最終購入日"] <= asof_date)
    )

    total_all = len(customer_summary)
    active_all = int(cond.sum())
    overall_summary = {
        "全体顧客数": total_all,
        "稼働顧客数": active_all,
        "稼働率": active_all / total_all if total_all > 0 else None,
        "基準日": asof_date.strftime("%Y-%m-%d"),
        "対象期間（発送予定日）": f"{delivery_low.strftime('%Y-%m-%d')} 〜 {delivery_high.strftime('%Y-%m-%d')}",
        "対象期間（最終購入日）": f"{three_months_ago.strftime('%Y-%m-%d')} 〜 {asof_date.strftime('%Y-%m-%d')}",
    }

    rows = []
    for month, sub in customer_summary.groupby("登録月"):
        n = len(sub)
        sub_cond = (
            (sub["次回発送予定日"] >= delivery_low)
            & (sub["次回発送予定日"] <= delivery_high)
            & (sub["最終購入日"] >= three_months_ago)
            & (sub["最終購入日"] <= asof_date)
        )
        active_n = int(sub_cond.sum())
        rows.append({
            "登録月": month,
            "全体顧客数": n,
            "稼働顧客数": active_n,
            "稼働率": active_n / n if n > 0 else None,
        })

    by_cohort_df = pd.DataFrame(rows).sort_values("登録月").reset_index(drop=True)

    return overall_summary, by_cohort_df, notes


# ===== from modules/kpi10_rf.py =====
# -*- coding: utf-8 -*-
"""
⑩ RF分析

定義：顧客を以下の2軸でクロス集計し、マトリクス表を作成。
- Recency（最終購入日からの経過日数）：0-30日、31-60日、61-90日、91-120日…
- Frequency（購入回数）：1回、2回、3回…

対象：受注データに登場した全顧客（顧客マスタの最終購入日を使用）。
基準日は画面上で選択可能（ユーザー確定版）。
"""

RECENCY_BINS = [-1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 10**6]
RECENCY_LABELS = [
    "0-30", "31-60", "61-90", "91-120", "121-150", "151-180",
    "181-210", "211-240", "241-270", "271-300", "301-330", "331以上",
]


def calculate_rf_matrix(
    customer_master: pd.DataFrame,
    df_orders: pd.DataFrame,
    asof_date: pd.Timestamp,
    customer_col: str = "顧客番号",
    teiki_kaisu_col: str = "定期回数",
    max_frequency_display: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns:
        (result_df, notes)
        result_df: index=Recency区分, columns=Frequency(1,2,3,...), values=人数（クロス集計表）
    """
    notes = []

    if "最終購入日" not in customer_master.columns:
        notes.append("❌ 顧客マスタに「最終購入日」列がありません。")
        return pd.DataFrame(), notes

    freq = df_orders.groupby(customer_col)[teiki_kaisu_col].max().rename("Frequency")
    master = customer_master.set_index(customer_col).join(freq, how="left")

    master["最終購入日"] = pd.to_datetime(master["最終購入日"], errors="coerce")
    master["Recency日数"] = (asof_date - master["最終購入日"]).dt.days

    n_negative = (master["Recency日数"] < 0).sum()
    if n_negative > 0:
        notes.append(
            f"⚠️ 最終購入日が基準日より後の顧客が{n_negative}件あります"
            f"（基準日を最新受注日より前に設定した可能性）。0-30日区分に含めています。"
        )

    master["Recency区分"] = pd.cut(
        master["Recency日数"], bins=RECENCY_BINS, labels=RECENCY_LABELS
    )

    master["Frequency表示"] = master["Frequency"].apply(
        lambda x: str(int(x)) if pd.notna(x) and x < max_frequency_display else f"{max_frequency_display}以上"
    )

    matrix = pd.crosstab(master["Recency区分"], master["Frequency表示"])

    # 列を数値順に並び替え
    def _sort_key(col):
        return int(col.replace("以上", "")) if col.replace("以上", "").isdigit() else 999
    matrix = matrix[sorted(matrix.columns, key=_sort_key)]
    matrix = matrix.reindex(RECENCY_LABELS)
    matrix["総計"] = matrix.sum(axis=1)
    matrix.loc["総計"] = matrix.sum(axis=0)

    return matrix, notes


# ===== from modules/kpi11_cancel_reason.py =====
# -*- coding: utf-8 -*-
"""
⑪ 解約理由

定義：定期受注データの「キャンセル理由」テキストを分類集計。
対象：ステータス＝キャンセル の行。
（「停止理由」列は実データ上ほぼ使われていないため、キャンセル理由を主に使用。
　停止理由が入っている場合はキャンセル理由が空の行に限り補完的に採用する）

理由が未記載の行は「理由未記載」として別枠集計する（勝手に推測しない）。
"""


def calculate_cancel_reasons(
    df_teiki: pd.DataFrame,
    status_col: str = "ステータス",
    reason_col: str = "キャンセル理由",
    fallback_reason_col: str = "停止理由",
    target_status: str = "キャンセル",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns:
        (result_df, notes)
        result_df columns: [理由, 件数, 構成比]
    """
    notes = []

    if status_col not in df_teiki.columns or reason_col not in df_teiki.columns:
        notes.append(f"❌ 必須列（{status_col} / {reason_col}）が見つかりません。")
        return pd.DataFrame(), notes

    cancelled = df_teiki[df_teiki[status_col] == target_status].copy()

    if cancelled.empty:
        notes.append(f"⚠️ ステータス＝{target_status} の行が見つかりませんでした。")
        return pd.DataFrame(), notes

    reason = cancelled[reason_col]
    if fallback_reason_col in cancelled.columns:
        reason = reason.fillna(cancelled[fallback_reason_col])

    reason = reason.fillna("理由未記載")
    reason = reason.astype(str).str.strip()
    reason = reason.replace("", "理由未記載")

    counts = reason.value_counts().reset_index()
    counts.columns = ["理由", "件数"]
    total = counts["件数"].sum()
    counts["構成比"] = counts["件数"] / total

    return counts, notes


# ===== from modules/kpi12_scoring.py =====
# -*- coding: utf-8 -*-
"""
⑫ スコアリング（顧客分類）

定義：全顧客を購入回数（MAX定期回数）で3区分に分類し、人数・累積売上・平均LTV・売上構成比を可視化。
- 初回購入顧客：MAX(定期回数) = 1
- 2回目購入顧客：MAX(定期回数) = 2
- 継続顧客　　：MAX(定期回数) >= 3

対象：受注データに一度でも登場した全顧客（登録月・スコアによる絞り込みはしない。
「これまでに買ったことがある全顧客を、購入回数で分類する」という定義のため）。
LTV（累積売上）は「合計（計算用）」（スコア=2＝完全購入のみ計上）の顧客ごとの合計を使用。

※Tier1〜6の数値ランクは、旧資料にも定義が存在しないため今回は実装しない（ユーザー確認済み）。
"""


CATEGORY_ORDER = ["初回購入顧客", "2回目購入顧客", "継続顧客"]


def _classify(max_teiki_kaisu: int) -> str:
    if max_teiki_kaisu == 1:
        return "初回購入顧客"
    elif max_teiki_kaisu == 2:
        return "2回目購入顧客"
    else:
        return "継続顧客"


def calculate_customer_scoring(
    df_orders: pd.DataFrame,
    customer_col: str = "顧客番号",
    teiki_kaisu_col: str = "定期回数",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns:
        (result_df, notes)
        result_df columns: [顧客分類, 人数, 人数構成比, 累積売上, 平均LTV, 売上構成比]
    """
    notes = []
    df = compute_amount_for_kpi(df_orders)
    notes.extend(df.attrs.get("quality_notes", []))

    customer_summary = df.groupby(customer_col).agg(
        LTV=("合計（計算用）", "sum"),
        定期回数_MAX=(teiki_kaisu_col, "max"),
    ).reset_index()

    customer_summary["顧客分類"] = customer_summary["定期回数_MAX"].apply(_classify)

    total_customers = len(customer_summary)
    total_ltv = customer_summary["LTV"].sum()

    rows = []
    for cat in CATEGORY_ORDER:
        sub = customer_summary[customer_summary["顧客分類"] == cat]
        n = len(sub)
        ltv_sum = sub["LTV"].sum()
        rows.append({
            "顧客分類": cat,
            "人数": n,
            "人数構成比": n / total_customers if total_customers > 0 else None,
            "累積売上": ltv_sum,
            "平均LTV": ltv_sum / n if n > 0 else None,
            "売上構成比": ltv_sum / total_ltv if total_ltv > 0 else None,
        })

    result = pd.DataFrame(rows)
    return result, notes


# ======================================================================
# アプリ本体（画面表示）※一括アップロード＋タブ切替方式
# ======================================================================
st.set_page_config(page_title="KPIレポート作成システム", layout="wide")

REQUIRED_ORDER_COLUMNS = ["顧客番号", "受注日", "定期回数", "対応状況", "決済状況", "合計"]


# --------------------------------------------------------------------
# パスワードゲート（合言葉を知っている人だけアクセス可能にする）
# --------------------------------------------------------------------
def check_password() -> bool:
    """
    合言葉方式の簡易認証。
    合言葉は .streamlit/secrets.toml の app_password に設定する（Streamlit Community Cloudでは
    ダッシュボードの Secrets 設定画面から登録すればコードやGitHubに書く必要はない）。
    設定ファイルが存在しない場合（ローカルでの動作確認時など）は認証をスキップする。
    """
    try:
        correct_password = st.secrets["app_password"]
    except Exception:
        # secrets.tomlが無い＝ローカル動作確認用。認証なしで通す。
        return True

    if st.session_state.get("password_correct", False):
        return True

    st.title("📊 KPIレポート作成システム")
    st.text_input("合言葉を入力してください", type="password", key="password_input")
    if st.button("入室する"):
        if st.session_state.get("password_input", "") == correct_password:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("合言葉が違います。")
    return False


if not check_password():
    st.stop()

st.title("📊 KPIレポート作成システム")
st.caption("左のサイドバーで必要なデータをまとめてアップロードしてください。アップロード後、上部のタブでKPIを自由に切り替えられます。")

with st.sidebar:
    st.header("① 受注データ")
    st.caption("①②⑤⑥⑦⑧⑨⑩⑫で使用（複数月分まとめてOK）")
    order_files = st.file_uploader(
        "受注データCSV（複数選択可）", type=["csv"], accept_multiple_files=True, key="order_files",
    )

    st.header("② 定期受注データ")
    st.caption("⑧⑪で使用（複数月分まとめてOK）")
    teiki_files = st.file_uploader(
        "定期受注データCSV（複数選択可）", type=["csv"], accept_multiple_files=True, key="teiki_files",
    )

    st.header("③ 定期継続率データ")
    st.caption("③④で使用（3種類とも必要）")
    continuation_files_all = st.file_uploader(
        "継続率データ（全体）", type=["csv"], accept_multiple_files=True, key="cont_all"
    )
    continuation_files_ari = st.file_uploader(
        "継続率データ（定期縛りあり）", type=["csv"], accept_multiple_files=True, key="cont_ari"
    )
    continuation_files_nashi = st.file_uploader(
        "継続率データ（定期縛りなし）", type=["csv"], accept_multiple_files=True, key="cont_nashi"
    )

    st.divider()
    st.header("基準日（⑨⑩で使用）")
    asof_date = st.date_input("基準日", value=datetime.date.today(), key="asof_date")


def render_notes(notes: list[str]):
    if not notes:
        return
    with st.expander("⚠️ 注記・データ品質チェック", expanded=True):
        for n in notes:
            st.write(n)


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=(df.index.name is not None))
    return buf.getvalue()


def download_buttons(result: pd.DataFrame, file_stub: str, sheet_name: str):
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 CSVダウンロード",
            data=result.to_csv(index=(result.index.name is not None)).encode("utf-8-sig"),
            file_name=f"{file_stub}.csv", mime="text/csv", key=f"csv_{file_stub}",
        )
    with col2:
        st.download_button(
            "📥 Excelダウンロード",
            data=df_to_excel_bytes(result, sheet_name),
            file_name=f"{file_stub}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"xlsx_{file_stub}",
        )


@st.cache_data(show_spinner="受注データを処理しています…")
def prepare_orders(files):
    notes = []
    df_orders, load_notes = load_multiple_csv(files, required_columns=REQUIRED_ORDER_COLUMNS, dedup_key="受注ID")
    notes.extend(load_notes)
    if df_orders.empty:
        return df_orders, pd.DataFrame(), notes
    df_orders, month_notes = add_order_month(df_orders)
    notes.extend(month_notes)
    master, master_notes = build_customer_master(df_orders)
    notes.extend(master_notes)
    df_with_cohort = attach_registration_month(df_orders, master)
    return df_with_cohort, master, notes


@st.cache_data(show_spinner="定期受注データを処理しています…")
def prepare_teiki(files):
    return load_teiki_data(files)


@st.cache_data(show_spinner="継続率データを処理しています…")
def prepare_continuation(files_all, files_ari, files_nashi):
    return load_all_continuation_segments(files_all or [], files_ari or [], files_nashi or [])


# --------------------------------------------------------------------
# データ読み込み（アップロードされているものだけ処理する）
# 新規アップロードがあればサーバー側に保存し、次回以降アップロードなしでも
# 最後に保存されたデータを自動で表示する（＝誰かが更新すれば全員に反映される）
# --------------------------------------------------------------------
LATEST_DIR = Path(__file__).resolve().parent / "latest_data"
LATEST_DIR.mkdir(exist_ok=True)

PATH_ORDER_COHORT = LATEST_DIR / "order_with_cohort.pkl"
PATH_ORDER_MASTER = LATEST_DIR / "order_master.pkl"
PATH_TEIKI = LATEST_DIR / "teiki.pkl"
PATH_CONTINUATION = LATEST_DIR / "continuation.pkl"


def _saved_at(path: Path) -> str:
    if not path.exists():
        return ""
    return datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


df_with_cohort, master, order_notes = (pd.DataFrame(), pd.DataFrame(), [])
if order_files:
    df_with_cohort, master, order_notes = prepare_orders(order_files)
    if not df_with_cohort.empty:
        df_with_cohort.to_pickle(PATH_ORDER_COHORT)
        master.to_pickle(PATH_ORDER_MASTER)
elif PATH_ORDER_COHORT.exists():
    df_with_cohort = pd.read_pickle(PATH_ORDER_COHORT)
    master = pd.read_pickle(PATH_ORDER_MASTER)
    st.sidebar.caption(f"📌 受注データ：保存済み（{_saved_at(PATH_ORDER_COHORT)}時点）を表示中")

df_teiki, teiki_notes = (pd.DataFrame(), [])
if teiki_files:
    df_teiki, teiki_notes = prepare_teiki(teiki_files)
    if not df_teiki.empty:
        df_teiki.to_pickle(PATH_TEIKI)
elif PATH_TEIKI.exists():
    df_teiki = pd.read_pickle(PATH_TEIKI)
    st.sidebar.caption(f"📌 定期受注データ：保存済み（{_saved_at(PATH_TEIKI)}時点）を表示中")

df_continuation, cont_notes = (pd.DataFrame(), [])
if continuation_files_all or continuation_files_ari or continuation_files_nashi:
    df_continuation, cont_notes = prepare_continuation(
        continuation_files_all, continuation_files_ari, continuation_files_nashi
    )
    if not df_continuation.empty:
        df_continuation.to_pickle(PATH_CONTINUATION)
elif PATH_CONTINUATION.exists():
    df_continuation = pd.read_pickle(PATH_CONTINUATION)
    st.sidebar.caption(f"📌 継続率データ：保存済み（{_saved_at(PATH_CONTINUATION)}時点）を表示中")

if df_with_cohort.empty and df_teiki.empty and df_continuation.empty:
    st.info("👈 左のサイドバーから、まずは受注データ／定期受注データ／継続率データをアップロードしてください。")
    st.stop()

# データサマリー表示
summary_cols = st.columns(3)
with summary_cols[0]:
    if not df_with_cohort.empty:
        st.success(f"✅ 受注データ：{len(df_with_cohort):,}行（{df_with_cohort['顧客番号'].nunique():,}顧客）")
        month_summary = summarize_order_months(df_with_cohort)
        if not month_summary.empty:
            st.caption(f"受注月範囲：{month_summary['受注月'].min()} 〜 {month_summary['受注月'].max()}")
    else:
        st.warning("⬜ 受注データ：未アップロード")
with summary_cols[1]:
    if not df_teiki.empty:
        st.success(f"✅ 定期受注データ：{len(df_teiki):,}行")
    else:
        st.warning("⬜ 定期受注データ：未アップロード")
with summary_cols[2]:
    if not df_continuation.empty:
        st.success(f"✅ 継続率データ：{len(df_continuation):,}行")
    else:
        st.warning("⬜ 継続率データ：未アップロード")

all_prep_notes = order_notes + teiki_notes + cont_notes
render_notes(all_prep_notes)

st.divider()

# --------------------------------------------------------------------
# タブでKPIを切り替え表示
# --------------------------------------------------------------------
kpi_results = {}  # 保存機能用：各タブで計算した結果をここに集約する

tab_labels = [
    "①初回離脱率", "②期間別解約率", "③F2転換率", "④定期継続率",
    "⑤⑥⑦LTV等", "⑧顧客推移", "⑨稼働顧客数", "⑩RF分析", "⑪解約理由", "⑫スコアリング",
]
tabs = st.tabs(tab_labels)

# ① 初回離脱率
with tabs[0]:
    st.header("① 初回離脱率")
    st.markdown(
        """
        **定義**：登録月ごとの初回購入者のうち、与信NG等のシステム的キャンセルを除いた
        「1回目購入後に離脱した顧客」の割合。
        - スコア = 対応状況スコア（0/1）＋ 決済状況スコア（0/1）
        - 初回購入者（分母）＝ スコア ≥ 1　／　離脱者（分子）＝ スコア ≤ 1（スコア=1は両方に計上）
        """
    )
    if df_with_cohort.empty:
        st.info("受注データをアップロードしてください。")
    else:
        result, kpi_notes = calculate_first_time_churn(df_with_cohort)
        kpi_results["01_初回離脱率"] = result
        render_notes(kpi_notes)
        if not result.empty:
            display_df = result.copy()
            display_df["初回離脱率"] = display_df["初回離脱率"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "―")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            fig = px.bar(
                result, x="登録月", y="初回離脱率",
                text=result["初回離脱率"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else ""),
                title="登録月別 初回離脱率",
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            download_buttons(result, "kpi01_初回離脱率", "初回離脱率")
            with st.expander("🔍 顧客マスタ（登録月）を確認する"):
                st.dataframe(master, use_container_width=True, hide_index=True)

# ② 期間別解約率
with tabs[1]:
    st.header("② 期間別解約率")
    st.markdown(
        """
        **定義**：定期回数=1（真の初回購入）を対象に、出荷・キャンセル・停止・決済保留エラーの件数と、
        キャンセルが「出荷予定日から何日以内に発生したか」を集計。
        - **出荷件数**：「発送日」に実データが入っている件数
        - **キャンセル件数**：受注データの「対応状況」＝キャンセル の件数
        - **停止件数**：定期受注データの「ステータス」＝停止 の件数（要：定期受注データのアップロード）
        - **決済保留エラー件数**：与信保留／与信審査エラー／仮売上失敗／取引修正失敗（別枠並列集計）
        - **期間別解約**：「発送予定日」を出荷日とみなし、「更新日」との差分日数で7日/14日以内を集計
        """
    )
    if df_with_cohort.empty:
        st.info("受注データをアップロードしてください。")
    else:
        if df_teiki.empty:
            st.caption("ℹ️ 定期受注データもアップロードすると「停止件数」も集計されます。")
        result, kpi_notes = calculate_period_cancellation(df_with_cohort, df_teiki, master)
        kpi_results["02_期間別解約率"] = result
        render_notes(kpi_notes)
        if not result.empty:
            display_df = result.copy()
            for c in ["7日以内解約率", "14日以内解約率"]:
                display_df[c] = display_df[c].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "―")
            if display_df["停止件数"].isna().all():
                display_df["停止件数"] = "―"
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            chart_cols = ["出荷件数", "キャンセル件数", "決済保留エラー件数"]
            if result["停止件数"].notna().any():
                chart_cols.insert(2, "停止件数")
            fig = px.bar(
                result, x="登録月", y=chart_cols,
                barmode="group", title="登録月別 出荷・キャンセル・停止・決済保留エラー件数",
            )
            st.plotly_chart(fig, use_container_width=True)
            download_buttons(result, "kpi02_期間別解約率", "期間別解約率")

# ③ F2転換率
with tabs[2]:
    st.header("③ F2転換率")
    st.markdown(
        """
        **定義**：1回目購入者のうち、2回目の購入データが存在する割合。
        継続率データCSVの「n~n+1回＝0～1」の行がそのままF2転換率に相当するため、**CSVの数値をそのまま使用**。
        """
    )
    if df_continuation.empty:
        st.info("継続率データ（全体／縛りあり／縛りなし）をアップロードしてください。")
    else:
        result, kpi_notes = calculate_f2_conversion(df_continuation)
        kpi_results["03_F2転換率"] = result
        render_notes(kpi_notes)
        if not result.empty:
            tab_kakutei, tab_mikakutei = st.tabs(["売上確定", "売上未確定含む"])
            with tab_kakutei:
                d = result[["登録月", "区分", "初回購入者数", "F2転換数（確定）", "F2転換率（確定）"]].copy()
                d["F2転換率（確定）"] = d["F2転換率（確定）"].apply(lambda x: f"{x:.2f}%")
                st.dataframe(d, use_container_width=True, hide_index=True)
                fig = px.bar(result, x="登録月", y="F2転換率（確定）", color="区分", barmode="group",
                             title="登録月別 F2転換率（売上確定）")
                st.plotly_chart(fig, use_container_width=True)
            with tab_mikakutei:
                d = result[["登録月", "区分", "初回購入者数", "F2転換率（未確定含む）"]].copy()
                d["F2転換率（未確定含む）"] = d["F2転換率（未確定含む）"].apply(lambda x: f"{x:.2f}%")
                st.dataframe(d, use_container_width=True, hide_index=True)
                fig = px.bar(result, x="登録月", y="F2転換率（未確定含む）", color="区分", barmode="group",
                             title="登録月別 F2転換率（売上未確定含む）")
                st.plotly_chart(fig, use_container_width=True)
            download_buttons(result, "kpi03_F2転換率", "F2転換率")

# ④ 定期継続率
with tabs[3]:
    st.header("④ 定期継続率")
    st.markdown(
        "**定義**：n回目→n+1回目への継続割合。継続率データCSV（全体）の数値を**そのまま使用**。"
        "（縛りあり/縛りなしの内訳を見たい場合は③F2転換率タブをご参照ください）"
    )
    if df_continuation.empty:
        st.info("継続率データ（全体）をアップロードしてください。")
    else:
        result, kpi_notes = build_continuation_table(df_continuation)
        kpi_results["04_定期継続率"] = result
        render_notes(kpi_notes)
        result_all = result[result["区分"] == "全体"].drop(columns=["区分"])
        if not result_all.empty:
            tab_kakutei, tab_mikakutei = st.tabs(["売上確定", "売上未確定含む"])
            with tab_kakutei:
                d = result_all[["登録月", "n~n+1回", "合計（件）", "売上済（件）", "離脱（件）", "継続率（確定）"]].copy()
                d["継続率（確定）"] = d["継続率（確定）"].apply(lambda x: f"{x:.2f}%")
                st.dataframe(d, use_container_width=True, hide_index=True)
            with tab_mikakutei:
                d = result_all[["登録月", "n~n+1回", "合計（件）", "売上済（件）", "売上前（件）",
                                 "待機中（件）", "離脱（件）", "継続率（未確定含む）"]].copy()
                d["継続率（未確定含む）"] = d["継続率（未確定含む）"].apply(lambda x: f"{x:.2f}%")
                st.dataframe(d, use_container_width=True, hide_index=True)
            download_buttons(result_all, "kpi04_定期継続率", "定期継続率")
        else:
            st.info("「全体」区分の継続率データが見つかりませんでした。")

# ⑤⑥⑦ LTV・平均購入回数・アップセル率
with tabs[4]:
    st.header("⑤⑥⑦ LTV・平均購入回数・アップセル率")
    st.markdown(
        """
        **共通の分母**：登録月ごとに、①と同じ「初回注文（定期回数=1）のスコア≥1」の顧客のみを対象。
        - **⑤LTV**：対象顧客の全注文の「合計（計算用）」（スコア=2のみ）の合計 ÷ 対象顧客数
        - **⑥平均購入回数**：対象顧客ごとの MAX(定期回数) の合計 ÷ 対象顧客数
        - **⑦アップセル率**：「お約束回数（定期）＞0」の注文を持つ対象顧客数 ÷ 対象顧客数
        """
    )
    if df_with_cohort.empty:
        st.info("受注データをアップロードしてください。")
    else:
        result, kpi_notes = calculate_ltv_purchase_upsell(df_with_cohort)
        kpi_results["05_06_07_LTV_購入回数_アップセル率"] = result
        render_notes(kpi_notes)
        if not result.empty:
            display_df = result.copy()
            display_df["LTV"] = display_df["LTV"].apply(lambda x: f"¥{x:,.0f}" if pd.notna(x) else "―")
            display_df["平均購入回数"] = display_df["平均購入回数"].apply(lambda x: f"{x:.3f}回" if pd.notna(x) else "―")
            display_df["アップセル率"] = display_df["アップセル率"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "―")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(px.bar(result, x="登録月", y="LTV", title="登録月別 LTV"), use_container_width=True)
            with c2:
                st.plotly_chart(px.bar(result, x="登録月", y="平均購入回数", title="登録月別 平均購入回数"), use_container_width=True)
            with c3:
                fig = px.bar(result, x="登録月", y="アップセル率", title="登録月別 アップセル率")
                fig.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
            download_buttons(result, "kpi05_07_LTV_購入回数_アップセル率", "LTV・購入回数・アップセル率")

# ⑧ 顧客推移データ
with tabs[5]:
    st.header("⑧ 顧客推移データ")
    st.markdown(
        """
        **定義**：対象月ごとに全顧客を「アクティブ/非アクティブ」×「初回購入/2回目購入/継続/優良」に分類。
        - **アクティブ**：定期受注データ上、対象月末時点で有効な契約を持つ顧客
        - **属性**：対象月末までの受注に基づく MAX(定期回数)（初回=1／2回目=2／継続=3／優良=4以上）
        """
    )
    if df_with_cohort.empty or df_teiki.empty:
        st.info("受注データと定期受注データの両方をアップロードしてください。")
    else:
        available_months = sorted(df_with_cohort["受注月"].dropna().unique().tolist())
        target_months = st.multiselect("対象月を選択", options=available_months, default=available_months, key="kpi08_months")
        if target_months:
            result, kpi_notes = build_customer_trend(df_with_cohort, df_teiki, target_months)
            kpi_results["08_顧客推移データ"] = result
            render_notes(kpi_notes)
            if not result.empty:
                pivot_n = result.pivot_table(index=["状態", "属性"], columns="対象月", values="人数", aggfunc="sum")
                pivot_pct = result.pivot_table(index=["状態", "属性"], columns="対象月", values="構成比", aggfunc="sum")
                st.subheader("人数")
                st.dataframe(pivot_n, use_container_width=True)
                st.subheader("構成比")
                st.dataframe(pivot_pct.style.format("{:.2%}"), use_container_width=True)
                fig = px.bar(result, x="対象月", y="人数", color="属性", barmode="stack", facet_col="状態",
                             title="対象月別 顧客推移")
                st.plotly_chart(fig, use_container_width=True)
                download_buttons(result, "kpi08_顧客推移データ", "顧客推移データ")

# ⑨ 稼働顧客数
with tabs[6]:
    st.header("⑨ 稼働顧客数")
    st.markdown(
        """
        **定義**：「3ヶ月以内購入者＋当月購入者＋翌月発送予定者」の合算 ÷ 全体顧客数。
        次回発送予定日が「基準日〜基準日+2ヶ月」、最終購入日が「基準日-3ヶ月〜基準日」の顧客を「稼働」とみなします。
        """
    )
    if df_with_cohort.empty:
        st.info("受注データをアップロードしてください。")
    else:
        asof_ts = pd.Timestamp(asof_date)
        overall, by_cohort, kpi_notes = calculate_active_customers(df_with_cohort, master, asof_ts)
        kpi_results["09_稼働顧客数"] = by_cohort
        render_notes(kpi_notes)
        if overall:
            c1, c2, c3 = st.columns(3)
            c1.metric("全体顧客数", f"{overall['全体顧客数']:,}人")
            c2.metric("稼働顧客数", f"{overall['稼働顧客数']:,}人")
            c3.metric("稼働率", f"{overall['稼働率']:.2%}" if overall["稼働率"] is not None else "―")
            st.caption(
                f"基準日：{overall['基準日']}／発送予定日レンジ：{overall['対象期間（発送予定日）']}／"
                f"最終購入日レンジ：{overall['対象期間（最終購入日）']}"
            )
            st.subheader("登録月別内訳（参考）")
            display_df = by_cohort.copy()
            display_df["稼働率"] = display_df["稼働率"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "―")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            download_buttons(by_cohort, "kpi09_稼働顧客数", "稼働顧客数")

# ⑩ RF分析
with tabs[7]:
    st.header("⑩ RF分析")
    st.markdown("**定義**：Recency（最終購入日からの経過日数）× Frequency（購入回数）のクロス集計マトリクス。")
    if df_with_cohort.empty:
        st.info("受注データをアップロードしてください。")
    else:
        asof_ts = pd.Timestamp(asof_date)
        matrix, kpi_notes = calculate_rf_matrix(master, df_with_cohort, asof_ts)
        kpi_results["10_RF分析"] = matrix.fillna(0).astype(int) if not matrix.empty else matrix
        render_notes(kpi_notes)
        if not matrix.empty:
            st.caption(f"基準日：{asof_ts.strftime('%Y-%m-%d')}")
            st.dataframe(matrix.fillna(0).astype(int), use_container_width=True)
            download_buttons(matrix.fillna(0).astype(int), "kpi10_RF分析", "RF分析")

# ⑪ 解約理由
with tabs[8]:
    st.header("⑪ 解約理由")
    st.markdown(
        """
        **定義**：定期受注データの「キャンセル理由」をステータス＝キャンセルの行に絞って分類集計。
        （「停止理由」列は実データ上ほぼ使われていないため、キャンセル理由を優先し、空欄時のみ停止理由で補完）
        """
    )
    if df_teiki.empty:
        st.info("定期受注データをアップロードしてください。")
    else:
        result, kpi_notes = calculate_cancel_reasons(df_teiki)
        kpi_results["11_解約理由"] = result
        render_notes(kpi_notes)
        if not result.empty:
            display_df = result.copy()
            display_df["構成比"] = display_df["構成比"].apply(lambda x: f"{x:.2%}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(result, x="理由", y="件数", title="解約理由別 件数"), use_container_width=True)
            download_buttons(result, "kpi11_解約理由", "解約理由")

# ⑫ スコアリング（顧客分類）
with tabs[9]:
    st.header("⑫ スコアリング（顧客分類）")
    st.markdown(
        """
        **定義**：全顧客を MAX(定期回数) で「初回購入顧客／2回目購入顧客／継続顧客（3回以上）」に分類し、
        人数・累積売上・平均LTV・売上構成比を可視化。※Tier1〜6は今回は実装していません。
        """
    )
    if df_with_cohort.empty:
        st.info("受注データをアップロードしてください。")
    else:
        result, kpi_notes = calculate_customer_scoring(df_with_cohort)
        kpi_results["12_スコアリング"] = result
        render_notes(kpi_notes)
        if not result.empty:
            display_df = result.copy()
            display_df["人数構成比"] = display_df["人数構成比"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "―")
            display_df["累積売上"] = display_df["累積売上"].apply(lambda x: f"¥{x:,.0f}")
            display_df["平均LTV"] = display_df["平均LTV"].apply(lambda x: f"¥{x:,.0f}" if pd.notna(x) else "―")
            display_df["売上構成比"] = display_df["売上構成比"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "―")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.pie(result, names="顧客分類", values="人数", title="顧客分類別 人数構成比"), use_container_width=True)
            with c2:
                st.plotly_chart(px.pie(result, names="顧客分類", values="累積売上", title="顧客分類別 売上構成比"), use_container_width=True)
            download_buttons(result, "kpi12_スコアリング", "スコアリング")

# ======================================================================
# レポート保存機能（バックナンバー保存）
# ======================================================================
st.divider()
st.header("📁 レポート保存（バックナンバー）")
st.caption(
    "現在計算されている①〜⑫の結果を1つのExcelファイルにまとめて、"
    "このapp.pyと同じフォルダの「saved_reports」フォルダに保存します。"
    "日付タイトルで管理すれば、データを取得するたびの記録として残せます。"
)

SAVE_DIR = Path(__file__).resolve().parent / "saved_reports"
SAVE_DIR.mkdir(exist_ok=True)

KPI_SHEET_LABELS = {
    "01_初回離脱率": "①初回離脱率",
    "02_期間別解約率": "②期間別解約率",
    "03_F2転換率": "③F2転換率",
    "04_定期継続率": "④定期継続率",
    "05_06_07_LTV_購入回数_アップセル率": "⑤⑥⑦LTV等",
    "08_顧客推移データ": "⑧顧客推移データ",
    "09_稼働顧客数": "⑨稼働顧客数",
    "10_RF分析": "⑩RF分析",
    "11_解約理由": "⑪解約理由",
    "12_スコアリング": "⑫スコアリング",
}


def build_full_report_excel(results: dict) -> bytes:
    """kpi_resultsの中身を1つのExcelファイル（複数シート）にまとめる。"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        wrote_any = False
        for key, label in KPI_SHEET_LABELS.items():
            df = results.get(key)
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=label[:31], index=(df.index.name is not None))
                wrote_any = True
        if not wrote_any:
            pd.DataFrame({"note": ["保存対象のデータがありません"]}).to_excel(writer, sheet_name="empty", index=False)
    return buf.getvalue()


def sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    return name or datetime.date.today().strftime("%Y%m%d")


save_col1, save_col2 = st.columns([2, 1])
with save_col1:
    default_title = datetime.date.today().strftime("%Y%m%d")
    save_title = st.text_input("保存名（例：20260702）", value=default_title, key="save_title")
with save_col2:
    st.write("")
    st.write("")
    save_clicked = st.button("💾 この内容を保存する", use_container_width=True)

if save_clicked:
    safe_name = sanitize_filename(save_title)
    non_empty_count = sum(1 for df in kpi_results.values() if df is not None and not df.empty)
    if non_empty_count == 0:
        st.error("保存できるKPI結果がありません。データをアップロードしてから保存してください。")
    else:
        excel_bytes = build_full_report_excel(kpi_results)
        save_path = SAVE_DIR / f"{safe_name}.xlsx"
        if save_path.exists():
            st.warning(f"「{safe_name}.xlsx」は既に存在します。上書きしました。")
        with open(save_path, "wb") as f:
            f.write(excel_bytes)
        st.success(f"✅ 保存しました：saved_reports/{safe_name}.xlsx（{non_empty_count}件のKPIを含む）")

# 過去の保存レポート一覧
st.subheader("📂 過去の保存レポート")
saved_files = sorted(SAVE_DIR.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)

if not saved_files:
    st.caption("まだ保存されたレポートはありません。")
else:
    for path in saved_files:
        col_a, col_b, col_c = st.columns([3, 2, 1])
        with col_a:
            st.write(f"📄 {path.stem}")
        with col_b:
            mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
            st.caption(f"保存日時：{mtime.strftime('%Y-%m-%d %H:%M')}")
        with col_c:
            with open(path, "rb") as f:
                st.download_button(
                    "📥 開く", data=f.read(), file_name=path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_saved_{path.stem}",
                )
