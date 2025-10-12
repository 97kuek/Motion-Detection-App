# -*- coding: utf-8 -*-

import os
import cv2
from collections import deque

# 入力ソースの設定
SOURCE = 0              # 0: 既定のWebカメラ / 例: "sample.mp4" で動画ファイル
SAVE_DIR = "./img"      # 検出時に保存する画像フォルダ
os.makedirs(SAVE_DIR, exist_ok=True)

# ROI（検出枠）：左上座標 (pointX, pointY) と幅・高さ
pointX, pointY = 100, 100
widthX, widthY = 400, 300

# 調整パラメータ（環境で調整）
WARMUP_FRAMES = 30      # 背景学習のウォームアップ枚数（この間はカウントしない）
LEARN_RATE    = 0.60    # 背景のランニング平均（0.3〜0.8で調整）
AREA_MIN      = 1200    # 最小面積しきい値（遠い被写体/低解像度なら小さめに）
MIN_ROI_W     = 20      # ROIの最小幅（これ未満なら検出スキップ）
MIN_ROI_H     = 20      # ROIの最小高さ（これ未満なら検出スキップ）

# 多重カウント防止（状態機械関連）
PERSIST_FRAMES = 5      # 「動きあり」がこのフレーム数連続したら“イベント開始”
QUIET_FRAMES   = 10     # 逆に「動きなし」がこのフレーム数連続したら“イベント終了”
MIN_EVENT_GAP_S = 1.0   # 2つのイベントの最短間隔（秒）= しきい値の二重起動防止

# カメラ・動画の準備
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"入力ソースを開けませんでした: {SOURCE}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0   # カメラは0を返すことがあるので保険
W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Input size: {W}x{H}, fps={fps:.2f}")

# ROIをフレーム内に丸め込む
x1 = max(0, min(pointX, W - 1))         # 左上X座標
y1 = max(0, min(pointY, H - 1))         # 左上Y座標
x2 = max(0, min(pointX + widthX, W))    # 右下X座標
y2 = max(0, min(pointY + widthY, H))    # 右下Y座標
roi_w = max(0, x2 - x1)                 # ROIの幅
roi_h = max(0, y2 - y1)                 # ROIの高さ
print(f"ROI: x={x1}, y={y1}, w={roi_w}, h={roi_h}")

# 検出に使う変数の初期化
avg = None             # 背景モデル（float型画像）
frameNo = 0            # 現在のフレーム番号
event_count = 0        # イベント（=カウント）数
last_event_frame = -10**9     # 直近のイベント開始フレーム番号
MIN_EVENT_GAP_FR = int(MIN_EVENT_GAP_S * fps)  # フレーム数に変換

# 状態機械（「動きあり」連続カウント/「動きなし」連続カウント）
event_active = False   # True: 現在イベント中（連続する1回の動きとみなす期間）
hit_streak = 0         # 「動きあり」が連続したカウント
quiet_streak = 0       # 「動きなし」が連続したカウント

# 補助：直近ヒット履歴（デバッグ/チューニング用）
recent_hits = deque(maxlen=10)

# メインループ
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    frameNo += 1

    # フレームの前処理
    # グレースケール化することで、計算量削減
    # ぼかし（平滑化）でノイズを減らす
    # GaussianBlur関数
    # 第1引数：入力画像
    # 第2引数：カーネルサイズ（奇数のタプル）
    # 第3引数：標準偏差（0なら自動計算）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # グレースケール化
    gray = cv2.GaussianBlur(gray, (7, 7), 0)        # ノイズ除去

    # 背景の初期化
    if avg is None:
        avg = gray.astype("float")

        # 表示だけして次へ（この時点の差分は意味がない）
        vis = frame.copy()
        if roi_w >= MIN_ROI_W and roi_h >= MIN_ROI_H:
            cv2.rectangle(vis, (x1, y1), (x1 + roi_w, y1 + roi_h), (0, 0, 255), 2)
        cv2.putText(vis, f"Count={event_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("Frame1", vis)
        cv2.imshow("Frame2", gray)
        if (cv2.waitKey(1) & 0xFF) == 27:  # ESCで終了
            break
        continue

    # 背景の更新
    # accumulateWeighted関数
    # 第1引数：現在のフレーム（グレースケール）
    # 第2引数：背景モデル（float型画像）
    # 第3引数：学習率
    # 学習率が大きいほど、最新フレームを重視する（背景の変化に追従しやすい）
    cv2.accumulateWeighted(gray, avg, LEARN_RATE)

    # 背景との差分を計算
    # absdiff関数：絶対差分を計算
    # convertScaleAbs関数：スケール変換と絶対値変換を行い、8ビット画像に変換
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # 表示用に原画像をコピー
    vis = frame.copy()

    # ROIの有効性をチェック
    # roi_valid は　roi_w と roi_h がそれぞれ最小値以上かどうかで判定
    roi_valid = (roi_w >= MIN_ROI_W) and (roi_h >= MIN_ROI_H)

    if not roi_valid:  # ROIが小さすぎる場合は検出をスキップ（まずはROIを大きく）
        cv2.putText(vis, "ROI too small - enlarge ROI", (20, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        show = frameDelta 

    else:   # ROIが有効な場合
        roi = frameDelta[y1:y1 + roi_h, x1:x1 + roi_w]

        if roi.size == 0:   # ROIが画像外にはみ出して空配列になる場合
            # 万一、空配列なら検出スキップ（はみ出し等）
            cv2.putText(vis, "Empty ROI - adjust ROI", (20, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            show = frameDelta
        else:

            # 大津の二値化
            # cv2.threshold関数
            # 第1引数：入力画像（グレースケール）
            # 第2引数：しきい値（0なら自動計算）
            # 第3引数：しきい値を超えたときの最大値
            # 第4引数：しきい値の種類（ここでは大津の二値化）
            # 戻り値は2つあるが、1つ目は使わないのでアンダースコアで受ける
            _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_OTSU)

            #　---- (h) 形態学的処理でノイズ除去 ----
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            thresh = cv2.dilate(thresh, kernel, iterations=2)

            # ---- (i) 輪郭抽出（OpenCVバージョン差に対応）----
            found = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = found[0] if len(found) == 2 else found[1]

            # ---- (j) 1フレーム内に“十分大きい動き”があるか判定 ----
            hit_this_frame = False

            # 1) 最大輪郭の面積で判定（単純で理解しやすい）
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < AREA_MIN:
                    continue

                # boundingRect関数：輪郭を囲む最小の直線矩形を計算
                x, y, w, h = cv2.boundingRect(cnt)

                # 検出枠（緑）を描画
                # rectangle関数：矩形を描画
                # 第1引数：画像
                # 第2引数：左上座標
                # 第3引数：右下座標
                # 第4引数：色（BGR）
                # 第5引数：線の太さ
                cv2.rectangle(vis, (x1 + x, y1 + y), (x1 + x + w, y1 + y + h), (0, 255, 0), 2)
                cv2.putText(vis, f"area={int(area)}", (x1 + x, max(0, y1 + y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                hit_this_frame = True

            if hit_this_frame:
                hit_streak += 1
                quiet_streak = 0
                recent_hits.append(1)
            else:
                hit_streak = 0
                quiet_streak += 1
                recent_hits.append(0)

            # イベント開始の条件：
            #   - まだイベントが始まっていない
            #   - ヒットがPERSIST_FRAMES連続
            #   - 前回イベントからMIN_EVENT_GAP_FRフレーム以上経過
            if (not event_active) and (hit_streak >= PERSIST_FRAMES) and \
               (frameNo - last_event_frame >= MIN_EVENT_GAP_FR) and \
               (frameNo > WARMUP_FRAMES):  # ウォームアップ中は無効
                event_active = True
                last_event_frame = frameNo
                event_count += 1

                # 必要ならイベント開始の瞬間だけ画像保存
                out_path = os.path.join(SAVE_DIR, f"img{event_count}.jpg")
                cv2.imwrite(out_path, vis)

            # イベント終了の条件：
            #   - 現在イベント中
            #   - ミス（動きなし）がQUIET_FRAMES連続
            if event_active and (quiet_streak >= QUIET_FRAMES):
                event_active = False

            # デバッグ用表示を右側に
            show = thresh

    # 情報表示
    cv2.putText(vis, f"Count={event_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA) # カウント数
    if roi_valid:
        # ROIを赤枠で描画
        # rectangle関数：矩形を描画
        # 第1引数：画像
        # 第2引数：左上座標
        # 第3引数：右下座標
        # 第4引数：色（BGR）
        # 第5引数：線の太さ
        cv2.rectangle(vis, (x1, y1), (x1 + roi_w, y1 + roi_h), (0, 0, 255), 2)

    # 背景が安定するまでウォームアップ中であることを表示
    if frameNo <= WARMUP_FRAMES:
        # cv2.putText関数：テキストを画像に描画
        # 第1引数：画像
        # 第2引数：描画するテキスト
        # 第3引数：テキストの左下座標
        # 第4引数：フォント
        # 第5引数：フォントサイズ
        # 第6引数：色（BGR）
        # 第7引数：太さ
        # 第8引数：線の種類
        cv2.putText(vis, f"Warming up... {frameNo}/{WARMUP_FRAMES}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 1, cv2.LINE_AA)

    # Window表示
    cv2.imshow("Frame1", vis)                                         # 元の画像と検出枠を表示
    cv2.imshow("Frame2", show if 'show' in locals() else frameDelta)  # 2値化画像等

    # Escキーで終了
    if (cv2.waitKey(1) & 0xFF) == 27:
        break

# 後処理
cap.release()               # キャプチャ解放
cv2.destroyAllWindows()     # ウィンドウをすべて閉じる
