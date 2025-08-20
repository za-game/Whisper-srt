# Whisper-srt 中文使用手冊

Whisper-srt 是一個利用 OpenAI Whisper 將音訊即時轉換成字幕並於螢幕上覆蓋顯示的工具，適合直播或影片播放時使用。

## 安裝
1. 安裝 Python 3.10 以上版本。
2. 安裝最新版 PyTorch（若使用 Facebook 翻譯模型需 2.6 以上）。
3. 安裝相依套件：
   ```
   pip install -r requirements.txt
   ```

## 使用方式
1. 執行 `python mWhisperSub.py` 啟動轉寫。
2. 系統匣圖示提供各項設定：
   - **文字樣式設定…**：調整字型、顏色、外框與陰影等效果。
   - **顯示預覽字幕**：顯示自訂的預覽文字，方便調整樣式。
   - 主視窗提供類似 Discord 的音量門檻滑桿，可視化調整自動 VAD 閾值，低於門檻的音訊會被丟棄。
   - 另可設定 VAD 靜音門檻、溫度、logprob 閾值與壓縮比閾值等參數。
3. 轉寫結果會寫入 `live.srt`，並於覆蓋視窗中顯示。

## 設定
所有設定會儲存在系統的 `QSettings` 中，下次啟動時會自動套用。

## 翻譯模型
程式可選擇性地使用 Transformers 模型進行翻譯。預設以英文為中介，必要時會自動下載所需模型：

- 英文 ↔ 日文：`Helsinki-NLP/opus-mt-en-ja`
- 英文 ↔ 韓文：`Helsinki-NLP/opus-mt-en-ko`
- 英文 ↔ 中文：`Helsinki-NLP/opus-mt-en-zh`

針對日文、韓文與中文之間無英文中介的翻譯，會退回多語模型：

- 日文 ↔ 韓文、日文 ↔ 中文：`facebook/m2m100_418M`（語言代碼 `ja`、`ko`、`zh`）
- 中文 ↔ 韓文：`facebook/nllb-200-distilled-600M`（語言代碼 `jpn_Jpan`、`kor_Hang`、`zho_Hans`）

如模型需要 Hugging Face 認證，程式會提示輸入 token，可先於終端機執行 `huggingface-cli login`。

GUI 選單會以「(未下載)」標示尚未取得的模型或翻譯語言，且僅在沒有可用模型時顯示「未選擇」。

## 模型下載設定
所有語音與翻譯模型的下載網址及儲存路徑集中於 `Config.json`，
欲更換鏡像或更新版本時，修改此檔即可。預設 `model_path` 為專案根目錄下的 `models`。
GUI 下載的模型與啟動時的模型搜尋都會使用此路徑；`cache_path` 可設定 Hugging Face 暫存位置，預設為系統的 `~/.cache/huggingface/hub`。
