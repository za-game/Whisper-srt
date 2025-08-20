#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Whisper Live Caption – EXE 發佈專用啟動器
========================================
功能：
1. 啟動時偵測 GPU 與驅動版本
2. 推算最適 CUDA runtime (cu123 / cu121 / cu118 / cpu)
3. 檢查必要套件 (torch, faster-whisper, PyQt5 等)
4. 缺少時 GUI 詢問並安裝
5. 完成後啟動 mWhisperSub.py，並在本程式內建字幕 Overlay + 系統匣設定
"""
import sys, os
import json
def ensure_local_path():
    base_dir = os.path.dirname(__file__)

    # 判斷條件 1: sys.path 沒有當前目錄或 base_dir
    has_local = ('' in sys.path) or (base_dir in sys.path)

    # 判斷條件 2: 是否有 _pth 檔 (embedded 的特徵)
    pth_exists = any(fname.endswith('._pth') for fname in os.listdir(os.path.dirname(sys.executable)))

    if not has_local or pth_exists:
        sys.path.insert(0, base_dir)
ensure_local_path()
from typing import Optional
import subprocess
import sys
import base64
import shutil
import importlib.util
from pathlib import Path
from PyQt5 import QtCore, QtWidgets, QtGui
from project_io import save_project, load_project

# Register text cursor/block types for thread-safe queued connections
_register_meta = getattr(QtCore, "qRegisterMetaType", None)
if _register_meta is not None:  # PyQt5 with qRegisterMetaType
    _register_meta(QtGui.QTextCursor)
    _register_meta(QtGui.QTextBlock)
else:  # Fallback for builds without qRegisterMetaType
    try:
        QtCore.QMetaType.registerType(QtGui.QTextCursor)
        QtCore.QMetaType.registerType(QtGui.QTextBlock)
    except Exception:
        pass

import os
import urllib.request
import sounddevice as sd
import signal
import threading
import tqdm
import numpy as np
from overlay import Settings, SubtitleOverlay, Tray
from srt_utils import LiveSRTWatcher
ROOT_DIR = Path(__file__).resolve().parent
ENGINE_PY = ROOT_DIR / "mWhisperSub.py"
OVERLAY_PY = ROOT_DIR / "srt_overlay_tool.py"

with (ROOT_DIR / "Config.json").open(encoding="utf-8") as f:
    CONFIG = json.load(f)
MODEL_PATH = (ROOT_DIR / CONFIG.get("model_path", "models")).resolve()
MODEL_PATH.mkdir(parents=True, exist_ok=True)
HF_CACHE = MODEL_PATH / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE))
MODEL_REPO_MAP = CONFIG["MODEL_REPO_MAP"]
TRANSLATE_REPO_MAP = {
    tuple(k.split("-")): v for k, v in CONFIG["TRANSLATE_REPO_MAP"].items()
}

# ──────────── 錄音設備偵測 ────────────
def list_audio_devices():
    devices = []
    try:
        devs = sd.query_devices()
        for idx, dev in enumerate(devs):
            if dev["max_input_channels"] > 0:  # 只列輸入設備
                sr = int(dev["default_samplerate"])
                devices.append((idx, f"{dev['name']} ({sr} Hz)", sr))
    except Exception as e:
        devices.append((-1, f"偵測失敗: {e}", 16000))
    return devices
# ──────────── GPU 偵測 ────────────
def detect_gpu():
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            encoding="utf-8"
        ).strip().split("\n")[0]
        name, driver = [x.strip() for x in out.split(",")]
        return name, driver
    except Exception:
        return None, None

def list_gpus():
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8",
        )
        return [n.strip() for n in out.splitlines() if n.strip()]
    except Exception:
        return []

def recommend_cuda_version(driver_version):
    try:
        major = int(driver_version.split(".")[0])
        if major >= 535:
            primary = "cu123"
        elif major >= 525:
            primary = "cu121"
        else:
            primary = "cu118"
    except:
        return "cpu"

    # 驗證該 CUDA wheel 是否存在，否則降級
    if torch_wheel_exists(primary):
        return primary
    for fallback in ["cu121", "cu118", "cpu"]:
        if torch_wheel_exists(fallback):
            return fallback
    return "cpu"

# ──────────── 套件檢查 ────────────
def torch_wheel_exists(cuda_tag):
    try:
        url = f"https://download.pytorch.org/whl/{cuda_tag}/torch/"
        with urllib.request.urlopen(url) as resp:
            return resp.status == 200
    except:
        return False
def is_installed(pkg):
    return importlib.util.find_spec(pkg) is not None

def run_pip(args, log_fn=None, cancel_flag=None):
    cmd = [sys.executable, "-m", "pip"] + args
    if log_fn:
        log_fn(f"執行: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if log_fn:
                log_fn(line.rstrip())
            if cancel_flag and cancel_flag():
                proc.terminate()
                break
        proc.wait()
    finally:
        if proc.stdout:
            proc.stdout.close()
    if cancel_flag and cancel_flag():
        raise RuntimeError("使用者取消")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

def install_deps(cuda_tag, log_fn=None, cancel_flag=None):
    pkgs = []
    if cuda_tag.startswith("cu"):
        # 安裝 PyTorch GPU 版
        pkgs += [
            "torch", "torchvision", "torchaudio",
            "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}"
        ]
        run_pip(["install", "--upgrade"] + pkgs, log_fn=log_fn, cancel_flag=cancel_flag)
    else:
        # CPU 版 PyTorch
        run_pip(
            [
                "install",
                "--upgrade",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            log_fn=log_fn,
            cancel_flag=cancel_flag,
        )
    # faster-whisper 與 PyQt5
    run_pip(
        [
            "install",
            "--upgrade",
            "faster-whisper",
            "PyQt5",
            "sounddevice",
            "webrtcvad-wheels",
            "scipy",
            "opencc-python-reimplemented",
            "srt",
            "tqdm",
            "huggingface_hub",
        ],
        log_fn=log_fn,
        cancel_flag=cancel_flag,
    )

# ──────────── GUI ────────────
class BootstrapWin(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Caption – 安裝與啟動器")
        self.resize(600, 300)
        main_layout = QtWidgets.QVBoxLayout()

        # 內嵌的設定 / overlay / 系統匣
        self.settings = Settings()
        self.overlay = None
        self.tray = None
        self.srt_watcher = None
        self.proc = None  # mWhisperSub 子程序的 handle

        # 參數設定區
        form_layout = QtWidgets.QFormLayout()

        # 模型選擇
        self.model_combo = QtWidgets.QComboBox()
        for m in ["tiny", "base", "small", "medium", "large-v2"]:
            self.model_combo.addItem(m, m)
        form_layout.addRow("模型", self.model_combo)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)

        # 語言選擇
        self.lang_combo = QtWidgets.QComboBox()
        self.lang_combo.addItems(["auto", "zh", "ja", "en", "ko"])
        self.lang_combo.setCurrentText("zh")  # 預設就是 zh
        form_layout.addRow("語言", self.lang_combo)

        self.translate_chk = QtWidgets.QCheckBox("翻譯")
        self.translate_lang_combo = QtWidgets.QComboBox()
        self.translate_lang_combo.setEnabled(False)
        self.translate_chk.toggled.connect(self.translate_lang_combo.setEnabled)
        self.lang_combo.currentTextChanged.connect(self._update_translate_lang_options)
        self.translate_lang_combo.currentIndexChanged.connect(self._on_translate_lang_changed)
        self._update_translate_lang_options()
        # 翻譯語言僅在勾選翻譯時生效
        form_layout.addRow(self.translate_chk, self.translate_lang_combo)

        # 終端機顯示與 log 等級
        self.console_chk = QtWidgets.QCheckBox("顯示終端機")
        self.console_chk.setChecked(False)
        self.log_level_combo = QtWidgets.QComboBox()
        # 依你的需求只提供 1 / 2
        self.log_level_combo.addItems(["1", "2"])
        self.log_level_combo.setCurrentText("1")
        self.log_level_combo.setEnabled(False)
        # 勾選時才允許選 log 等級
        self.console_chk.toggled.connect(self.log_level_combo.setEnabled)
        form_layout.addRow(self.console_chk, self.log_level_combo)


        # 中文轉換（OpenCC 模式）
        self.zh_combo = QtWidgets.QComboBox()
        # 與 mWhisperSub.py 的 choices 對齊
        self.zh_combo.addItems(["t2tw", "s2t", "s2twp", "none"])
        self.zh_combo.setCurrentText("s2twp")
        form_layout.addRow("中文轉換 (OpenCC)", self.zh_combo)
        # ───────── 專案（在 Hotword / SRT 之上）─────────
        proj_layout = QtWidgets.QHBoxLayout()
        self.project_name_edit = QtWidgets.QLabel("未選擇專案")
        self.project_path_edit = QtWidgets.QLineEdit()
        self.project_path_edit.setPlaceholderText("未選擇專案路徑")
        self.project_path_edit.setReadOnly(True)
        self.project_dir: Optional[Path] = None
        self.pick_project_btn = QtWidgets.QPushButton("選擇專案…")
        self.pick_project_btn.clicked.connect(self.choose_project)
        self.new_project_btn = QtWidgets.QPushButton("新建專案…")
        self.new_project_btn.clicked.connect(self.create_project)
        proj_layout.addWidget(self.project_name_edit, 2)   # 專案名稱
        proj_layout.addWidget(self.project_path_edit, 4)   # 路徑
        proj_layout.addWidget(self.pick_project_btn, 1)    # 選擇
        proj_layout.addWidget(self.new_project_btn, 1)     # 新建
        form_layout.addRow("專案", proj_layout)

        # ───────── Hotword（專案名稱 / 路徑 / 選擇檔案 / 編輯）─────────
        hot_layout = QtWidgets.QHBoxLayout()
        self.hot_proj_name_edit = QtWidgets.QLabel("未選擇專案")
        self.hotwords_edit = QtWidgets.QLineEdit()
        self.hotwords_edit.setPlaceholderText("未選擇熱詞檔（*.txt）")
        self.hotwords_edit.setReadOnly(True)
        self.pick_hot_btn = QtWidgets.QPushButton("選擇檔案…")
        self.pick_hot_btn.clicked.connect(self.pick_hotwords_file)
        self.edit_hot_btn = QtWidgets.QPushButton("編輯")
        self.edit_hot_btn.clicked.connect(self.edit_hotwords_file)
        hot_layout.addWidget(self.hot_proj_name_edit, 2)   # 專案名稱
        hot_layout.addWidget(self.hotwords_edit, 4)        # 路徑
        hot_layout.addWidget(self.pick_hot_btn, 1)         # 選擇檔案
        hot_layout.addWidget(self.edit_hot_btn, 1)         # 編輯
        form_layout.addRow("Hotword", hot_layout)

        # ───────── SRT（專案名稱 / 路徑 / 選擇檔案 / 編輯）─────────
        srt_layout = QtWidgets.QHBoxLayout()
        self.srt_proj_name_edit = QtWidgets.QLabel("未選擇專案")
        self.srt_edit = QtWidgets.QLineEdit()
        self.srt_edit.setPlaceholderText(str(self.settings.srt_path))
        self.srt_edit.setReadOnly(True)
        self.pick_srt_btn = QtWidgets.QPushButton("選擇檔案…")
        self.pick_srt_btn.clicked.connect(self.pick_srt_file)
        self.edit_srt_btn = QtWidgets.QPushButton("編輯")
        self.edit_srt_btn.clicked.connect(self.edit_srt_file)
        srt_layout.addWidget(self.srt_proj_name_edit, 2)   # 專案名稱
        srt_layout.addWidget(self.srt_edit, 4)             # 路徑
        srt_layout.addWidget(self.pick_srt_btn, 1)         # 選擇檔案
        srt_layout.addWidget(self.edit_srt_btn, 1)         # 編輯
        form_layout.addRow("SRT", srt_layout)

        # 裝置選擇
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        form_layout.addRow("裝置", self.device_combo)
        self._update_cuda_option(False)

        # GPU 選擇
        self.gpu_combo = QtWidgets.QComboBox()
        self.refresh_gpu_list()
        self.gpu_combo.setEnabled(self.device_combo.currentText().startswith("cuda"))
        form_layout.addRow("GPU", self.gpu_combo)

        # 錄音設備選擇（展開前自動刷新）
        self.audio_device_combo = QtWidgets.QComboBox()
        form_layout.addRow("錄音設備", self.audio_device_combo)
        self.refresh_audio_devices()
        def _showPopup():
            self.refresh_audio_devices()
            QtWidgets.QComboBox.showPopup(self.audio_device_combo)
        self.audio_device_combo.showPopup = _showPopup

        # VAD 控制
        self.vad_combo = QtWidgets.QComboBox()
        self.vad_combo.addItems(["0", "1", "2", "3", "Auto"])
        self.vad_combo.setCurrentText("1")
        self.vad_combo.setToolTip("VAD 等級：0 最寬鬆、3 最嚴格；Auto 會依噪音自動選擇")
        form_layout.addRow("VAD", self.vad_combo)
        form_layout.labelForField(self.vad_combo).setToolTip(self.vad_combo.toolTip())

        # 音量門檻滑桿與即時音量條
        self.mic_level_bar = QtWidgets.QProgressBar()
        self.mic_level_bar.setRange(0, 100)
        self.mic_level_bar.setTextVisible(False)
        self.mic_level_bar.setFixedHeight(6)
        self.mic_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mic_slider.setRange(0, 100)
        self.mic_slider.setValue(20)
        self.mic_slider.setToolTip("自動 VAD 的音量門檻，低於此值視為靜音")
        gate_widget = QtWidgets.QWidget()
        gate_layout = QtWidgets.QVBoxLayout(gate_widget)
        gate_layout.setContentsMargins(0, 0, 0, 0)
        gate_layout.addWidget(self.mic_level_bar)
        gate_layout.addWidget(self.mic_slider)
        form_layout.addRow("音量門檻", gate_widget)
        form_layout.labelForField(gate_widget).setToolTip(self.mic_slider.toolTip())
        self.mic_slider.valueChanged.connect(lambda _=None: self.schedule_autosave(300))

        # 靜音門檻秒數（mWhisperSub: --silence，預設 0.3）
        self.silence_spin = QtWidgets.QDoubleSpinBox()
        self.silence_spin.setDecimals(2)
        self.silence_spin.setRange(0.00, 5.00)
        self.silence_spin.setSingleStep(0.05)
        self.silence_spin.setValue(0.30)
        self.silence_spin.setToolTip("兩句之間最少需要的靜音長度，調低可更快切句")
        form_layout.addRow("靜音門檻 (秒)", self.silence_spin)
        form_layout.labelForField(self.silence_spin).setToolTip(self.silence_spin.toolTip())

        # 溫度 / 幻覺過濾參數
        self.temp_edit = QtWidgets.QLineEdit("0")
        self.temp_edit.setToolTip("解碼溫度，可填入多個以逗號分隔；0 為最穩定")
        form_layout.addRow("溫度", self.temp_edit)
        form_layout.labelForField(self.temp_edit).setToolTip(self.temp_edit.toolTip())
        self.logprob_spin = QtWidgets.QDoubleSpinBox()
        self.logprob_spin.setRange(-5.0, 0.0)
        self.logprob_spin.setSingleStep(0.1)
        self.logprob_spin.setValue(-1.0)
        self.logprob_spin.setToolTip("平均 logprob 低於此值則丟棄；-1 表停用")
        form_layout.addRow("logprob 閾值", self.logprob_spin)
        form_layout.labelForField(self.logprob_spin).setToolTip(self.logprob_spin.toolTip())
        self.comp_ratio_spin = QtWidgets.QDoubleSpinBox()
        self.comp_ratio_spin.setRange(0.0, 10.0)
        self.comp_ratio_spin.setSingleStep(0.1)
        self.comp_ratio_spin.setValue(2.4)
        self.comp_ratio_spin.setToolTip("gzip 壓縮比超過此值視為重複幻覺；0 表停用")
        form_layout.addRow("壓縮比閾值", self.comp_ratio_spin)
        form_layout.labelForField(self.comp_ratio_spin).setToolTip(self.comp_ratio_spin.toolTip())

        self.noise_btn = QtWidgets.QPushButton("偵測噪音等級")
        self.noise_btn.clicked.connect(self.detect_noise_level)
        form_layout.addRow(self.noise_btn)

        main_layout.addLayout(form_layout)
        self.status = QtWidgets.QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        main_layout.addWidget(self.status)

        self.start_btn = QtWidgets.QPushButton("開始轉寫")
        self.start_btn.clicked.connect(self.start_clicked)
        main_layout.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("停止轉寫")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_clicked)
        main_layout.addWidget(self.stop_btn)

        self.exit_btn = QtWidgets.QPushButton("結束")
        self.exit_btn.clicked.connect(self.exit_clicked)
        main_layout.addWidget(self.exit_btn)

        main_tab = QtWidgets.QWidget()
        main_tab.setLayout(main_layout)

        env_layout = QtWidgets.QVBoxLayout()
        self.pkg_label = QtWidgets.QLabel("torch/CUDA: 未檢測")
        self.gpu_label = QtWidgets.QLabel("GPU: 未檢測")
        self.driver_label = QtWidgets.QLabel("驅動版本: 未檢測")
        env_layout.addWidget(self.pkg_label)
        env_layout.addWidget(self.gpu_label)
        env_layout.addWidget(self.driver_label)
        self.detect_gpu_btn = QtWidgets.QPushButton("偵測GPU加速")
        self.detect_gpu_btn.clicked.connect(self.check_env)
        self.uninstall_torch_btn = QtWidgets.QPushButton("解除安裝torch/CUDA")
        self.uninstall_torch_btn.clicked.connect(self.uninstall_torch_cuda)
        self.install_torch_btn = QtWidgets.QPushButton("安裝torch/CUDA")
        self.install_torch_btn.clicked.connect(self.install_torch_cuda)
        env_layout.addWidget(self.detect_gpu_btn)
        env_layout.addWidget(self.uninstall_torch_btn)
        env_layout.addWidget(self.install_torch_btn)
        env_layout.addStretch()
        env_tab = QtWidgets.QWidget()
        env_tab.setLayout(env_layout)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(main_tab, "基本設定")
        self.tabs.addTab(env_tab, "環境設定")
        self.setCentralWidget(self.tabs)

        self._last_model_index = self.model_combo.currentIndex()
        self._refresh_model_items()
        QtCore.QTimer.singleShot(100, self.check_env)
        # project autosave
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._do_autosave)
        self._autosave_pending = False
        self._level_timer = QtCore.QTimer(self)
        self._level_timer.timeout.connect(self._poll_mic_level)
        self._level_timer.start(200)
        # 設定變更 → 觸發自動儲存（debounce）
        self.settings.changed.connect(lambda: self.schedule_autosave(300))
        # GUI 欄位變更也要 autosave
        self.hotwords_edit.textChanged.connect(lambda _=None: self.schedule_autosave(300))
        self.srt_edit.textChanged.connect(lambda _=None: self.schedule_autosave(300))
        self.lang_combo.currentIndexChanged.connect(lambda _=None: self.schedule_autosave(300))
        self.console_chk.toggled.connect(lambda _=None: self.schedule_autosave(300))
        self.log_level_combo.currentIndexChanged.connect(lambda _=None: self.schedule_autosave(300))
        self.zh_combo.currentIndexChanged.connect(lambda _=None: self.schedule_autosave(300))
        self.device_combo.currentIndexChanged.connect(self.device_changed)
        self.gpu_combo.currentIndexChanged.connect(lambda _=None: self.schedule_autosave(300))
        self.audio_device_combo.currentIndexChanged.connect(lambda _=None: self.schedule_autosave(300))
        self.vad_combo.currentIndexChanged.connect(lambda _=None: self.schedule_autosave(300))
        self.silence_spin.valueChanged.connect(lambda _=None: self.schedule_autosave(300))
        # 主視窗移動/縮放 → autosave main_window_geometry
        self.installEventFilter(self)
        # 啟動時嘗試還原上次專案（使用全域 QSettings）
        QtCore.QTimer.singleShot(0, self._auto_open_last_project)
        # —— 統一的本地模型資料夾：model_path/<Repo> —— #
    def _repo_local_dir(self, repo_id: str) -> Path:
        return MODEL_PATH / repo_id.replace("/", "--")

    def _update_translate_lang_options(self):
        opts = ["JA", "EN", "KO", "ZH"]
        src = self.lang_combo.currentText().upper()
        if src in opts:
            opts.remove(src)
        self.translate_lang_combo.blockSignals(True)
        self.translate_lang_combo.clear()
        model = self.translate_lang_combo.model()
        available_items: list[tuple[int, str]] = []
        for i, code in enumerate(opts):
            pair = (src.lower(), code.lower())
            repo = TRANSLATE_REPO_MAP.get(pair)
            available = True
            if repo:
                available = self._translate_model_downloaded(repo)
            text = code if available else f"{code} (未下載)"
            self.translate_lang_combo.addItem(text, code)
            brush = None if available else QtGui.QBrush(QtGui.QColor("gray"))
            model.setData(model.index(i, 0), brush, QtCore.Qt.ForegroundRole)
            if available:
                available_items.append((i, code))
        if available_items:
            target = next((i for i, c in available_items if c == "EN"), available_items[0][0])
            self.translate_lang_combo.setCurrentIndex(target)
        else:
            self.translate_lang_combo.insertItem(0, "未選擇", "")
            self.translate_lang_combo.setCurrentIndex(0)
        self.translate_lang_combo.blockSignals(False)
        self._last_trans_index = self.translate_lang_combo.currentIndex()

    def _model_downloaded(self, name: str) -> bool:
        repo = MODEL_REPO_MAP.get(name, name)
        paths = [
            MODEL_PATH / name,
            MODEL_PATH / repo.replace("/", "--"),
        ]
        for p in paths:
            if p.exists() and any(p.iterdir()):
                return True
        return False

    def _translate_model_downloaded(self, repo: str) -> bool:
        paths = [
            MODEL_PATH / repo,
            MODEL_PATH / repo.replace("/", "--"),
        ]
        for p in paths:
            if p.exists() and any(p.iterdir()):
                return True
        return False

    def _refresh_model_items(self):
        self.model_combo.blockSignals(True)
        try:
            model = self.model_combo.model()
            available_names: list[str] = []
            for i in range(self.model_combo.count()):
                base = self.model_combo.itemData(i)
                if not base:
                    continue
                available = self._model_downloaded(base)
                text = base if available else f"{base} (未下載)"
                self.model_combo.setItemText(i, text)
                brush = None if available else QtGui.QBrush(QtGui.QColor("gray"))
                model.setData(model.index(i, 0), brush, QtCore.Qt.ForegroundRole)
                if available:
                    available_names.append(base)
            placeholder_idx = self.model_combo.findData("")
            if available_names:
                if placeholder_idx != -1:
                    self.model_combo.removeItem(placeholder_idx)
                current_name = self.model_combo.currentData()
                if not current_name or not self._model_downloaded(current_name):
                    self.model_combo.setCurrentIndex(self.model_combo.findData(available_names[0]))
            else:
                if placeholder_idx == -1:
                    self.model_combo.insertItem(0, "未選擇", "")
                    placeholder_idx = 0
                self.model_combo.setCurrentIndex(placeholder_idx)
        finally:
            self.model_combo.blockSignals(False)
        self._last_model_index = self.model_combo.currentIndex()

    def _set_model_name(self, name: str):
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == name:
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(i)
                self.model_combo.blockSignals(False)
                self._last_model_index = i
                return

    def _on_model_changed(self, idx: int):
        base = self.model_combo.itemData(idx)
        if not base:
            self._last_model_index = idx
            return
        if not self._model_downloaded(base):
            resp = QtWidgets.QMessageBox.question(
                self,
                "下載模型",
                f"模型 {base} 未下載，現在下載嗎？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if resp == QtWidgets.QMessageBox.Yes:
                repo = MODEL_REPO_MAP.get(base, base)
                try:
                    self._download_model_with_progress(repo)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "下載失敗", str(e))
                self._refresh_model_items()
            else:
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(self._last_model_index)
                self.model_combo.blockSignals(False)
                return
        self._last_model_index = self.model_combo.currentIndex()
        self.schedule_autosave(300)

    def _on_translate_lang_changed(self, idx: int):
        tgt = self.translate_lang_combo.itemData(idx)
        if not tgt:
            self._last_trans_index = idx
            return
        src = self.lang_combo.currentText().lower()
        repo = TRANSLATE_REPO_MAP.get((src, tgt.lower()))
        if repo and not self._translate_model_downloaded(repo):
            resp = QtWidgets.QMessageBox.question(
                self,
                "下載模型",
                f"翻譯模型 {repo} 未下載，現在下載嗎？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if resp == QtWidgets.QMessageBox.Yes:
                try:
                    self._download_model_with_progress(repo)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "下載失敗", str(e))
                self._update_translate_lang_options()
            else:
                self.translate_lang_combo.blockSignals(True)
                self.translate_lang_combo.setCurrentIndex(self._last_trans_index)
                self.translate_lang_combo.blockSignals(False)
                return
        self._last_trans_index = idx
    def closeEvent(self, ev: QtGui.QCloseEvent):
        ev.ignore()
        self.hide()
    def _auto_open_last_project(self):
        last = self.settings._qs.value("last_project_dir", "", type=str)
        if not last:
            return
        d = Path(last)
        if not d.exists():
            return
        self.project_dir = d
        self.project_name_edit.setText(d.name)
        self.project_path_edit.setText(str(d))
        self.hot_proj_name_edit.setText(d.name)
        self.srt_proj_name_edit.setText(d.name)
        # 嘗試載入專案（若 overlay 尚未建立，先還原 Settings；overlay 幾何會在 start 後再套用）
        self._load_project()
        # 專案檔可能記錄了過期的 Hotwords/SRT 路徑；若檔案不存在或不在當前專案資料夾，清空後 fallback
        hot_p = Path(self.hotwords_edit.text().strip()) if self.hotwords_edit.text().strip() else None
        if not hot_p or not hot_p.exists() or hot_p.parent != d:
            self.hotwords_edit.clear()
        srt_p = Path(self.srt_edit.text().strip()) if self.srt_edit.text().strip() else None
        if not srt_p or not srt_p.exists() or srt_p.parent != d:
            self.srt_edit.clear()
        # 若專案檔缺少或失效路徑，從資料夾內最新檔案補齊
        if not self.hotwords_edit.text().strip() or not self.srt_edit.text().strip():
            self._fallback_fill_paths_from_dir(d)
    # ---- Project persistence ----
    def _project_path(self) -> Path | None:
        if not getattr(self, 'project_dir', None):
            return None
        return Path(self.project_dir) / "overlay.mwsproj"

    def _write_project(self):
        p = self._project_path()
        if not p:
            return
        overlay_bundle = None
        if self.overlay:
            overlay_bundle = self.overlay.to_dict()
        else:
            # 尚未建立 overlay 時保留既有設定，避免被覆寫
            if p.exists():
                try:
                    overlay_bundle = load_project(str(p)).get("overlay_bundle", {}) or {}
                except Exception:
                    overlay_bundle = {}
            else:
                overlay_bundle = {}
        payload = {
            "app": "WhisperLiveCaption",
            "settings": {
                "strategy": getattr(self.settings, "strategy", "overlay"),
                "cps": float(getattr(self.settings, "cps", 15)),
                "fixed": float(getattr(self.settings, "fixed", 2)),
                "preview": bool(getattr(self.settings, "preview", False)),
            },
            "overlay_bundle": overlay_bundle,
            "gui": {
                # 主視窗幾何（Qt 的 QByteArray → base64 字串以存 JSON）
                "main_window_geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
                # 保存 GUI 關鍵路徑與設定，開專案時可直接還原
                "project_name": self.project_name_edit.text().strip(),
                "project_path": self.project_path_edit.text().strip(),
                "hot_proj_name": self.hot_proj_name_edit.text().strip(),
                "srt_proj_name": self.srt_proj_name_edit.text().strip(),
                "hotwords_path": self.hotwords_edit.text().strip(),
                "srt_path": str(self.settings.srt_path) if getattr(self.settings, "srt_path", None) else "",
                "model": self.model_combo.currentData(),
                "lang": self.lang_combo.currentText(),
                "console": bool(self.console_chk.isChecked()),
                "log_level": self.log_level_combo.currentText(),
                "zh_mode": self.zh_combo.currentText(),
                "device": "cuda" if self.device_combo.currentText().startswith("cuda") else "cpu",
                "gpu_index": self.gpu_combo.currentIndex(),
                "audio_device_text": self.audio_device_combo.currentText(),
                "audio_device_index": self.audio_device_combo.currentIndex(),
                "vad": self.vad_combo.currentText(),
                "mic_gate": int(self.mic_slider.value()),
                "silence": float(self.silence_spin.value()),
            },
        }
        try:
            save_project(str(p), payload)
            self.status.appendPlainText(f"[Autosave] {p}")
        except Exception as e:
            self.status.appendPlainText(f"[Autosave] 失敗: {e}")

    def _load_project(self):
        p = self._project_path()
        if not p or not p.exists():
            return
        try:
            obj = load_project(str(p))
            if 'settings' in obj:
                self.settings.update(**obj['settings'])
            if self.overlay and 'overlay_bundle' in obj:
                self.overlay.from_dict(obj['overlay_bundle'])
            # 還原 GUI 狀態（若專案內有存）
            gui = obj.get("gui", {}) or {}
            # 1) 主視窗幾何
            geo_b64 = gui.get("main_window_geometry")
            if geo_b64:
                try:
                    ba = QtCore.QByteArray.fromBase64(geo_b64.encode("ascii"))
                    if not ba.isEmpty():
                        self.restoreGeometry(ba)
                except Exception:
                    pass
            # 2) 其他 GUI 設定
            proj_name = gui.get("project_name")
            if proj_name:
                self.project_name_edit.setText(proj_name)
            proj_path = gui.get("project_path")
            if proj_path:
                self.project_path_edit.setText(proj_path)
            model = gui.get("model")
            if model:
                self._set_model_name(model)
            lang = gui.get("lang")
            if lang:
                self.lang_combo.setCurrentText(lang)
            console = gui.get("console")
            if console is not None:
                self.console_chk.setChecked(bool(console))
            log_level = gui.get("log_level")
            if log_level:
                self.log_level_combo.setCurrentText(str(log_level))
            zh_mode = gui.get("zh_mode")
            if zh_mode:
                self.zh_combo.setCurrentText(zh_mode)
            device = gui.get("device")
            if device:
                if str(device).startswith("cuda"):
                    self.device_combo.setCurrentIndex(0)
                else:
                    self.device_combo.setCurrentText(str(device))
            gpu_idx = gui.get("gpu_index")
            if isinstance(gpu_idx, int) and 0 <= gpu_idx < self.gpu_combo.count():
                self.gpu_combo.setCurrentIndex(gpu_idx)
            vad = gui.get("vad")
            if vad and self.vad_combo.findText(str(vad)) >= 0:
                self.vad_combo.setCurrentText(str(vad))
            mic_gate = gui.get("mic_gate")
            if mic_gate is not None:
                try:
                    self.mic_slider.setValue(int(mic_gate))
                except Exception:
                    pass
            silence = gui.get("silence")
            if silence is not None:
                try: self.silence_spin.setValue(float(silence))
                except Exception: pass
            audio_text = gui.get("audio_device_text")
            audio_idx = gui.get("audio_device_index")
            if audio_text:
                i = self.audio_device_combo.findText(audio_text)
                if i >= 0:
                    self.audio_device_combo.setCurrentIndex(i)
                elif isinstance(audio_idx, int) and 0 <= audio_idx < self.audio_device_combo.count():
                    self.audio_device_combo.setCurrentIndex(audio_idx)
            elif isinstance(audio_idx, int) and 0 <= audio_idx < self.audio_device_combo.count():
                self.audio_device_combo.setCurrentIndex(audio_idx)
            hot_proj_name = gui.get("hot_proj_name")
            if hot_proj_name:
                self.hot_proj_name_edit.setText(hot_proj_name)
            srt_proj_name = gui.get("srt_proj_name")
            if srt_proj_name:
                self.srt_proj_name_edit.setText(srt_proj_name)
            # 3) Hotwords 路徑
            hot_p = (gui.get("hotwords_path") or "").strip()
            if hot_p:
                self.hotwords_edit.setText(hot_p)
                try:
                    self.hot_proj_name_edit.setText(Path(hot_p).parent.name)
                except Exception:
                    pass
            # 4) SRT 路徑（同步 settings 與 watcher）
            srt_p = (gui.get("srt_path") or "").strip()
            if srt_p:
                self.srt_edit.setText(srt_p)
                try:
                    self.srt_proj_name_edit.setText(Path(srt_p).parent.name)
                except Exception:
                    pass
                try:
                    self.settings.update(srt_path=Path(srt_p))
                except Exception:
                    pass
                # 依目前狀態重建 watcher（如尚未建立）
                if getattr(self, "srt_watcher", None):
                    try: self.srt_watcher.deleteLater()
                    except Exception: pass
                    self.srt_watcher = None
                self.srt_watcher = LiveSRTWatcher(
                    self.settings.srt_path, self, initial_emit=True
                )
                if self.overlay:
                    self.srt_watcher.updated.connect(self.overlay.show_entry_text)
            self.status.appendPlainText(f"[Load] {p}")
        except Exception as e:
            self.status.appendPlainText(f"[Load] 失敗: {e}")
        self._refresh_model_items()

    def schedule_autosave(self, delay_ms: int = 300):
        self._autosave_pending = True
        self._autosave_timer.start(delay_ms)

    def _do_autosave(self):
        if not self._autosave_pending:
            return
        self._autosave_pending = False
        self._write_project()

    def eventFilter(self, obj, ev):
        t = ev.type()
        # overlay 視窗移動/縮放 → autosave overlay 幾何
        if hasattr(self, 'overlay') and obj is self.overlay and t in (QtCore.QEvent.Move, QtCore.QEvent.Resize):
            self.schedule_autosave(200)
        # 主視窗移動/縮放 → autosave main_window_geometry
        if obj is self and t in (QtCore.QEvent.Move, QtCore.QEvent.Resize):
            self.schedule_autosave(300)
        return super().eventFilter(obj, ev)

    def pick_hotwords_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "選擇 Hotwords 檔案", str(ROOT_DIR),
            "Text Files (*.txt);;All Files (*)"
        )
        if path:
            self.hotwords_edit.setText(path)
            try:
                self.hot_proj_name_edit.setText(Path(path).parent.name)
            except Exception:
                pass
            self.append_log(f"已選擇熱詞檔：{path}")

    def pick_srt_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "選擇 SRT 檔案", str(self.project_dir or ROOT_DIR),
            "SRT Files (*.srt);;All Files (*)"
        )
        if path:
            self.srt_edit.setText(path)
            try:
                self.srt_proj_name_edit.setText(Path(path).parent.name)
            except Exception:
                pass
            self.settings.update(srt_path=Path(path))
            # 轉向 watcher
            if self.srt_watcher:
                try: self.srt_watcher.deleteLater()
                except Exception: pass
                self.srt_watcher = None
            self.srt_watcher = LiveSRTWatcher(
                self.settings.srt_path, self, initial_emit=True
            )
            if self.overlay:
                self.srt_watcher.updated.connect(self.overlay.show_entry_text)
            self.append_log(f"已選擇 SRT：{path}")

    def edit_srt_file(self):
        p = self.srt_edit.text().strip()
        if not p:
            return
        try:
            os.startfile(p)  # Windows 預設關聯
        except AttributeError:
            QtWidgets.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(p))
        except Exception as e:
            self.append_log(f"開啟 SRT 失敗：{e}")

    def edit_hotwords_file(self):
        p = self.hotwords_edit.text().strip()
        if not p:
            # 若尚未選擇，提供建立新檔
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "建立 Hotwords 檔案", str(ROOT_DIR / "hotwords.txt"),
                "Text Files (*.txt);;All Files (*)"
            )
            if not path:
                return
            # 建立空白檔（若不存在）
            try:
                Path(path).touch(exist_ok=True)
            except Exception as e:
                self.append_log(f"建立熱詞檔失敗：{e}")
                return
            self.hotwords_edit.setText(path)
            p = path
        try:
            # Windows：用預設關聯程式開啟
            os.startfile(p)  # type: ignore[attr-defined]
        except AttributeError:
            # 其他平台退而求其次
            QtWidgets.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(p))
        except Exception as e:
            self.append_log(f"開啟熱詞檔失敗：{e}")

    def refresh_audio_devices(self):
        self.audio_device_combo.clear()
        for idx, label, sr in list_audio_devices():
            self.audio_device_combo.addItem(label, (idx, sr))

    def detect_noise_level(self):
        dev_data = self.audio_device_combo.currentData()
        dev_id, sr = dev_data if dev_data is not None else (-1, 16000)
        duration = 15
        self.append_log("偵測噪音等級中…")
        try:
            audio = sd.rec(
                int(duration * sr), samplerate=sr, channels=1, dtype="float32", device=dev_id
            )
            sd.wait()
            rms = float(np.sqrt(np.mean(np.square(audio))))
            noise_db = 20 * np.log10(rms + 1e-12)
            if noise_db < -45:
                level = 0
            elif noise_db < -35:
                level = 1
            elif noise_db < -25:
                level = 2
            else:
                level = 3
            self.vad_combo.setCurrentText(str(level))
            self.append_log(f"背景噪音 {noise_db:.1f} dB → 選擇 VAD {level}")
        except Exception as e:
            self.append_log(f"偵測失敗：{e}")

    def _poll_mic_level(self):
        if self.proc and self.proc.poll() is None:
            return
        dev_data = self.audio_device_combo.currentData()
        dev_id, sr = dev_data if dev_data is not None else (-1, 16000)
        try:
            audio = sd.rec(int(0.05 * sr), samplerate=sr, channels=1, dtype="float32", device=dev_id)
            sd.wait()
            rms = float(np.sqrt(np.mean(np.square(audio))))
            level = min(100, int(rms * 1000))
            self.mic_level_bar.setValue(level)
        except Exception:
            pass

    def refresh_gpu_list(self):
        self.gpu_combo.clear()
        names = list_gpus()
        if names:
            for idx, name in enumerate(names):
                self.gpu_combo.addItem(name, idx)
        else:
            self.gpu_combo.addItem("無 GPU", -1)

    def _update_cuda_option(self, ready: bool):
        item = self.device_combo.model().item(0)
        if ready:
            item.setText("cuda")
            item.setForeground(QtGui.QBrush())
            self.cuda_ready = True
        else:
            item.setText("cuda (未就緒)")
            item.setForeground(QtGui.QBrush(QtGui.QColor("gray")))
            self.cuda_ready = False

    def device_changed(self, _=None):
        text = self.device_combo.currentText()
        if text.startswith("cuda") and not getattr(self, "cuda_ready", False):
            QtWidgets.QMessageBox.information(self, "提示", "請至環境設定檢查GPU加速")
            self.device_combo.setCurrentText("cpu")
            text = "cpu"
        self.gpu_combo.setEnabled(text.startswith("cuda"))
        self.schedule_autosave(300)
    def create_project(self):
        # 1) 讓使用者輸入專案名稱
        proj_name, ok = QtWidgets.QInputDialog.getText(
            self, "新建專案", "專案名稱：",
        )
        if not ok or not proj_name.strip():
            return
        proj_name = proj_name.strip()
        # 2) 選擇專案根資料夾（預設在程式目錄下的 projects）
        base_dir = (ROOT_DIR / "projects")
        base_dir.mkdir(exist_ok=True)
        target_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "選擇專案保存位置", str(base_dir)
        )
        if not target_dir:
            return
        parent_dir = Path(target_dir)
        # 2.5) 在所選位置底下建立「專案名稱」資料夾
        proj_dir = parent_dir / proj_name
        try:
            proj_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            # 已存在就沿用
            proj_dir.mkdir(parents=True, exist_ok=True)
        self.project_dir = proj_dir
        # 記住最近專案（全域 QSettings，不放在專案資料夾）
        self.settings._qs.setValue("last_project_dir", str(proj_dir))

        # 3) 建立含時間戳的檔名（放在專案名稱資料夾裡）
        ts = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd-hhmmss")
        hot_fn = f"{proj_name}_{ts}.txt"
        srt_fn = f"{proj_name}_{ts}.srt"
        hot_path = proj_dir / hot_fn
        srt_path = proj_dir / srt_fn
        try:
            hot_path.touch(exist_ok=False)
            # SRT 先建空檔，讓 watcher 有檔可監看
            srt_path.touch(exist_ok=False)
        except Exception as e:
            self.append_log(f"建立專案失敗：{e}")
            return
        self.project_name_edit.setText(proj_name)
        self.project_path_edit.setText(str(proj_dir))
        # 記錄全域最近專案（不放在專案資料夾）
        self.settings._qs.setValue("last_project_dir", str(proj_dir))
        self.hot_proj_name_edit.setText(proj_name)
        self.srt_proj_name_edit.setText(proj_name)
        self.hotwords_edit.setText(str(hot_path))
        self.srt_edit.setText(str(srt_path))
        self.settings.update(srt_path=srt_path)
        self.append_log(f"已建立專案：{proj_dir}")
        self.append_log(f"Hotwords：{hot_path}")
        self.append_log(f"SRT：{srt_path}")
        # 5) 讓現有 watcher 轉向新的 srt 路徑
        if self.srt_watcher:
            try:
                self.srt_watcher.deleteLater()
            except Exception:
                pass
            self.srt_watcher = None
        self.srt_watcher = LiveSRTWatcher(
            self.settings.srt_path, self, initial_emit=True
        )
        if self.overlay:
            self.srt_watcher.updated.connect(self.overlay.show_entry_text)
        # 聚焦：把新檔打開編輯（可選）
        # os.startfile(str(hot_path))  # 若你想自動打開
        self._load_project()
        if self.overlay:
            self.overlay.installEventFilter(self)
    def choose_project(self):
        """選擇既有專案資料夾：優先載入專案檔；缺少才以資料夾最新檔 fallback。"""
        base_dir = (ROOT_DIR / "projects")
        base_dir.mkdir(exist_ok=True)
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇專案資料夾", str(base_dir))
        if not p:
            return
        d = Path(p)
        self.project_dir = d
        # 記住最近專案
        self.project_name_edit.setText(d.name)
        self.project_path_edit.setText(str(d))
        self.settings._qs.setValue("last_project_dir", str(d))
        self.hot_proj_name_edit.setText(d.name)
        self.srt_proj_name_edit.setText(d.name)
        # 先載入專案檔；若其中記錄的路徑已失效或不在此資料夾，清空後再以資料夾最新檔補齊
        self._load_project()
        hot_p = Path(self.hotwords_edit.text().strip()) if self.hotwords_edit.text().strip() else None
        if not hot_p or not hot_p.exists() or hot_p.parent != d:
            self.hotwords_edit.clear()
        srt_p = Path(self.srt_edit.text().strip()) if self.srt_edit.text().strip() else None
        if not srt_p or not srt_p.exists() or srt_p.parent != d:
            self.srt_edit.clear()
        self._fallback_fill_paths_from_dir(d)
        if self.overlay:
            self.overlay.installEventFilter(self)
        if not self.hotwords_edit.text().strip() and not self.srt_edit.text().strip():
            self.append_log("此資料夾內未找到 .txt 或 .srt，請手動選擇。")
    def check_env(self):
        gpu_name, driver_ver = detect_gpu()
        self.refresh_gpu_list()
        if gpu_name:
            cuda_tag = recommend_cuda_version(driver_ver)
            self.cuda_tag = cuda_tag
            self.gpu_label.setText(f"GPU: {gpu_name}")
            self.driver_label.setText(f"驅動版本: {driver_ver}")
            self.append_log(f"偵測到 GPU: {gpu_name} | 驅動: {driver_ver} | 推薦 CUDA: {cuda_tag}")
        else:
            self.cuda_tag = "cpu"
            self.gpu_label.setText("GPU: 無")
            self.driver_label.setText("驅動版本: 無")
            self.append_log("未偵測到 NVIDIA GPU，將使用 CPU 模式")

        torch_ok = is_installed("torch") and is_installed("faster_whisper")
        state = "已安裝" if torch_ok else "未安裝"
        self.pkg_label.setText(f"torch/CUDA: {state} (推薦 {self.cuda_tag})")
        self._update_cuda_option(torch_ok and self.cuda_tag != "cpu")
        if torch_ok:
            self.append_log("環境已安裝，可直接啟動。")
        else:
            self.append_log("需要安裝相應套件。")

    def _run_with_progress(self, title: str, task, label: str = "處理中…"):
        dlg = QtWidgets.QProgressDialog(label, "取消", 0, 0, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumWidth(480)
        bar = QtWidgets.QProgressBar(dlg)
        bar.setRange(0, 0)
        dlg.setBar(bar)
        dlg.show()

        cancelled = {"flag": False}
        result = {"error": None}

        def on_cancel():
            cancelled["flag"] = True

        dlg.canceled.connect(on_cancel)

        def worker():
            try:
                task(lambda: cancelled["flag"])
            except Exception as e:
                result["error"] = str(e)
            finally:
                QtCore.QMetaObject.invokeMethod(dlg, "close", QtCore.Qt.QueuedConnection)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while t.is_alive():
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 100)
        if result["error"]:
            raise RuntimeError(result["error"])
        if cancelled["flag"]:
            raise RuntimeError("使用者取消")

    def uninstall_torch_cuda(self):
        try:
            self.append_log("解除安裝 torch/CUDA…")
            self._run_with_progress(
                "解除安裝 torch/CUDA",
                lambda is_cancelled: run_pip(
                    ["uninstall", "-y", "torch", "torchvision", "torchaudio"],
                    log_fn=self.append_log,
                    cancel_flag=is_cancelled,
                ),
                label="解除安裝中…",
            )
            self.append_log("解除安裝完成")
        except Exception as e:
            self.append_log(f"解除安裝失敗: {e}")
        self.check_env()

    def install_torch_cuda(self):
        try:
            self.append_log("安裝 torch/CUDA…")
            self._run_with_progress(
                "安裝 torch/CUDA",
                lambda is_cancelled: install_deps(
                    getattr(self, "cuda_tag", "cpu"),
                    log_fn=self.append_log,
                    cancel_flag=is_cancelled,
                ),
                label="安裝中…",
            )
            self.append_log("安裝完成")
        except Exception as e:
            self.append_log(f"安裝失敗: {e}")
        self.check_env()

    def _download_model_with_progress(self, repo_id: str):
        """
        使用 snapshot_download + 自訂 Qt 版 tqdm，顯示**位元組級**真實進度。
        """
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            raise RuntimeError("缺少 huggingface_hub，請先完成套件安裝")

        # 顯示快取路徑（診斷用）
        hf_cache = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        self.append_log(f"HuggingFace cache: {hf_cache}")

        # 進度對話框（byte 級）
        dlg = QtWidgets.QProgressDialog("準備下載…", "取消", 0, 100, self)
        dlg.setWindowTitle("下載語言模型")
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumWidth(480)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        bar = QtWidgets.QProgressBar(dlg)
        bar.setRange(0, 0)  # 未知進度時顯示忙碌動畫
        dlg.setBar(bar)
        dlg.show()

        progress = {"target": 0}
        timer = QtCore.QTimer(dlg)
        timer.setInterval(100)

        def animate():
            bar.setValue(progress["target"])

        timer.timeout.connect(animate)
        timer.start()

        cancelled = {"flag": False}
        result = {"error": None}  # None=成功, "cancelled"=使用者取消, 其他=錯誤訊息

        class QtTqdm(tqdm.tqdm):
            """最簡做法：直接用 tqdm 的 desc 與 n/total → 以 0–100% 顯示目前這條下載列"""

            def __init__(self, *args, **kwargs):
                # 僅顯示「檔案下載」那種 bytes 進度列；像 'Fetching N files' 一律關閉
                unit = kwargs.get("unit")
                unit_scale = kwargs.get("unit_scale", False)
                self._is_bytes = bool((unit == "B") or unit_scale)
                if not self._is_bytes:
                    # 非 bytes 進度列（例如檔案數量），不參與 GUI 統計
                    kwargs["disable"] = True
                super().__init__(*args, **kwargs)
                # 就用 0~100 來顯示百分比；每換一條 bytes 進度列（新的檔案）就重置 label
                if self._is_bytes:
                    QtCore.QMetaObject.invokeMethod(
                        bar, "setMaximum", QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(int, 100)
                    )
                    QtCore.QMetaObject.invokeMethod(
                        bar, "setValue", QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(int, 0)
                    )
                    QtCore.QMetaObject.invokeMethod(
                        dlg, "setLabelText", QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, f"{self.desc or '下載中'}: 0%")
                    )

            def update(self, n=1):
                if cancelled["flag"]:
                    raise RuntimeError("使用者取消下載")
                super().update(n)
                if self._is_bytes and self.total:
                    pct = int(min(100, max(0, round(self.n * 100.0 / float(self.total)))))
                    progress["target"] = pct
                    # 標題顯示「檔名: XX%」，對齊你終端機看到的樣式
                    label = f"{self.desc or '下載中'}: {pct}%"
                    QtCore.QMetaObject.invokeMethod(
                        dlg, "setLabelText", QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, label)
                    )

        def worker():
            try:
                # ✅ 不分平台：一律下載到專案內的 model_path/<Repo>
                local_dir = self._repo_local_dir(repo_id)
                QtCore.QMetaObject.invokeMethod(
                    self, "append_log", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, f"下載到本地模型資料夾：{local_dir}")
                )
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="model",
                    local_dir=str(local_dir),   # 新版 hub: 指定 local_dir 即為實體檔，無 symlink
                    cache_dir=str(HF_CACHE),
                    tqdm_class=QtTqdm,
                )
            except RuntimeError as e:
                # 只有在「真的中途取消」時才記為 cancelled
                if "取消下載" in str(e):
                    result["error"] = "cancelled"
                else:
                    result["error"] = str(e)
            except Exception as e:
                result["error"] = str(e)
            finally:
                # 補滿進度並關閉對話框（不會觸發 canceled）
                progress["target"] = 100
                QtCore.QMetaObject.invokeMethod(
                    bar, "setValue", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(int, 100)
                )
                QtCore.QMetaObject.invokeMethod(timer, "stop", QtCore.Qt.QueuedConnection)
                QtCore.QMetaObject.invokeMethod(dlg, "close", QtCore.Qt.QueuedConnection)

        def on_cancel():
            cancelled["flag"] = True

        dlg.canceled.connect(on_cancel)
        t = threading.Thread(target=worker, daemon=True); t.start()
        while t.is_alive():
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 100)
        # 只有在 worker 回報「真正取消」或錯誤時才丟例外
        if result["error"] == "cancelled":
            raise RuntimeError("使用者取消下載")
        elif result["error"]:
            raise RuntimeError(result["error"])
        # 成功：把進度條補滿並提示完成
        self.append_log("模型檢查/下載完成。")

    def start_clicked(self):
        self.append_log("啟動中…")
        QtWidgets.QApplication.processEvents()
        # 顯示 Hugging Face 快取（診斷用）
        hf_cache = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        self.append_log(f"HuggingFace cache: {hf_cache}")
        # 收集參數
        args = []
        # 模型：本地有就用路徑；沒有就用名稱交給 faster-whisper 下載
        model_name = self.model_combo.currentData()
        repo = MODEL_REPO_MAP.get(model_name, model_name)
        name_dir = MODEL_PATH / model_name
        repo_dir = MODEL_PATH / repo.replace("/", "--")
        if name_dir.exists() and any(name_dir.iterdir()):
            use_dir = name_dir.resolve()
            args += ["--model_dir", str(use_dir)]
            self.append_log(f"使用本地模型：{use_dir}")
        elif repo_dir.exists() and any(repo_dir.iterdir()):
            use_dir = repo_dir.resolve()
            args += ["--model_dir", str(use_dir)]
            self.append_log(f"使用本地模型：{use_dir}")
        else:
            # 無本地資料夾 → 交給 faster-whisper 以 Repo ID 取用（會走快取）
            args += ["--model_dir", repo]
            self.append_log(f"使用 Hugging Face 模型：{repo}（若已在快取將直接重用）")
        # 語言（你預設要 zh；UI 選擇一律明確傳遞，避免分支縮排導致漏傳）
        args += ["--lang", self.lang_combo.currentText()]
        if self.translate_chk.isChecked():
            args += [
                "--translate",
                "--translate_lang",
                self.translate_lang_combo.currentText().lower(),
            ]
        if self.console_chk.isChecked():
            args += ["--log", self.log_level_combo.currentText()]
        # 中文轉換（OpenCC）
        zh_mode = self.zh_combo.currentText().strip()
        if zh_mode:
            args += ["--zh", zh_mode]
            self.append_log(f"OpenCC 模式：{zh_mode}")
        # 熱詞檔
        hot_p = self.hotwords_edit.text().strip()
        if hot_p:
            args += ["--hotwords_file", hot_p]
        # 指定 SRT 輸出路徑（搭配 mWhisperSub 的 --srt_path）
        if self.settings.srt_path:
            args += ["--srt_path", str(self.settings.srt_path)]
        # GPU 選擇
        if self.device_combo.currentText().startswith("cuda"):
            gpu_idx = self.gpu_combo.currentData()
            if gpu_idx is None or gpu_idx < 0:
                gpu_idx = 0
            args += ["--gpu", str(gpu_idx)]
        else:
            args += ["--gpu", "-1"]

        # 錄音設備 ID 與採樣率
        dev_data = self.audio_device_combo.currentData()
        if dev_data is not None:
            dev_id, sr = dev_data
            args += ["--device", str(dev_id), "--sr", str(sr)]
        else:
            args += ["--sr", "auto"]

        # 傳遞 VAD 相關參數（永遠顯式傳遞，避免預設值不明）
        vad_opt = self.vad_combo.currentText()
        if vad_opt == "Auto":
            args += ["--auto-vad", "--mic-thr", f"{self.mic_slider.value() / 100.0:.3f}"]
        else:
            args += ["--vad_level", vad_opt]
        args += ["--silence", f"{self.silence_spin.value():.2f}"]
        args += ["--logprob-thr", f"{self.logprob_spin.value():.2f}"]
        args += ["--compression-ratio-thr", f"{self.comp_ratio_spin.value():.2f}"]
        temp_str = self.temp_edit.text().strip()
        if temp_str:
            args += ["--temperature", temp_str]
        # 啟動 mWhisperSub（在 Windows 上讓它進入新的 process group，之後可用 CTRL_BREAK_EVENT 做優雅關閉）
        env = os.environ.copy()
        env.setdefault("HF_HOME", str(MODEL_PATH))
        popen_kwargs = {"cwd": ROOT_DIR, "env": env}
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            if self.console_chk.isChecked():
                # 開新主控台視窗
                creationflags |= subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
            else:
                # 隱藏主控台
                creationflags |= subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            popen_kwargs["creationflags"] = creationflags
        # 非 Windows：不特別處理 console 視窗（由桌面環境決定），但仍會把 --log 傳給子程式（若有勾選）
        self.proc = subprocess.Popen([sys.executable, str(ENGINE_PY)] + args, **popen_kwargs)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # 建立內嵌 Overlay 與系統匣；主窗最小化到系統匣
        if self.overlay is None:
            self.overlay = SubtitleOverlay(self.settings)
            # 監看 overlay 幾何（Move/Resize）→ autosave
            self.overlay.installEventFilter(self)
            # 載入專案（此時 overlay 已存在，幾何/樣式可套回）
            self._load_project()
            # 若缺少 Hotwords/SRT 路徑，從專案資料夾補齊
            if self.project_dir:
                self._fallback_fill_paths_from_dir(self.project_dir)
        self.overlay.show()  # 出現到桌面
        self.overlay.show_entry_text("")
        if self.tray is None:
            self.tray = Tray(self.settings, self.overlay, parent=self, on_stop=self.stop_clicked)
        # 監看設定中的 srt_path → 更新最後一行到 overlay
        srt_path = self.settings.srt_path
        if self.srt_watcher is None:
            self.srt_watcher = LiveSRTWatcher(
                srt_path, self, initial_emit=True
            )
        else:
            # 若 path 變了，換一個新的 watcher（避免舊 watcher 卡在舊路徑）
            if Path(srt_path).resolve() != Path(self.srt_watcher.srt_path).resolve():
                try:
                    self.srt_watcher.deleteLater()
                except Exception:
                    pass
                self.srt_watcher = LiveSRTWatcher(
                    srt_path, self, initial_emit=True
                )
        # ── Fallback：當專案檔沒記錄 hotwords/srt 時，從專案資料夾補上；再不行就維持預設 ──
    def _fallback_fill_paths_from_dir(self, d: Path):
        def _latest(glob_pat: str) -> Optional[Path]:
            cands = sorted(d.glob(glob_pat), key=lambda x: x.stat().st_mtime, reverse=True)
            return cands[0] if cands else None
        # Hotwords：若尚未載入（字串為空），嘗試用資料夾最新 .txt
        if not self.hotwords_edit.text().strip():
            hot = _latest("*.txt")
            if hot:
                self.hotwords_edit.setText(str(hot))
                try:
                    self.hot_proj_name_edit.setText(hot.parent.name)
                except Exception:
                    pass
                self.append_log(f"專案熱詞：{hot}")
        # SRT：若尚未載入，嘗試用資料夾最新 .srt；再不行就維持 QSettings 預設
        if not self.srt_edit.text().strip():
            srt = _latest("*.srt")
            if srt:
                self.srt_edit.setText(str(srt))
                try:
                    self.srt_proj_name_edit.setText(srt.parent.name)
                except Exception:
                    pass
                self.settings.update(srt_path=srt)
                # 轉向 watcher（若此時就需要）
                if getattr(self, "srt_watcher", None):
                    try: self.srt_watcher.deleteLater()
                    except Exception: pass
                    self.srt_watcher = None
                self.srt_watcher = LiveSRTWatcher(
                    self.settings.srt_path, self, initial_emit=True
                )
                if self.overlay:
                    self.srt_watcher.updated.connect(self.overlay.show_entry_text)
                self.append_log(f"專案 SRT：{srt}")
        # 關鍵：不管 watcher 何時建立，都要**確保**把 updated 接到 overlay
        if self.srt_watcher and self.overlay:
            try:
                self.srt_watcher.updated.disconnect(self.overlay.show_entry_text)
            except (TypeError, RuntimeError):
                pass
            self.srt_watcher.updated.connect(self.overlay.show_entry_text)
        
    def _graceful_terminate_proc(self, timeout=5.0):
        """優雅終止 mWhisperSub；成功回傳 True。"""
        if not self.proc or self.proc.poll() is not None:
            return True
        try:
            if os.name == "nt":
                # 優雅：CTRL_BREAK_EVENT（只對 CREATE_NEW_PROCESS_GROUP 有效）
               self.proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                # 優雅：SIGINT
                self.proc.send_signal(signal.SIGINT)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            try:
                # 次強：terminate
                self.proc.terminate()
                self.proc.wait(timeout=2.0)
                return True
            except Exception:
                try:
                    # 最後手段：kill
                    self.proc.kill()
                    self.proc.wait(timeout=2.0)
                except Exception:
                    pass
        return self.proc.poll() is not None

    def stop_clicked(self):
        self.append_log("正在停止轉寫…")
        ok = self._graceful_terminate_proc()
        if ok:
            self.append_log("轉寫已停止。")
        else:
            self.append_log("轉寫停止逾時，已強制結束。")
        self.proc = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # 清一下 overlay 畫面（保留視窗，避免第二次設定要再叫出）
        if self.overlay:
            self.overlay.show_entry_text("")

    def exit_clicked(self):
        if self.proc:
            self.stop_clicked()
        if getattr(self, "tray", None):
            self.tray.hide()
        if self.overlay:
            self.overlay.close()
        QtWidgets.QApplication.quit()
    @QtCore.pyqtSlot(str)
    def append_log(self, text: str):
        """可被跨執行緒透過 invokeMethod 呼叫的安全 log 方法"""
        self.status.appendPlainText(f"@@ {text}")
        # 自動捲到最底
        self.status.verticalScrollBar().setValue(self.status.verticalScrollBar().maximum())
# ──────────── 主程式 ────────────
if __name__ == "__main__":
    def _sigint_handler(*_):
        app = QtWidgets.QApplication.instance()
        for w in app.topLevelWidgets():
            if isinstance(w, BootstrapWin):
                w.stop_clicked()
                break
        QtWidgets.QApplication.quit()
    try: signal.signal(signal.SIGINT, _sigint_handler)
    except Exception: pass
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    win = BootstrapWin()
    win.show()
    sys.exit(app.exec_())
