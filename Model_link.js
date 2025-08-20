const { spawn } = require("child_process");

// Map simple model names to their Hugging Face repositories
const MODEL_REPO_MAP = {
  tiny: "Systran/faster-whisper-tiny",
  base: "Systran/faster-whisper-base",
  small: "Systran/faster-whisper-small",
  medium: "Systran/faster-whisper-medium",
  "large-v2": "Systran/faster-whisper-large-v2",
};

// Map translation pairs to repositories
const TRANSLATE_REPO_MAP = {
  "en-ja": "Helsinki-NLP/opus-mt-en-ja",
  "en-ko": "Helsinki-NLP/opus-mt-en-ko",
  "en-zh": "Helsinki-NLP/opus-mt-en-zh",
  "ja-en": "Helsinki-NLP/opus-mt-ja-en",
  "ko-en": "Helsinki-NLP/opus-mt-ko-en",
  "zh-en": "Helsinki-NLP/opus-mt-zh-en",
  "ja-ko": "facebook/m2m100_418M",
  "ko-ja": "facebook/m2m100_418M",
  "ja-zh": "facebook/m2m100_418M",
  "zh-ja": "facebook/m2m100_418M",
  "zh-ko": "facebook/nllb-200-distilled-600M",
  "ko-zh": "facebook/nllb-200-distilled-600M",
};

function downloadRepo(repo, localDir) {
  return new Promise((resolve, reject) => {
    const proc = spawn("huggingface-cli", [
      "download",
      repo,
      "--local-dir",
      localDir,
    ], { stdio: "inherit" });
    proc.on("close", code => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Download failed: ${repo}`));
      }
    });
  });
}

function downloadModel(name, localDir = "./models") {
  const repo = MODEL_REPO_MAP[name];
  if (!repo) {
    return Promise.reject(new Error(`Unknown model: ${name}`));
  }
  return downloadRepo(repo, localDir);
}

function downloadTranslate(src, tgt, localDir = "./models") {
  const repo = TRANSLATE_REPO_MAP[`${src}-${tgt}`];
  if (!repo) {
    return Promise.reject(new Error(`Unknown translation pair: ${src}-${tgt}`));
  }
  return downloadRepo(repo, localDir);
}

module.exports = {
  MODEL_REPO_MAP,
  TRANSLATE_REPO_MAP,
  downloadModel,
  downloadTranslate,
};
