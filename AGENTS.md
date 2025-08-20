# AGENTS

## Code Style
- Use Python 3.10 features allowed in project.
- Prefer double quotes for strings.

## Testing
- After modifying any Python code, run:
  ```
  python -m py_compile overlay.py srt_utils.py whisper_gui_bootstrap.py mWhisperSub.py
  pytest -q
  ```
- Ensure command exits without errors before committing.

## Commit Guidelines
- Do not create new branches.
- Keep worktree clean before finishing.
