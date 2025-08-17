Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
shell.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ""& { .venv\Scripts\Activate.ps1; python whisper_gui_bootstrap.py }""", 0, False
