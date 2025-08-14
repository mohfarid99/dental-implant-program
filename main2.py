import sys, subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

def pick_folder():
    folder = filedialog.askdirectory(title="Choose a folder")
    if not folder: return
    lb.delete(0, tk.END)
    for p in Path(folder).rglob("*"):
        if p.is_file():
            lb.insert(tk.END, str(p.relative_to(folder)))

def run_example():
    out.delete("1.0", tk.END)
    try:
        # Cross-platform: list current directory
        if sys.platform.startswith("win"):
            cmd = ["cmd", "/c", "dir"]
        else:
            cmd = ["ls", "-la"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        out.insert(tk.END, result.stdout or result.stderr or "(no output)")
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("My Python Desktop App")
root.geometry("800x500")

top = tk.Frame(root); top.pack(fill="x", padx=8, pady=8)
tk.Button(top, text="Load Folder…", command=pick_folder).pack(side="left")
tk.Button(top, text="Run Example Command", command=run_example).pack(side="left", padx=8)

mid = tk.PanedWindow(root, sashrelief="raised"); mid.pack(fill="both", expand=True)
lb = tk.Listbox(mid); mid.add(lb)
out = tk.Text(mid, wrap="word"); mid.add(out)

root.mainloop()