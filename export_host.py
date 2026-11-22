"""Build a self-contained server-side JavaScript version of the dashboard."""
import json
import sys
from pathlib import Path
from app import models, report, ROOT

target = Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
data = dict(models, report=report, html=(ROOT/"index.html").read_text(encoding="utf-8"))
target.write_text("const data="+json.dumps(data,separators=(",",":"))+";\n"+(ROOT/"worker-template.js").read_text(encoding="utf-8"), encoding="utf-8")
