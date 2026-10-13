"""Minimal notebook executor (Jupyter is not installable in the build sandbox).

Builds a standard nbformat-4 .ipynb from a cell list, executes the code cells in
one shared namespace and embeds stdout, rich DataFrame HTML and matplotlib
figures as outputs. The result opens and re-runs normally in Jupyter.
"""
import base64
import contextlib
import io
import json
import os
import sys
import traceback


def _src(text):
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def execute(cells, cwd, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = {"__name__": "__main__"}
    old = os.getcwd()
    os.chdir(cwd)
    nb_cells = []
    count = 0
    try:
        for kind, text in cells:
            if kind == "md":
                nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": _src(text)})
                continue
            count += 1
            outputs = []
            buf = io.StringIO()
            result = None
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    code = text.strip("\n")
                    lines = code.split("\n")
                    # evaluate the last line as an expression if possible (Jupyter-like display)
                    try:
                        last = compile(lines[-1], "<cell>", "eval")
                        exec(compile("\n".join(lines[:-1]), "<cell>", "exec"), ns)
                        result = eval(last, ns)
                    except SyntaxError:
                        exec(compile(code, "<cell>", "exec"), ns)
            except Exception:
                buf.write(traceback.format_exc())
                print(f"ERROR in cell {count}:\n{buf.getvalue()[-3000:]}", file=sys.stderr)
                raise
            if buf.getvalue():
                outputs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue()})
            if result is not None:
                data = {"text/plain": repr(result)}
                if hasattr(result, "to_html"):
                    data["text/html"] = result.to_html(float_format=lambda v: f"{v:.3f}", max_rows=60)
                    data["text/plain"] = result.to_string(max_rows=60)
                outputs.append({"output_type": "execute_result", "execution_count": count, "metadata": {}, "data": data})
            for num in plt.get_fignums():
                fig = plt.figure(num)
                b = io.BytesIO()
                fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
                outputs.append({"output_type": "display_data", "metadata": {},
                                "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": "<Figure>"}})
            plt.close("all")
            nb_cells.append({"cell_type": "code", "execution_count": count, "metadata": {}, "outputs": outputs,
                             "source": _src(text)})
            print(f"cell {count} ok", flush=True)
    finally:
        os.chdir(old)
    nb = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
