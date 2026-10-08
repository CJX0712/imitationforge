"""CLI 冒烟：端到端跑通并产出 benchmark.json。作者：晨星"""

from __future__ import annotations

import json
import os
import tempfile

from cli import main as cli_main


def test_cli_runs_and_writes_json():
    out = os.path.join(tempfile.gettempdir(), "imita_bench_cli.json")
    if os.path.exists(out):
        os.remove(out)
    rc = cli_main(
        [
            "--env",
            "cartpole",
            "--method",
            "bc",
            "imitafuse",
            "--seeds",
            "7",
            "--n-eval",
            "6",
            "--out",
            out,
        ]
    )
    assert rc == 0
    assert os.path.exists(out)
    with open(out, encoding="utf-8") as f:
        bench = json.load(f)
    assert "envs" in bench and "cartpole" in bench["envs"]
    assert "bc" in bench["envs"]["cartpole"]
    assert "imitafuse" in bench["envs"]["cartpole"]
    os.remove(out)
