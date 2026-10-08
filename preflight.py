"""密钥自查：扫描源码与配置，拒绝提交任何疑似 secret。作者：晨星

扫描规则覆盖常见硬编码凭证：GitHub token、OpenAI/Anthropic key、
AWS access key、私钥 PEM、以及显式 password/secret/api_key/token 赋值。
CI 与本地推送前都应跑一次；发现命中即非零退出。

用法:
  python preflight.py
"""

from __future__ import annotations

import os
import re

# 疑似敏感模式（保守、低误报）
PATTERNS = [
    r"ghp_[A-Za-z0-9]{30,}",
    r"gho_[A-Za-z0-9]{30,}",
    r"ghu_[A-Za-z0-9]{30,}",
    r"ghs_[A-Za-z0-9]{30,}",
    r"ghr_[A-Za-z0-9]{30,}",
    r"sk-[A-Za-z0-9]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"(?i)password\s*=\s*['\"][^'\"]{4,}['\"]",
    r"(?i)secret\s*=\s*['\"][^'\"]{4,}['\"]",
    r"(?i)api_key\s*=\s*['\"][^'\"]{8,}['\"]",
    r"(?i)apikey\s*=\s*['\"][^'\"]{8,}['\"]",
    r"(?i)token\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
]

# 排除：缓存、构建产物、已知安全文档
EXCLUDE_DIRS = {".git", ".ruff_cache", ".pytest_cache", "__pycache__", ".github"}
EXCLUDE_FILES = {"benchmark.json", "benchmark_quick.json"}

SCAN_EXTS = (".py", ".toml", ".yml", ".yaml", ".cfg", ".txt", ".md", ".json", ".ini")


def scan(root: str):
    hits = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in EXCLUDE_DIRS]
        for f in fn:
            if f in EXCLUDE_FILES or not f.endswith(SCAN_EXTS):
                continue
            p = os.path.join(dp, f)
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read()
            except OSError:
                continue
            for pat in PATTERNS:
                for m in re.finditer(pat, txt):
                    snippet = m.group(0)
                    if len(snippet) > 14:
                        snippet = snippet[:14] + "\u2026"
                    hits.append((os.path.relpath(p, root), snippet))
    return hits


def main() -> int:
    hits = scan(".")
    if hits:
        print("\u26a0\ufe0f 密钥自查未通过：发现疑似硬编码凭证")
        for p, s in hits:
            print(f"  {p}: {s}")
        return 1
    print("\u2705 密钥自查通过：未发现硬编码密钥/敏感信息")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
