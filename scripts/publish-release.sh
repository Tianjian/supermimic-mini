#!/usr/bin/env bash
# 一键发布 GitHub About + Release（需要 repo 权限的 PAT）
#
#   GITHUB_TOKEN=ghp_xxxxxxxx bash scripts/publish-release.sh
#
# 只做两件事：
#   1) PATCH /repos/{owner}/{repo}           -> 改 About（描述 + topics）
#   2) POST  /repos/{owner}/{repo}/releases  -> 用已存在的 v1.0.0 tag 建 Release
#
# tag 已经推上去了，所以不需要再操作 git。
set -euo pipefail

OWNER="Tianjian"
REPO="supermimic-mini"
TAG="v1.0.0"
API="https://api.github.com/repos/${OWNER}/${REPO}"

: "${GITHUB_TOKEN:?请先设置 GITHUB_TOKEN=ghp_xxx（需 repo 权限）}"
command -v curl >/dev/null || { echo "缺少 curl"; exit 1; }
command -v python3 >/dev/null || { echo "缺少 python3"; exit 1; }

cd "$(dirname "$0")/.."

# ---------- 1. About：描述 + topics ----------
read -r -d '' ABOUT_JSON <<'JSON' || true
{"description": "在 RTX 2060 6GB / RX 6800 XT 16GB 上复现 SuperMimic 的 WM-BPTT：世界模型能学会动作（capture 81%）、用于规划有效（+3.79%~+7.29%），但用于训练策略在消费级算力下无正收益。给出「算力→精度→机制」的五道检查归因。",
 "topics": ["supermimic", "world-model", "bptt", "differentiable-simulation", "model-based-reinforcement-learning", "humanoid", "unitree-g1", "mujoco", "motion-tracking", "lafan1", "reproduction-study", "negative-results", "rocm", "amd-gpu", "pytorch"],
 "homepage": ""}
JSON

echo "==> 更新 About ..."
HTTP=$(curl -sS -o /tmp/about_resp.json -w "%{http_code}" -X PATCH "$API" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -d "$ABOUT_JSON")
if [ "$HTTP" = "200" ]; then
  echo "    ✅ About 已更新（description + $(python3 -c 'import json;print(len(json.load(open("/tmp/about_resp.json")).get("topics",[])))') 个 topics）"
else
  echo "    ❌ 失败 HTTP $HTTP"; cat /tmp/about_resp.json; echo
fi

# ---------- 2. Release：正文取自 CHANGELOG 的 v1.0.0 段 ----------
echo "==> 创建 Release ${TAG} ..."
RELEASE_JSON=$(python3 - <<'PY'
import json, pathlib, re
txt = pathlib.Path("CHANGELOG.md").read_text()
m = re.search(r"^## v1\.0\.0.*?(?=^## v|\Z)", txt, re.S | re.M)
body = (m.group(0) if m else txt).replace("## v1.0.0 —— ", "## ")
print(json.dumps({
    "tag_name": "v1.0.0",
    "name": "v1.0.0 — 阶段二收口：WM-BPTT 复现的负面结果与精度归因",
    "body": body.strip() + "\n\n---\n\n完整文档见 [`docs/README.md`](docs/README.md)，"
            "核心结论见 [`docs/findings/`](docs/findings/)。\n",
    "draft": False,
    "prerelease": False,
}, ensure_ascii=False))
PY
)
HTTP=$(curl -sS -o /tmp/release_resp.json -w "%{http_code}" -X POST "${API}/releases" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -d "$RELEASE_JSON")
if [ "$HTTP" = "201" ]; then
  URL=$(python3 -c 'import json;print(json.load(open("/tmp/release_resp.json")).get("html_url",""))')
  echo "    ✅ Release 已创建：$URL"
else
  echo "    ❌ 失败 HTTP $HTTP"; cat /tmp/release_resp.json; echo
  echo "    （若提示 already_exists，说明该 tag 的 Release 已存在，直接改网页即可）"
fi

echo
echo "完成。检查：https://github.com/${OWNER}/${REPO}"
