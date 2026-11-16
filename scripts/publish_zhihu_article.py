"""把 docs/zhihu-article.html 发布到知乎「技术有限」专栏。

用法:
  python publish_zhihu.py --dry-run     # 只上传图片并生成最终 HTML，不发布
  python publish_zhihu.py --publish     # 真正发布
"""
import argparse, re, sys
from pathlib import Path

REPO = Path("/home/toby/Desktop/DSH-WS/mimic/supermimic-mini")
HTML = REPO / "docs/zhihu-article.html"
FIGS = {
    "F1": REPO/"docs/figures/f1-planning-vs-horizon.png",
    "F2": REPO/"docs/figures/f2-perturbation-control.png",
    "F3": REPO/"docs/figures/f3-K-sensitivity.png",
    "F4": REPO/"docs/figures/f4-per-step-curves.png",
    "F5": REPO/"docs/figures/f5-leaderboard.png",
}
TITLE = "从 6GB 到 16GB：SuperMimic 的机制是真的，但收益是假的"
COLUMN = "c_2105569998672548040"      # 技术有限

import hashlib, time
from PIL import Image
from zhihu_cli.client import ZhihuClient, ZHIHU_IMAGE_API
from zhihu_cli.auth import get_cookie_string, cookie_str_to_dict


def upload(c, path, source="article", attempts=8, interval=3.0, retries=4):
    """首传处理较慢; 超时后重新注册(此时 state=1)再轮询, 通常已就绪。"""
    last = None
    for k in range(retries):
        try:
            return _upload_once(c, path, source, attempts, interval)
        except RuntimeError as e:
            last = e
            print(f"   重试 {k+1}/{retries} ({e})", flush=True)
            time.sleep(4)
    raise last


def _upload_once(c, path, source, attempts, interval):
    data = path.read_bytes()
    md5 = hashlib.md5(data).hexdigest()
    r = c._session.post(ZHIHU_IMAGE_API, json={"image_hash": md5, "source": source}, timeout=60)
    r.raise_for_status()
    j = r.json()
    uf = j["upload_file"]
    if uf["state"] == 2:
        c._upload_to_oss(uf["object_key"], data, j["upload_token"])
    elif uf["state"] != 1:
        raise RuntimeError(f"unexpected image state {uf['state']}")
    iid = str(uf["image_id"])
    for _ in range(attempts):
        rr = c._session.get(f"{ZHIHU_IMAGE_API}/{iid}", timeout=60)
        d = rr.json()
        if d.get("status") == "success":
            info = {"src": d["src"], "original_src": d.get("original_src", d["src"]),
                    "watermark": d.get("watermark", "watermark"),
                    "watermark_src": d.get("watermark_src", "")}
            with Image.open(path) as im:
                info["width"], info["height"] = im.size
            return info
        time.sleep(interval)
    raise RuntimeError("image processing timed out (2min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    if not (a.dry_run or a.publish):
        sys.exit("需要 --dry-run 或 --publish")

    html = HTML.read_text()
    with ZhihuClient(cookie_str_to_dict(get_cookie_string())) as c:
        infos = {}
        for key, path in FIGS.items():
            print(f"上传 {key}: {path.name} ...", flush=True)
            infos[key] = upload(c, path)
            print(f"   -> {infos[key].get('src','')[:70]}  "
                  f"({infos[key].get('width')}x{infos[key].get('height')})", flush=True)

        def repl(m):
            key = m.group(1)
            return ZhihuClient._build_img_html([infos[key]])

        out = re.sub(r'<img [^>]*src="(F\d)"[^>]*/>', repl, html)
        missing = re.findall(r'src="(F\d)"', out)
        if missing:
            sys.exit(f"仍有未替换的占位符: {missing}")
        dst = REPO/"docs/zhihu-article-final.html"
        dst.write_text(out)
        print(f"\n最终 HTML: {len(out)} 字符, 图片 {len(infos)} 张 -> {dst}")

        if a.dry_run:
            print("dry-run 结束，未发布")
            return

        from zhihu_cli.client import ZHIHU_ZHUANLAN_API
        r = c._session.post(f"{ZHIHU_ZHUANLAN_API}/articles/drafts", json={}, timeout=60)
        did = r.json()["id"]
        print(f"草稿 {did}")
        r = c._session.patch(f"{ZHIHU_ZHUANLAN_API}/articles/{did}/draft",
                             json={"title": TITLE, "content": out}, timeout=120)
        if r.status_code not in (200, 204):
            sys.exit(f"写草稿失败 {r.status_code}: {r.text[:300]}")
        print("正文已写入草稿")
        r = c._session.put(f"{ZHIHU_ZHUANLAN_API}/articles/{did}/publish",
                           json={"column": {"id": COLUMN}, "commentPermission": "anyone"},
                           timeout=120)
        print("publish HTTP", r.status_code)
        if r.status_code != 200:
            sys.exit(f"发布失败: {r.text[:300]}")
        j = r.json()
        print("文章 ID:", j.get("id"))
        print("URL:", j.get("url") or f"https://zhuanlan.zhihu.com/p/{j.get('id')}")
        cols = [x.get("column", {}).get("title") for x in j.get("contributions", [])]
        print("所属专栏:", cols)


if __name__ == "__main__":
    main()
