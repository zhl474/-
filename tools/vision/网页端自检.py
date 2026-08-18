#!/home/zhl/fr3env/fr3env/bin/python
"""网页端(operator_panel)快速自检：不改任何状态，只读探测。

用法（面板已在跑时直接运行）：
    /home/zhl/fr3env/fr3env/bin/python tools/vision/网页端自检.py

判定：
    FAIL = 面板本身坏了/起不来，按结尾提示处理
    WARN = 面板正常，但某个前置条件没满足（如还没跑过识别、相机没开）
"""

# ========== 传参区 ==========
HOST = "127.0.0.1"
PORT = 8765
TIMEOUT_SEC = 5
# ===========================

import json
import urllib.error
import urllib.request

BASE = f"http://{HOST}:{PORT}"
RESULTS = []


def get(path):
    """返回 (状态码, 文本)。连接被拒/超时返回 (None, 错误信息)。"""
    try:
        with urllib.request.urlopen(BASE + path, timeout=TIMEOUT_SEC) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except Exception as exc:
        return None, str(exc)


def check(name, ok, detail="", warn_instead=False, action=""):
    level = "PASS" if ok else ("WARN" if warn_instead else "FAIL")
    RESULTS.append(level)
    mark = {"PASS": "✓", "WARN": "△", "FAIL": "✗"}[level]
    line = f"[{mark}{level}] {name}"
    if detail:
        line += f" —— {detail}"
    print(line)
    if not ok and action:
        for tip in action.splitlines():
            print(f"         → {tip}")


def main():
    print(f"目标: {BASE}\n")

    # 1. 面板存活：页面能打开
    code, _ = get("/")
    check("面板进程存活(页面)", code == 200,
          f"HTTP {code}" if code else "连接不上",
          action="桌面重击「控制台」图标重启；启动日志: ~/.local/state/single-arm-tetris/控制台启动.log；"
                 "端口被占时: ss -ltnp | grep 8765")

    # 2. 静态资源（页面白屏多半是 js 挂了）
    for path in ("/static/js/app.js", "/static/css/app.css"):
        code, _ = get(path)
        check(f"静态资源 {path.rsplit('/', 1)[-1]}", code == 200, f"HTTP {code}",
              action="重启面板；浏览器 Ctrl+Shift+R 强刷缓存")

    # 3. 状态接口 + ROS 健康
    code, text = get("/api/state")
    if code == 200:
        try:
            state = json.loads(text)
            health = state.get("health", {})
            flags = {k: health.get(k) for k in
                     ("ros_master", "camera_node", "perception_node", "control_node")}
            check("状态接口 /api/state", True, f"health={flags}")
            check("roscore 在线", flags["ros_master"] is True,
                  f"ros_master={flags['ros_master']}",
                  warn_instead=True, action="先起 roscore 再重启面板")
            for node in ("camera_node", "perception_node", "control_node"):
                if not flags.get(node):
                    check(f"节点 {node} 在线", False, "未检测到（不阻塞网页功能）",
                          warn_instead=True,
                          action="面板 01 视图启动 hardware/runtime，或按 runbook 起 launch")
        except ValueError:
            check("状态接口 /api/state", False, "返回的不是 JSON")
    else:
        check("状态接口 /api/state", False, f"HTTP {code}")

    # 4. 配置接口：能列出、能读到 V2 配置
    code, text = get("/api/config")
    if code == 200:
        try:
            files = [f.get("file_id") or f.get("id") for f in json.loads(text).get("files", [])]
            check("配置列表", "perception" in files, f"共{len(files)}个: {files}")
        except (ValueError, AttributeError):
            check("配置列表", False, "解析失败")
    else:
        check("配置列表", False, f"HTTP {code}")

    code, text = get("/api/config/perception")
    if code == 200:
        try:
            data = json.loads(text)
            body = data.get("data", data)
            mode = body.get("block_recognition", {}).get("mode")
            check("V2 配置可读", mode in ("v1", "v2", "shadow"), f"block_recognition.mode={mode}",
                  action="03 配置视图检查 perception 的 block_recognition 分组")
        except (ValueError, AttributeError):
            check("V2 配置可读", False, "解析失败")
    else:
        check("V2 配置可读", False, f"HTTP {code}")

    # 5. 图片接口：调试图没生成只算 WARN
    code, _ = get("/api/images/camera")
    check("相机预览", code == 200,
          "有画面" if code == 200 else "暂无画面(相机未开或没帧)",
          warn_instead=True, action="起相机节点后刷新 01 视图")
    for image_id, label in (("block_mask", "方块Mask"), ("template_match", "模板匹配(V2)"),
                            ("board_grid", "托盘格点")):
        code, _ = get(f"/api/images/{image_id}")
        check(f"调试图 {label}", code == 200,
              "有图" if code == 200 else "尚未生成(跑一次识别即出)",
              warn_instead=True, action="01 视图点「准备识别」后再看「识别调试图」卡片")

    # 汇总
    fail, warn = RESULTS.count("FAIL"), RESULTS.count("WARN")
    print("\n" + "=" * 46)
    if fail:
        print(f"结论: ✗ 有 {fail} 项 FAIL —— 面板有问题，按上面每条 → 提示处理")
        print("通用兜底: 杀掉面板进程后重击桌面图标")
        print("          pkill -f operator_panel/scripts/operator_panel.py")
    elif warn:
        print(f"结论: △ 面板本身正常，有 {warn} 项前置条件未满足(WARN)")
    else:
        print("结论: ✓ 网页端全部正常")
    print("改配置改坏了 → 04 只读视图/历史记录可回滚(每次保存都有历史)")


if __name__ == "__main__":
    main()
