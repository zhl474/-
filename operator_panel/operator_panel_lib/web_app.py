"""Flask HTTP、SSE 与固定图片白名单接口。"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import secrets

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
)

from .config_manager import (
    ConfigConflict,
    ConfigError,
    DangerousChangeRequired,
)
from .constants import DEBUG_IMAGE_FILES, PACKAGE_DIR
from .coordinator import OperationBusy, OperationRejected
from .localization_chain import build_localization_z_chain
from .process_supervisor import LAUNCH_LOG_FILES, tail_lines
from .template_size import calculate_template_size


LAUNCH_LOG_LABELS = {"hardware": "硬件", "runtime": "感知", "task": "任务执行"}


def create_app(
    coordinator, event_bus, config_manager, ros_gateway, panel_config,
    page_token=None, launch_log_dir=None,
):
    """创建 Web 应用；默认仅信任本机，通配监听时信任任意带正确端口的访问来源。"""
    token = page_token or secrets.token_urlsafe(32)
    host = str(panel_config["server"]["host"])
    port = int(panel_config["server"]["port"])
    listen_all = host == "0.0.0.0"
    allowed_origin = f"http://{host}:{port}"
    allowed_hosts = {f"{host}:{port}", host}
    app = Flask(
        __name__,
        template_folder=str(PACKAGE_DIR / "templates"),
        static_folder=str(PACKAGE_DIR / "static"),
    )
    app.config.update(
        MAX_CONTENT_LENGTH=8 * 1024 * 1024,
        OPERATOR_PAGE_TOKEN=token,
    )
    # 配置对象必须维持 YAML 原有顺序；中文接口也不转义为 \uXXXX。
    app.json.sort_keys = False
    app.json.ensure_ascii = False

    @app.before_request
    def protect_local_writes():
        host_header = request.host.split("@")[-1]
        host_ok = host_header in allowed_hosts or (
            listen_all and host_header.endswith(f":{port}")
        )
        if not host_ok:
            return jsonify({"error": "非法 Host，请通过本机控制台地址访问", "code": "invalid_host"}), 403
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            expected_origin = f"http://{host_header}" if listen_all else allowed_origin
            if request.headers.get("Origin") != expected_origin:
                return jsonify({"error": "写操作只接受本机页面 Origin", "code": "invalid_origin"}), 403
            supplied = request.headers.get("X-Operator-Token", "")
            if not supplied or not secrets.compare_digest(
                supplied.encode("utf-8"), token.encode("utf-8")
            ):
                return jsonify({"error": "页面令牌无效，请刷新控制台", "code": "invalid_token"}), 403
        return None

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(OperationBusy)
    def handle_busy(exc):
        return jsonify({"error": str(exc), "code": "operation_busy"}), 409

    @app.errorhandler(OperationRejected)
    def handle_rejected(exc):
        return jsonify({"error": str(exc), "code": "operation_rejected"}), 409

    @app.errorhandler(ConfigConflict)
    def handle_conflict(exc):
        return jsonify({"error": str(exc), "code": "revision_conflict"}), 409

    @app.errorhandler(DangerousChangeRequired)
    def handle_dangerous(exc):
        return jsonify({"error": str(exc), "code": "dangerous_confirmation_required"}), 428

    @app.errorhandler(ConfigError)
    def handle_config_error(exc):
        return jsonify({"error": str(exc), "code": "invalid_config"}), 400

    @app.errorhandler(ValueError)
    def handle_value_error(exc):
        return jsonify({"error": str(exc), "code": "invalid_request"}), 400

    @app.errorhandler(PermissionError)
    def handle_permission_error(exc):
        return jsonify({"error": str(exc), "code": "forbidden"}), 403

    @app.errorhandler(404)
    def handle_not_found(_exc):
        if request.path.startswith("/api/"):
            return jsonify({"error": "接口或资源不存在", "code": "not_found"}), 404
        return "页面不存在", 404

    @app.errorhandler(500)
    def handle_internal(exc):
        app.logger.exception("控制台接口异常", exc_info=exc)
        return jsonify({"error": "控制台内部错误，详细信息已写入日志", "code": "internal_error"}), 500

    def body():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("请求正文必须是 JSON 对象")
        return payload

    @app.get("/")
    def index():
        return render_template("index.html", page_token=token)

    @app.get("/api/state")
    def get_state():
        return jsonify(coordinator.snapshot())

    @app.get("/api/ros-system")
    def get_ros_system():
        return jsonify(ros_gateway.ros_system_snapshot())

    @app.get("/api/hardware/usb-occupancy")
    def get_usb_occupancy():
        return jsonify(coordinator.usb_occupancy())

    @app.get("/api/operations/<operation_id>")
    def get_operation(operation_id):
        operation = coordinator.operation(operation_id)
        if operation is None:
            return jsonify({"error": "操作不存在", "code": "not_found"}), 404
        return jsonify(operation)

    @app.get("/api/events")
    def events():
        try:
            last_id = int(request.headers.get("Last-Event-ID") or request.args.get("last_id") or 0)
        except ValueError:
            last_id = 0
        event_bus.publish("state", coordinator.snapshot())

        @stream_with_context
        def stream():
            current_id = last_id
            yield "retry: 1500\n\n"
            while True:
                batch = event_bus.wait_after(current_id, timeout=15.0)
                if not batch:
                    yield ": 心跳\n\n"
                    continue
                for event in batch:
                    current_id = event["id"]
                    payload = json.dumps(event["data"], ensure_ascii=False, separators=(",", ":"))
                    yield f"id: {current_id}\nevent: {event['type']}\ndata: {payload}\n\n"

        return Response(
            stream(), mimetype="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    @app.post("/api/process/hardware/start")
    def start_hardware():
        return jsonify(coordinator.start_hardware()), 202

    @app.post("/api/process/hardware/stop")
    def stop_hardware():
        return jsonify(coordinator.stop_hardware()), 202

    @app.post("/api/process/runtime/start")
    def start_runtime():
        return jsonify(coordinator.start_runtime(body().get("mode", "formal"))), 202

    @app.post("/api/process/runtime/stop")
    def stop_runtime():
        return jsonify(coordinator.stop_runtime()), 202

    @app.post("/api/task/prepare")
    def prepare_task():
        payload = body()
        return jsonify(coordinator.prepare_task(
            advanced=payload.get("advanced", False),
            place_order=payload.get("place_order", []),
        )), 202

    @app.post("/api/task/confirm")
    def confirm_task():
        return jsonify(coordinator.confirm_task())

    @app.post("/api/task/interaction/respond")
    def respond_task_interaction():
        payload = body()
        return jsonify(coordinator.respond_interaction(
            payload.get("prompt_id"),
            payload.get("choice"),
            confirm_speed_mismatch=payload.get(
                "confirm_speed_mismatch",
                False,
            ),
        ))

    @app.post("/api/task/discard")
    def discard_task():
        return jsonify(coordinator.discard_task())

    @app.post("/api/task/start")
    def start_task():
        return jsonify(coordinator.execute_task()), 202

    @app.post("/api/task/pause")
    def pause_task():
        return jsonify(coordinator.pause_task())

    @app.post("/api/task/resume")
    def resume_task():
        return jsonify(coordinator.resume_task())

    @app.post("/api/task/stop")
    def stop_execution():
        return jsonify(coordinator.stop_execution())

    @app.post("/api/task/abort")
    def abort_task():
        return jsonify(coordinator.emergency_stop()), 202

    @app.post("/api/control/stop")
    def emergency_stop():
        return jsonify(coordinator.emergency_stop()), 202

    @app.post("/api/control/clear-stop")
    def clear_stop():
        return jsonify(coordinator.clear_stop()), 202

    @app.post("/api/control/suction")
    def control_suction():
        payload = body()
        return jsonify(coordinator.control_suction(
            payload.get("action"),
            continuous=payload.get("continuous", False),
            confirm_continuous=payload.get("confirm_continuous", False),
        )), 202

    @app.post("/api/control/servo")
    def control_servo():
        payload = body()
        return jsonify(coordinator.control_servo(
            payload.get("angle_deg"),
            confirm_outside_safe=payload.get("confirm_outside_safe", False),
        )), 202

    @app.post("/api/control/servo-sweep/start")
    def start_servo_sweep():
        payload = body()
        return jsonify(coordinator.servo_sweep_start(
            payload.get("min_deg"),
            payload.get("max_deg"),
            payload.get("wait_seconds"),
            payload.get("repeat_count"),
            confirm_outside_safe=payload.get("confirm_outside_safe", False),
        )), 202

    @app.post("/api/control/servo-sweep/stop")
    def stop_servo_sweep():
        return jsonify(coordinator.servo_sweep_stop()), 202

    @app.post("/api/control/reset")
    def reset_arm():
        return jsonify(coordinator.reset_arm(
            confirmed_pose=body().get("confirmed_pose", False)
        )), 202

    @app.post("/api/control/move-relative")
    def move_relative():
        payload = body()
        return jsonify(coordinator.move_arm_relative(
            payload.get("dx"),
            payload.get("dy"),
            payload.get("dz"),
            speed=payload.get("speed"),
        )), 202

    @app.post("/api/tools/aruco-align")
    def start_aruco_align():
        payload = body()
        return jsonify(coordinator.aruco_align(
            payload.get("low_tcp_z_mm"),
            confirmed=payload.get("confirmed", False),
        )), 202

    @app.get("/api/tools/line-template-size")
    def line_template_size():
        try:
            p1 = (float(request.args["p1_x"]), float(request.args["p1_y"]))
            p2 = (float(request.args["p2_x"]), float(request.args["p2_y"]))
            short_side_mm = float(request.args.get("short_side_mm", 17.8))
            long_side_mm = float(request.args.get("long_side_mm", 78.2))
        except (KeyError, TypeError, ValueError):
            raise ValueError("P1、P2 和真实尺寸必须是数值")
        long_px, block_px, connector_px = calculate_template_size(
            p1, p2, short_side_mm=short_side_mm, long_side_mm=long_side_mm,
        )
        return jsonify({
            "long_side_px": long_px,
            "block_px": block_px,
            "connector_px": connector_px,
        })

    @app.get("/api/control/pose")
    def get_pose():
        return jsonify(coordinator.get_pose())

    @app.get("/api/camera/exposure")
    def get_camera_exposure():
        try:
            return jsonify(ros_gateway.get_exposure_state())
        except Exception as exc:
            return jsonify({
                "error": f"读取相机曝光状态失败：{exc}",
                "code": "camera_unavailable",
            }), 503

    @app.post("/api/camera/exposure")
    def set_camera_exposure():
        payload = body()
        sensor = str(payload.get("sensor", ""))
        key = str(payload.get("key", ""))
        value = payload.get("value")
        if sensor not in ("rgb", "depth"):
            raise ValueError("sensor 只能是 rgb 或 depth")
        if key not in ("auto_exposure", "exposure", "gain"):
            raise ValueError("key 只能是 auto_exposure、exposure 或 gain")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("value 必须是整数（auto_exposure 用 0/1）")
        try:
            return jsonify(ros_gateway.set_exposure_param(sensor, key, value))
        except Exception as exc:
            return jsonify({
                "error": f"设置相机参数失败：{exc}",
                "code": "camera_unavailable",
            }), 503

    @app.get("/api/camera/yolo-preview")
    def get_yolo_preview():
        return jsonify({"enabled": ros_gateway.is_yolo_preview_enabled()})

    @app.post("/api/camera/yolo-preview")
    def set_yolo_preview():
        return jsonify(ros_gateway.set_yolo_preview_enabled(bool(body().get("enabled", False))))

    @app.get("/api/config")
    def list_configs():
        return jsonify({"files": config_manager.list_configs()})

    @app.get("/api/config/<file_id>")
    def get_config(file_id):
        return jsonify(config_manager.get_config(file_id))

    @app.put("/api/config/<file_id>")
    def save_config(file_id):
        return jsonify(coordinator.save_config(file_id, body())), 202

    @app.get("/api/config/history")
    def config_history():
        file_id = request.args.get("file_id")
        limit = request.args.get("limit", 100)
        return jsonify({"history": config_manager.store.list_history(file_id, limit)})

    @app.post("/api/config/history")
    def restore_history():
        return jsonify(coordinator.restore_history(body())), 202

    @app.get("/api/presets")
    def get_presets():
        preset_id = request.args.get("preset_id")
        include_diff = request.args.get("include_diff") == "1"
        response = {"presets": config_manager.store.list_presets()}
        if preset_id and include_diff:
            response["diffs"] = config_manager.preset_diff(int(preset_id))
        return jsonify(response)

    @app.post("/api/presets")
    def post_presets():
        payload = body()
        action = payload.get("action", "save")
        if action == "save":
            return jsonify(coordinator.save_preset(
                payload.get("name"), overwrite=payload.get("overwrite", False)
            )), 202
        if action == "restore":
            return jsonify(coordinator.restore_preset(payload)), 202
        raise ValueError("预设 action 只能是 save 或 restore")

    @app.delete("/api/presets")
    def delete_preset():
        payload = body()
        return jsonify(coordinator.delete_preset(payload.get("preset_id"))), 202

    @app.get("/api/read-only-config")
    def read_only_config():
        return jsonify({"items": config_manager.inspect_read_only()})

    @app.get("/api/localization-z-chain")
    def localization_z_chain():
        # 数值由当前配置实时代入公式，只读接口，不触发任何写操作。
        return jsonify(build_localization_z_chain())

    @app.post("/api/calibration/deploy-pair")
    def deploy_calibration_pair():
        payload = body()
        return jsonify(coordinator.deploy_calibration_pair(
            payload.get("block_yaml", ""), payload.get("tray_yaml", ""),
            confirmed=payload.get("confirmed", False),
        )), 202

    @app.post("/api/calibration/deploy-hand-eye")
    def deploy_hand_eye():
        payload = body()
        return jsonify(coordinator.deploy_hand_eye(
            payload.get("matrix"), confirmed=payload.get("confirmed", False)
        )), 202

    @app.get("/api/launch-log")
    def list_launch_logs():
        files = []
        if launch_log_dir is not None:
            root = Path(launch_log_dir)
            for kind in ("hardware", "runtime", "task"):
                path = root / LAUNCH_LOG_FILES[kind]
                stat = path.stat() if path.is_file() else None
                files.append({
                    "kind": kind,
                    "label": LAUNCH_LOG_LABELS[kind],
                    "file": LAUNCH_LOG_FILES[kind],
                    "path": str(path),
                    "exists": stat is not None,
                    "size": stat.st_size if stat else 0,
                    "modified_at": (
                        datetime.fromtimestamp(stat.st_mtime).astimezone()
                        .isoformat(timespec="seconds") if stat else ""
                    ),
                })
        return jsonify({"dir": str(launch_log_dir or ""), "files": files})

    @app.get("/api/launch-log/<kind>")
    def get_launch_log(kind):
        if launch_log_dir is None or kind not in LAUNCH_LOG_LABELS:
            return jsonify({"error": "日志类别不存在", "code": "not_found"}), 404
        path = Path(launch_log_dir) / LAUNCH_LOG_FILES[kind]
        if not path.is_file():
            return jsonify({"error": "日志尚未生成", "code": "log_unavailable"}), 404
        if request.args.get("download") == "1":
            return send_file(
                str(path), as_attachment=True, mimetype="text/plain", max_age=0
            )
        try:
            lines = int(request.args.get("lines", 5000))
        except ValueError:
            raise ValueError("lines 必须是整数")
        text = tail_lines(path, max(1, min(lines, 1_000_000)))
        return Response(text, mimetype="text/plain; charset=utf-8")

    @app.get("/api/images/<image_id>")
    def get_image(image_id):
        if image_id == "camera":
            jpeg, updated_at = ros_gateway.image_bytes()
            if jpeg is None:
                return jsonify({"error": "尚未收到相机预览", "code": "image_unavailable"}), 404
            response = Response(jpeg, mimetype="image/jpeg")
            response.headers["X-Image-Updated-At"] = updated_at
            response.set_etag(hashlib.sha256(jpeg).hexdigest())
            return response
        if image_id == "yolo":
            jpeg, _summary, updated_at = ros_gateway.yolo_preview_snapshot()
            if jpeg is None:
                return jsonify({"error": "YOLO 预览尚未生成", "code": "image_unavailable"}), 404
            response = Response(jpeg, mimetype="image/jpeg")
            response.headers["X-Image-Updated-At"] = updated_at
            response.set_etag(hashlib.sha256(jpeg).hexdigest())
            return response
        if image_id not in DEBUG_IMAGE_FILES:
            return jsonify({"error": "图片 ID 不在白名单", "code": "not_found"}), 404
        debug_dir = Path(panel_config["output"]["debug_output_dir"])
        path = debug_dir / DEBUG_IMAGE_FILES[image_id]
        if not path.is_file():
            return jsonify({"error": "调试图尚未生成", "code": "image_unavailable"}), 404
        return send_file(str(path), mimetype="image/jpeg", conditional=True, max_age=0)

    @app.post("/api/system/exit")
    def exit_panel():
        return jsonify(coordinator.request_exit()), 202

    return app
