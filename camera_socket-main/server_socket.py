import socket
import threading
import json
import base64
import numpy as np
import time
from datetime import datetime
import cv2
import logging
import sys
import os
from realtime_test_origin import detector, RealTimePoseDetector
from dataset_saver import saver


def get_external_ip():
    """探测本机对外可达 IP（供同 WiFi 客户端连接）"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 不真正发包，仅让系统选出口 IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def np_encoder(obj):
    """numpy类型序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

# ... existing code ...

class VisionAPISocketServer:
    """视觉API Socket服务端"""
    
    def __init__(self, host='0.0.0.0', port=65432):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(5)
    def start(self):
        """启动服务端"""
        ext_ip = get_external_ip()
        print(f'Server started on {self.host}:{self.port}')
        print('=' * 60)
        print(f'同 WiFi 客户端连接地址: {ext_ip}:{self.port}')
        print(f'本机所有 IP: {self._all_local_ips()}')
        print('数据集自动保存: datasets/<类别>/  (图+JSON)')
        print('=' * 60)
        # 先启动检测器（后台线程运行，不阻塞Socket）
        threading.Thread(target=detector.run, daemon=True).start()
        time.sleep(2)  # 等待检测器初始化完成

        while True:
            conn, addr = self.sock.accept()
            print(f'Connected: {addr}')
            threading.Thread(target=self.handle_client, args=(conn,)).start()

    @staticmethod
    def _all_local_ips():
        """列出本机所有 IPv4（排除回环）"""
        ips = []
        try:
            hostname = socket.gethostname()
            for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                ip = info[4][0]
                if ip not in ips and not ip.startswith("127."):
                    ips.append(ip)
        except Exception:
            pass
        return ips

    def handle_client(self, conn):
        """处理客户端请求"""
        with conn:
            while True:
                receive_time = time.time()
                receive_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                # 接收请求（以换行符结束）
                data = b''
                while True:
                    chunk = conn.recv(1024)
                    if not chunk:
                        # 连接关闭
                        return
                    data += chunk
                    if b'\n' in data:
                        break

                req = data.decode('utf-8').strip()
                if not req:
                    continue

                print(f'[{receive_datetime}] Received: {req}')

                process_start_time = time.time()
                resp = self.process_request(req)
                process_end_time = time.time()

                if resp is not None:
                    self.send_response(conn, resp, receive_time,
                                      send_time=time.time(),
                                      process_time=process_end_time - process_start_time)

    def _create_disconnected_frame(self):
        """创建相机未连接的提示帧"""
        height, width = 1080, 1920
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        text_lines = [
            "Camera Not Connected",
            "Please wait..."
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        line_height = 80
        
        total_height = len(text_lines) * line_height
        start_y = (height - total_height) // 2 + line_height
        
        for i, line in enumerate(text_lines):
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            x = (width - text_size[0]) // 2
            y = start_y + i * line_height
            
            color = (0, 0, 255) if i == 0 else (0, 255, 255)
            cv2.putText(frame, line, (x, y), font, font_scale, color, font_thickness, cv2.LINE_AA)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (20, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2, cv2.LINE_AA)
        
        return frame

    def _save_dataset(self, action, result_value):
        """把当前处理后帧连同该类别检测结果保存到 datasets/（一张图）"""
        try:
            _, processed_frame = detector.get_last_frames()
            saver.save(action, processed_frame, result_value)
        except Exception as e:
            logging.error(f"数据集保存失败: {e}")

    def process_request(self, req):
        """处理请求"""
        try:
            # 检查相机连接状态
            camera_connected = detector.camera_connected if hasattr(detector, 'camera_connected') else False
            
            # 如果相机未连接，对于需要相机的请求返回提示帧
            if not camera_connected and req in ['cross_points_without_granding', 'screw_points', 'board_angle', 'board_position', 'u_points', 'slot_points', 'get_frame']:
                logging.warning("相机未连接，返回提示帧")
                
                # 创建提示帧
                disconnected_frame = self._create_disconnected_frame()
                _, buffer = cv2.imencode('.jpg', disconnected_frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                response = {
                    'success': False,
                    'frame': frame_base64,
                    'shape': disconnected_frame.shape,
                    'camera_status': 'disconnected',
                    'message': '相机未连接，请等待'
                }
                return self.clean_for_utf8(response)

            # 相机已连接时的正常处理逻辑
            if req.startswith('save_stage:'):
                """按阶段名+序号保存当前帧，如 save_stage:pending:1"""
                parts = req.split(':')
                if len(parts) != 3 or not parts[1].strip() or not parts[2].strip():
                    return self.clean_for_utf8({
                        'success': False,
                        'error': '格式错误，应为 save_stage:<阶段名>:<序号>'
                    })
                stage, seq = parts[1].strip(), parts[2].strip()
                from realtime_test_origin import get_shared_frame
                frame = get_shared_frame()
                if frame is None:
                    return self.clean_for_utf8({
                        'success': False, 'stage': stage, 'seq': seq,
                        'error': 'No frame available'
                    })
                saved_path = saver.save_stage(stage, seq, frame)
                return self.clean_for_utf8({
                    'success': saved_path is not None,
                    'stage': stage, 'seq': seq,
                    'file': saved_path
                })

            elif req in ('take_photo', 'take photo'):
                """收到拍照指令：把当前相机帧保存到 data/manual/（命名与界面手动保存一致）"""
                from realtime_test_origin import get_shared_frame, _find_dir
                frame = get_shared_frame()
                if frame is None:
                    return self.clean_for_utf8({
                        'success': False,
                        'error': 'No frame available'
                    })
                save_dir = os.path.join(_find_dir('data'), 'manual')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"manual_{int(time.time() * 1000)}.jpg")
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
                print(f'take_photo saved: {save_path}')
                return self.clean_for_utf8({
                    'success': True,
                    'file': save_path
                })

            elif req == 'cross_points_without_granding':
                detect_result = detector.get_result()
                position_pixels = detect_result['cross']
                if len(position_pixels) > 0:
                    print('cross_points successed:', position_pixels)
                    response = {'cross_points_pixels': position_pixels}
                else:
                    print('cross_points failed: 未检测到十字交叉点')
                    response = {'cross_points_pixels': []}
                self._save_dataset('cross_points_without_granding', position_pixels)
                return self.clean_for_utf8(response)
                    
            elif req == 'screw_points':
                detect_result = detector.get_result()
                position_pixels = detect_result['stud']
                response = {'screw_points_pixels': position_pixels if len(position_pixels) > 0 else []}
                self._save_dataset('screw_points', position_pixels)
                return self.clean_for_utf8(response)

            elif req == 'board_angle':
                detect_result = detector.get_result()
                angle = detect_result['board_angle']
                response = {'board_angle': angle if angle is not None else None}
                self._save_dataset('board_angle', angle)
                return self.clean_for_utf8(response)
            elif req == 'board_position':
                detect_result = detector.get_result()
                position = detect_result['board_position']
                response = {'board_position': position if position is not None else None}
                self._save_dataset('board_position', position)
                return self.clean_for_utf8(response)
            elif req == 'u_points':
                detect_result = detector.get_result()
                u_points = detect_result['u']
                response = {'u_points': u_points if len(u_points) > 0 else []}
                self._save_dataset('u_points', u_points)
                return self.clean_for_utf8(response)
            elif req == 'slot_points':
                detect_result = detector.get_result()
                slot_points = detect_result['slot']
                response = {'slot_points': slot_points if slot_points is not None else None}
                self._save_dataset('slot_points', slot_points)
                return self.clean_for_utf8(response)
            elif req == 'detect_count':
                detect_result = detector.get_result()
                detect_count = detect_result['detect_count']
                response = {'detect_count': detect_count if detect_count is not None else None}
                self._save_dataset('detect_count', detect_count)
                return self.clean_for_utf8(response)
            elif req == 'get_frame':
                """获取当前视频帧"""
                from realtime_test_origin import get_shared_frame
                frame = get_shared_frame()
                if frame is not None:
                    # 将帧编码为JPEG格式，然后base64编码
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    response = {
                        'success': True,
                        'frame': frame_base64,
                        'shape': frame.shape
                    }
                else:
                    response = {'success': False, 'error': 'No frame available'}
                self._save_dataset('get_frame', None)
                return self.clean_for_utf8(response)
            elif req == 'model_cross_on':
                result = detector.enable_model_cross()
                return self.clean_for_utf8(result)
            elif req == 'model_cross_off':
                result = detector.disable_model_cross()
                return self.clean_for_utf8(result)
            elif req == 'model_stud_on':
                result = detector.enable_model_stud()
                return self.clean_for_utf8(result)
            elif req == 'model_stud_off':
                result = detector.disable_model_stud()
                return self.clean_for_utf8(result)
            elif req == 'model_u_on':
                result = detector.enable_model_u()
                return self.clean_for_utf8(result)
            elif req == 'model_u_off':
                result = detector.disable_model_u()
                return self.clean_for_utf8(result)
            elif req == 'model_slot_on':
                result = detector.enable_model_slot()
                return self.clean_for_utf8(result)
            elif req == 'model_slot_off':
                result = detector.disable_model_slot()
                return self.clean_for_utf8(result)
            elif req == 'model_detect_on':
                result = detector.enable_model_detect()
                return self.clean_for_utf8(result)
            elif req == 'model_detect_off':
                result = detector.disable_model_detect()
                return self.clean_for_utf8(result)
            elif req == 'model_board_on':
                result = detector.enable_model_board()
                return self.clean_for_utf8(result)
            elif req == 'model_board_off':
                result = detector.disable_model_board()
                return self.clean_for_utf8(result)
            elif req == 'model_status':
                result = detector.get_model_status()
                return self.clean_for_utf8({"success": True, "models": result})
            else:
                return self.clean_for_utf8({"status": "connected", "message": "Server is ready"})
                
        except Exception as e:
            logging.error(f"处理请求时出错: {e}", exc_info=True)
            return self.clean_for_utf8({'error': str(e)})
        
    def clean_for_utf8(self, obj):
        """清理非UTF-8字符，并将 numpy 类型统一归一化为 Python 原生类型"""
        # 优先处理 numpy 类型，避免 np.float64 等被当作 Python float 透传出去
        if isinstance(obj, np.ndarray):
            return [self.clean_for_utf8(item) for item in obj.tolist()]
        elif isinstance(obj, np.generic):
            # np.float64 / np.int32 / np.bool_ 等标量统一转原生类型
            return self.clean_for_utf8(obj.item())
        elif isinstance(obj, dict):
            return {self.clean_for_utf8(k): self.clean_for_utf8(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # tuple 也统一成 list，保证序列化协议一致（board_position 原本是 tuple）
            return [self.clean_for_utf8(item) for item in obj]
        elif isinstance(obj, str):
            return obj.encode('utf-8', errors='ignore').decode('utf-8')
        elif isinstance(obj, (int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj).encode('utf-8', errors='ignore').decode('utf-8')

    def send_response(self, conn, resp, receive_time, send_time, process_time):
        """发送响应"""
        try:
            send_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            if isinstance(resp, dict):
                resp['timestamp'] = {
                    'received': send_datetime,
                    'sent': send_datetime,
                    'time_diff_ms': round((send_time - receive_time) * 1000, 3),
                    'processing_time_ms': round(process_time * 1000, 3)
                }

            json_str = json.dumps(resp, default=np_encoder, ensure_ascii=False)
            json_bytes = json_str.encode('utf-8')

            # 先发送数据长度（4字节，网络字节序）
            # data_length = len(json_bytes)
            # conn.sendall(data_length.to_bytes(4, byteorder='big'))

            # 再发送实际数据
            conn.sendall(json_bytes)

            # # 智能打印：对于包含大量数据的响应（如图像），只打印摘要
            # if 'frame' in resp:
            #     # 这是图像数据，只打印摘要
            #     frame_size = len(resp.get('frame', ''))
            #     shape = resp.get('shape', 'unknown')
            #     print(f'[{send_datetime}] Sent: frame_data (size: {frame_size} bytes, shape: {shape}) (耗时: {process_time*1000:.3f}ms)')
            # else:
            #     # 对于其他响应，打印完整内容（但如果太长也截断）
            #     json_display = json_str if len(json_str) < 500 else json_str[:500] + '...(truncated)'
            print(f'[{send_datetime}] Sent: {json_str} (耗时: {process_time*1000:.3f}ms)')
            
        except Exception as e:
            logging.error(f"发送响应时出错: {e}", exc_info=True)
            error_msg = json.dumps({'error': 'Server error', 'message': str(e)}, ensure_ascii=False)
            conn.sendall(error_msg.encode('utf-8'))


if __name__ == '__main__':
    # 启动Socket服务端（会自动启动检测器）
    server = VisionAPISocketServer()
    server.start()