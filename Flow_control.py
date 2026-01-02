import cv2
import threading
import time
import queue
from collections import deque
from datetime import datetime
import psutil
import os

class VideoStreamController:
    """
    Manages camera connectivity, frame capture threading, and flow control.
    Optimizes performance by utilizing a producer-consumer model for frame processing.
    """

    def __init__(self, camera_id=0, target_fps=30, process_every_n=5):
        """
        Initialize the video stream controller.
        Args:
            camera_id: OS index for camera (0 is usually default).
            target_fps: Desired capture rate.
            process_every_n: Sampling rate (processes 1 out of every N frames).
        """
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.process_every_n = process_every_n
        self.frame_interval = 1.0 / target_fps

        self.cap = None
        self.is_streaming = False

        # Thread-safe queue for frame delivery with overflow protection
        self.frame_queue = queue.Queue(maxsize=20)
        self.frame_counter = 0

        self.performance_stats = {
            'fps': 0,
            'frame_count': 0,
            'start_time': None
        }

        self._init_camera()

    def _init_camera(self):
        """Initializes OpenCV VideoCapture and sets hardware properties."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise ValueError(f"Unable to open camera source: {self.camera_id}")

            # Configure hardware buffer and resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            print(f"[INFO] Camera {self.camera_id} initialized successfully.")
            print(f"[INFO] Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"[INFO] Hardware FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")

        except Exception as e:
            print(f"[ERROR] Camera initialization failed: {e}")
            print("[WARN] Attempting to load fallback test video...")
            self.cap = cv2.VideoCapture('test_video.mp4')

    def start_streaming(self):
        """Launches the frame capture thread in the background."""
        if self.cap is None:
            print("[ERROR] Cannot start stream: Camera not initialized.")
            return False

        self.is_streaming = True
        self.performance_stats['start_time'] = time.time()

        # Execute frame capture in a background daemon thread
        self.stream_thread = threading.Thread(target=self._capture_frames)
        self.stream_thread.daemon = True
        self.stream_thread.start()

        print("[INFO] Video capture thread started.")
        return True

    def _capture_frames(self):
        """Producer thread: Continuously captures frames and regulates FPS."""
        last_frame_time = time.time()

        while self.is_streaming and self.cap.isOpened():
            current_time = time.time()
            elapsed = current_time - last_frame_time

            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)

            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] Failed to read frame from source.")
                time.sleep(0.1)
                continue

            self.frame_counter += 1
            self.performance_stats['frame_count'] += 1

            # Update real-time FPS metrics every 30 frames
            if self.performance_stats['frame_count'] % 30 == 0:
                elapsed_total = current_time - self.performance_stats['start_time']
                self.performance_stats['fps'] = self.performance_stats['frame_count'] / elapsed_total

            # Sample frames based on process_every_n frequency
            if self.frame_counter % self.process_every_n == 0:
                if not self.frame_queue.full():
                    self.frame_queue.put((frame.copy(), self.frame_counter))
                else:
                    # Drop stale frames to maintain low latency
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put((frame.copy(), self.frame_counter))
                    except queue.Empty:
                        pass

            last_frame_time = current_time

    def get_frame_for_processing(self):
        """Consumer method: Fetches the next sampled frame from the queue."""
        try:
            frame, frame_num = self.frame_queue.get(timeout=0.1)
            return frame, frame_num
        except queue.Empty:
            return None, 0

    def stop_streaming(self):
        """Releases hardware resources and stops capture threads."""
        self.is_streaming = False
        if self.cap:
            self.cap.release()
        print("[INFO] Video stream terminated.")

    def get_performance_stats(self):
        """Returns current telemetry: FPS, Queue depth, and Memory footprint."""
        stats = self.performance_stats.copy()
        stats['queue_size'] = self.frame_queue.qsize()
        stats['memory_usage'] = psutil.Process(os.getpid()).memory_percent()
        return stats


class AntiFraudController:
    """
    Prevents duplicate attendance and recognition errors.
    Implements cooldown periods and multi-frame consistency checks.
    """

    def __init__(self, cooldown_seconds=30, confirm_frames=5):
        """
        Args:
            cooldown_seconds: Minimum interval between successful check-ins for the same user.
            confirm_frames: Required consecutive frames of consistent recognition.
        """
        self.cooldown_seconds = cooldown_seconds
        self.confirm_frames = confirm_frames
        self.attendance_records = {}  # {person_id: last_attendance_time}
        self.recognition_history = {} # {person_id: deque of scores}
        self.confirmation_buffer = deque(maxlen=confirm_frames)

    def check_can_attendance(self, person_id, person_name, confidence_score, threshold=0.8):
        """
        Evaluates recognition results against security and cooldown rules.
        """
        current_time = datetime.now()

        # 1. Validation of confidence threshold
        if confidence_score < threshold:
            return False, f"Low confidence: {confidence_score:.2f} < {threshold}"

        # 2. Cooldown period enforcement
        if person_id in self.attendance_records:
            last_time = self.attendance_records[person_id]
            time_diff = (current_time - last_time).total_seconds()

            if time_diff < self.cooldown_seconds:
                remaining = self.cooldown_seconds - time_diff
                return False, f"Cooldown active. Please wait {remaining:.1f}s"

        # 3. Multi-frame consistency verification
        self.confirmation_buffer.append((person_id, person_name, confidence_score))

        if len(self.confirmation_buffer) < self.confirm_frames:
            return False, f"Verifying... ({len(self.confirmation_buffer)}/{self.confirm_frames})"

        buffer_ids = [item[0] for item in self.confirmation_buffer]
        buffer_names = [item[1] for item in self.confirmation_buffer]
        buffer_scores = [item[2] for item in self.confirmation_buffer]

        # Ensure all IDs in buffer match and average score meets threshold
        if len(set(buffer_ids)) == 1 and sum(buffer_scores) / len(buffer_scores) >= threshold:
            self.confirmation_buffer.clear()
            self.attendance_records[person_id] = current_time

            history_key = (person_id, person_name)
            if history_key not in self.recognition_history:
                self.recognition_history[history_key] = deque(maxlen=100)
            self.recognition_history[history_key].append({
                'time': current_time,
                'score': confidence_score
            })
            return True, f"SUCCESS: {person_id} ({person_name})"

        return False, "Inconsistent recognition across frames."

    def get_attendance_status(self, person_id, person_name):
        """Retrieves real-time cooldown status for a specific user."""
        if person_id in self.attendance_records:
            last_time = self.attendance_records[person_id]
            time_diff = (datetime.now() - last_time).total_seconds()
            return {
                'person_name': person_name,
                'last_attendance': last_time.strftime("%Y-%m-%d %H:%M:%S"),
                'seconds_since_last': time_diff,
                'in_cooldown': time_diff < self.cooldown_seconds
            }
        return None


class PerformanceOptimizer:
    """Monitors and optimizes system performance through telemetry."""

    def __init__(self):
        self.monitoring = True
        self.performance_data = {
            'frame_processing_times': deque(maxlen=100),
            'recognition_times': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100)
        }

        self.monitor_thread = threading.Thread(target=self._monitor_performance)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _monitor_performance(self):
        """Background thread for hardware utilization telemetry."""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory_percent = psutil.Process(os.getpid()).memory_percent()

            self.performance_data['cpu_usage'].append(cpu_percent)
            self.performance_data['memory_usage'].append(memory_percent)
            time.sleep(2)

    def record_processing_time(self, process_type, time_taken):
        if process_type in self.performance_data:
            self.performance_data[process_type].append(time_taken)

    def get_performance_report(self):
        """Calculates statistical metrics for performance logs."""
        report = {}
        for key, data in self.performance_data.items():
            if data:
                report[key] = {
                    'current': data[-1],
                    'avg': sum(data) / len(data),
                    'max': max(data),
                    'min': min(data),
                    'count': len(data)
                }
        report['optimization_suggestions'] = self._generate_suggestions(report)
        return report

    def _generate_suggestions(self, report):
        """Heuristic analysis of performance data to suggest optimizations."""
        suggestions = []
        if 'frame_processing_times' in report and report['frame_processing_times']['avg'] > 0.1:
            suggestions.append("High latency in frame processing; consider sampling reduction.")
        if 'memory_usage' in report and report['memory_usage']['avg'] > 70:
            suggestions.append("High memory usage; check for cache leaks.")
        if 'cpu_usage' in report and report['cpu_usage']['avg'] > 80:
            suggestions.append("CPU bottleneck; consider multi-threading or hardware acceleration.")
        return suggestions if suggestions else ["System performance within optimal parameters."]


class AttendanceFlowController:
    """Master controller coordinating video stream, recognition, and security rules."""

    def __init__(self):
        self.video_controller = VideoStreamController(
            camera_id=0,
            target_fps=30,
            process_every_n=5
        )
        self.anti_fraud = AntiFraudController(
            cooldown_seconds=30,
            confirm_frames=5
        )
        self.performance_optimizer = PerformanceOptimizer()

        self.is_running = False
        self.recognition_callback = None

    def set_recognition_callback(self, callback_func):
        """Hooks the external recognition logic into the stream loop."""
        self.recognition_callback = callback_func

    def start(self):
        if self.is_running:
            print("[WARN] System is already running.")
            return False

        if not self.video_controller.start_streaming():
            print("[ERROR] Failed to start video capture.")
            return False

        self.is_running = True
        self.main_thread = threading.Thread(target=self._main_loop)
        self.main_thread.daemon = True
        self.main_thread.start()
        print("[INFO] Attendance master loop initiated.")
        return True

    def _main_loop(self):
        """Execution loop for frame processing and recognition."""
        while self.is_running:
            try:
                frame, frame_num = self.video_controller.get_frame_for_processing()
                if frame is None:
                    time.sleep(0.01)
                    continue

                process_start = time.time()

                if self.recognition_callback:
                    result = self.recognition_callback(frame)

                    if result and 'person_id' in result:
                        person_id = result.get('person_id')
                        person_name = result.get('person_name', 'unknown')
                        confidence = result.get('confidence', 0)

                        can_attend, reason = self.anti_fraud.check_can_attendance(
                            person_id, person_name, confidence
                        )

                        log_time = datetime.now().strftime('%H:%M:%S')
                        if can_attend:
                            print(f"[{log_time}] Verified: {person_id} ({person_name})")
                        elif "Verifying" not in reason:
                            print(f"[{log_time}] Denied: {person_id} ({person_name}) - {reason}")

                process_time = time.time() - process_start
                self.performance_optimizer.record_processing_time('frame_processing_times', process_time)
                time.sleep(0.01)

            except Exception as e:
                print(f"[ERROR] Main loop exception: {e}")
                time.sleep(0.1)

    def stop(self):
        self.is_running = False
        self.video_controller.stop_streaming()
        print("[INFO] System shutdown complete.")

    def get_system_status(self):
        """Aggregates status metrics from all sub-controllers."""
        return {
            'running': self.is_running,
            'video': self.video_controller.get_performance_stats(),
            'performance': self.performance_optimizer.get_performance_report(),
            'security': {
                'cooldown': self.anti_fraud.cooldown_seconds,
                'verified_count': len(self.anti_fraud.attendance_records)
            }
        }