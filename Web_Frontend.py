import time
import os
import ssl
import cv2
import numpy as np
import base64
import re
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

# ===================== 1. Module and Config Import =====================
from config import SystemConfig as Config
from ArcFace_model.models import ArcFaceModel
from ArcFace_model.utils import set_memory_growth
from Database import db
from Face_recognizer import recognizer
from Flow_control import AntiFraudController
import Reports_analytic
from Face_processing.Face_processing import FaceProcessor

# ===================== 2. Production-Grade Logging =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, Config.LOG_DIR)
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

log_handler = RotatingFileHandler(
    os.path.join(LOG_PATH, 'sys_access.log'),
    maxBytes=10 * 1024 * 1024,
    backupCount=10
)
log_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
log_handler.setFormatter(log_formatter)
logger = logging.getLogger('attendance_v2')
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

ssl._create_default_https_context = ssl._create_unverified_context
# Set font for charts
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
set_memory_growth()

# ===================== 3. Components Initialization =====================
app = Flask(__name__)

face_handler = FaceProcessor(model_path=Config.YOLO_MODEL_PATH)
anti_fraud = AntiFraudController(
    cooldown_seconds=Config.ANTI_FRAUD_COOLDOWN,
    confirm_frames=1
)

hcy_model = ArcFaceModel(
    size=Config.IMAGE_SIZE,
    backbone_type=Config.BACKBONE,
    training=False,
    embd_shape=Config.EMBEDDING_SIZE
)
hcy_model.load_weights(Config.FACE_MODEL_PATH)

# Ensure necessary directories exist
SNAPSHOT_DIR = os.path.join(BASE_DIR, 'static', 'attendance_snapshots')
for path in [Config.UPLOAD_DIR, Config.DATA_DIR, SNAPSHOT_DIR]:
    os.makedirs(path, exist_ok=True)


# ===================== 4. Business Logic =====================

class AttendanceService:
    @staticmethod
    def get_request_info():
        return {"ip": request.remote_addr, "ua": request.headers.get('User-Agent')}

    @staticmethod
    def process_face_feature(face_img):
        face_img = face_img.astype(np.float32)
        face_img = (face_img - 127.5) / 128.0
        face_img = np.expand_dims(face_img, axis=0)
        embedding = hcy_model.predict(face_img, verbose=0)
        return embedding / np.linalg.norm(embedding)


recognizer.set_extract_feature_function(AttendanceService.process_face_feature)


def base64_to_cv2(base64_string):
    try:
        encoded = base64_string.split(",", 1)[1] if "," in base64_string else base64_string
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Base64 Decode Error: {str(e)}")
        return None


# ===================== 5. Route Logic =====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register')
def register_page():
    return render_template('register.html')


@app.route('/employees')
def employee_list():
    """Route for the Employee Management/Profiles page"""
    emps = db._read_json(str(db.employees_file))
    return render_template('employees.html', emps=emps)


@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    req_info = AttendanceService.get_request_info()
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({'status': 'error', 'msg': 'Missing image data'})

        frame = base64_to_cv2(image_data)
        face_img, _ = face_handler.process_frame_realtime(frame)

        if face_img is None:
            return jsonify({'status': 'no_face', 'msg': 'No face detected'})

        feat = AttendanceService.process_face_feature(face_img)
        emp_id, sim, recognized, info = recognizer.identify(feat.flatten().tolist())

        if recognized and float(sim) >= Config.RECOGNITION_THRESHOLD:
            name = info.get('name', 'Unknown')
            can_attend, _ = anti_fraud.check_can_attendance(emp_id, name, float(sim))

            if can_attend:
                # Capture snapshot for record evidence
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snap_{emp_id}_{timestamp}.jpg"
                save_path = os.path.join(SNAPSHOT_DIR, filename)
                cv2.imwrite(save_path, frame)

                # DB Path relative to static for web rendering
                web_pic_path = f"static/attendance_snapshots/{filename}"

                db.add_attendance_record(
                    emp_id,
                    float(sim),
                    status="success",
                    pic_path=web_pic_path,
                    raw_data_info=req_info
                )
                logger.info(f"Check-in Success: {name} ({emp_id})")
                return jsonify({'status': 'success', 'name': name, 'sim': round(float(sim), 4)})

            return jsonify({'status': 'cool_down', 'name': name, 'msg': 'Please do not check in too frequently'})

        return jsonify({'status': 'unknown', 'msg': 'Employee not found'})
    except Exception as e:
        logger.critical(f"Recognition Exception: {str(e)}")
        return jsonify({'status': 'error', 'msg': 'System Exception'})


@app.route('/api/register', methods=['POST'])
def api_register():
    req_info = AttendanceService.get_request_info()
    try:
        data = request.json
        name = data.get('name', '').strip()
        emp_id = data.get('emp_id', '').strip()
        position = data.get('position', 'Staff').strip()  # New Position field
        image_data = data.get('image')

        if not re.match(r'^[a-zA-Z0-9\s]{1,30}$', name) or not re.match(r'^[a-zA-Z0-9]{1,20}$', emp_id):
            return jsonify({'status': 'error', 'msg': 'Invalid Input Format'})

        frame = base64_to_cv2(image_data)
        face_img, _ = face_handler.process_frame_realtime(frame)
        if face_img is None:
            return jsonify({'status': 'error', 'msg': 'Low quality face capture'})

        feat = AttendanceService.process_face_feature(face_img)
        if db.add_employee(emp_id, name, position=position):
            db.add_face_template(emp_id, feat.flatten().tolist())
            recognizer.refresh_templates()
            file_path = os.path.join(Config.UPLOAD_DIR, f"{emp_id}_{name}.jpg")
            cv2.imwrite(file_path, face_img)
            logger.info(f"REGISTER: {name}({emp_id}) as {position}")
            return jsonify({'status': 'success', 'msg': 'Registration successful'})

        return jsonify({'status': 'fail', 'msg': 'Employee ID already exists'})
    except Exception as e:
        logger.error(f"Registration Error: {str(e)}")
        return jsonify({'status': 'error', 'msg': 'Registration failed'})


@app.route('/api/delete_record/<int:record_id>', methods=['POST'])
def api_delete_record(record_id):
    """API to delete a single check-in record and its physical image"""
    try:
        if db.delete_attendance_record(record_id):
            return jsonify({'status': 'success', 'msg': 'Record deleted successfully'})
        return jsonify({'status': 'error', 'msg': 'Record not found'})
    except Exception as e:
        logger.error(f"Deletion Error: {str(e)}")
        return jsonify({'status': 'error', 'msg': str(e)})


@app.route('/api/clear_all_records', methods=['POST'])
def api_clear_all_records():
    """Wipe all attendance logs and snapshots from DB and disk"""
    try:
        if db.clear_all_attendance():
            logger.info("ADMIN: All attendance history cleared.")
            return jsonify({'status': 'success', 'msg': 'All records cleared successfully'})
        return jsonify({'status': 'error', 'msg': 'Failed to clear records'})
    except Exception as e:
        logger.error(f"Clear All Error: {str(e)}")
        return jsonify({'status': 'error', 'msg': str(e)})


@app.route('/api/delete_employee/<string:emp_id>', methods=['POST'])
def api_delete_employee(emp_id):
    """API to permanently remove an employee and refresh face templates"""
    try:
        if db.delete_employee(emp_id):
            # Refresh recognizer memory so the deleted person cannot clock in anymore
            recognizer.refresh_templates()
            logger.info(f"ADMIN: Employee ID {emp_id} removed.")
            return jsonify({'status': 'success', 'msg': 'Employee deleted successfully'})
        return jsonify({'status': 'error', 'msg': 'Employee not found'})
    except Exception as e:
        logger.error(f"Employee Deletion Error: {str(e)}")
        return jsonify({'status': 'error', 'msg': str(e)})


@app.route('/report')
def report():
    try:
        search_query = request.args.get('search', '').strip()
        date_filter = request.args.get('date', '')

        df = Reports_analytic.read_attendance_data(db)
        if not df.empty:
            # 1. Filter logic
            if search_query:
                df = df[df['name'].str.contains(search_query, case=False) | df['emp_id'].str.contains(search_query)]
            if date_filter:
                df = df[df['check_time'].dt.strftime('%Y-%m-%d') == date_filter]

            # 2. Analytics and Plotting
            df = Reports_analytic.analyze_attendance(df)
            Reports_analytic.visualize_attendance(df)

            records = df.to_dict(orient='records')
            return render_template('report.html',
                                   records=records,
                                   t=int(time.time()),
                                   search=search_query,
                                   date=date_filter)

        return render_template('report.html', records=[], t=int(time.time()))
    except Exception as e:
        logger.error(f"Report Failure: {str(e)}")
        return "Reporting System Maintenance", 500


@app.route('/history')
def history():
    """Route for the History/Gallery page - """
    upload_dir = Config.UPLOAD_DIR
    records = []
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            if f.endswith(('.jpg', '.png')):
                file_path = os.path.join(upload_dir, f)
                mtime = os.path.getmtime(file_path)
                dt_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                records.append({
                    'file_name': f,
                    'time': dt_str
                })

    
    records.sort(key=lambda x: x['time'], reverse=True)
    return render_template('history.html', records=records)


if __name__ == '__main__':
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG, ssl_context='adhoc')
