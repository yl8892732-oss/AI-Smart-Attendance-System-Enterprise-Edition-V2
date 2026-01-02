import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
import hashlib
from pathlib import Path


class SimpleDatabase:
    """
    Reinforced File Database: Supports atomic writes,
    image path tracking, and physical cascading deletion.
    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            self.base_dir = Path(__file__).parent
            self.data_dir = self.base_dir / "data"
        else:
            self.data_dir = Path(data_dir)

        self.employees_file = self.data_dir / "employees.json"
        self.templates_file = self.data_dir / "templates.json"
        self.attendance_file = self.data_dir / "attendance.json"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._init_files()

    def _init_files(self):
        """Initialize empty JSON files if they do not exist."""
        for file_path in [self.employees_file, self.templates_file, self.attendance_file]:
            if not file_path.exists():
                self._write_json(str(file_path), [])

    def _read_json(self, file_path: str) -> List:
        """Safe read from JSON file."""
        p = Path(file_path)
        if not p.exists(): return []
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ Data file {p.name} corrupted: {e}")
            return []

    def _write_json(self, file_path: str, data: List):
        """Atomic write using temporary file to prevent data loss."""
        p = Path(file_path)
        temp_file = p.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, p)
        except Exception as e:
            if temp_file.exists(): temp_file.unlink()
            print(f"❌ Database write failed: {e}")

    def add_employee(self, emp_id: str, name: str, position: str = "Staff", department: str = "Default") -> bool:
        """Register new employee with position and department info."""
        if not emp_id or not name: return False
        employees = self._read_json(str(self.employees_file))
        if any(emp.get("emp_id") == str(emp_id) for emp in employees): return False

        new_employee = {
            "emp_id": str(emp_id),
            "name": name,
            "department": department,
            "position": position,
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "active"
        }
        employees.append(new_employee)
        self._write_json(str(self.employees_file), employees)
        return True

    def delete_employee(self, emp_id: str) -> bool:
        """
        Permanently delete an employee.
        Includes physical removal of registration photo and face templates.
        """
        employees = self._read_json(str(self.employees_file))
        emp_id_str = str(emp_id)

        # 1. Physical removal of registration photo
        target_emp = next((e for e in employees if e.get("emp_id") == emp_id_str), None)
        if target_emp:
            # Assumes registration photo naming convention: empID_Name.jpg
            photo_name = f"{target_emp['emp_id']}_{target_emp['name']}.jpg"
            # Path relative to static/uploads
            photo_path = Path(__file__).parent.parent / "static" / "uploads" / photo_name
            if photo_path.exists():
                try:
                    os.remove(photo_path)
                except:
                    pass

        # 2. Update employee list
        initial_len = len(employees)
        employees = [e for e in employees if e.get("emp_id") != emp_id_str]

        if len(employees) < initial_len:
            self._write_json(str(self.employees_file), employees)
            # 3. Clean up biometric templates
            templates = self._read_json(str(self.templates_file))
            templates = [t for t in templates if t.get("emp_id") != emp_id_str]
            self._write_json(str(self.templates_file), templates)
            return True
        return False

    def add_face_template(self, emp_id: str, embedding_vector: List[float], source: str = "enrollment") -> bool:
        """Add face biometric feature vector."""
        templates = self._read_json(str(self.templates_file))
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        template_hash = hashlib.md5(f"{emp_id}_{timestamp}".encode()).hexdigest()[:8]

        new_template = {
            "template_id": template_hash,
            "emp_id": str(emp_id),
            "embedding_vector": embedding_vector,
            "source": source,
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_active": True
        }
        templates.append(new_template)
        self._write_json(str(self.templates_file), templates)
        return True

    def get_all_templates(self) -> List[Dict]:
        """Fetch all active face templates."""
        templates = self._read_json(str(self.templates_file))
        return [t for t in templates if t.get("is_active", True)]

    def add_attendance_record(self, emp_id: str, confidence: float, status: str, pic_path: str,
                              raw_data_info: dict) -> bool:
        """Add attendance log with snapshot path and millisecond timestamp ID."""
        attendance = self._read_json(str(self.attendance_file))
        new_record = {
            "record_id": int(time.time() * 1000),
            "emp_id": str(emp_id),
            "confidence": round(float(confidence), 4),
            "status": status,
            "pic_path": pic_path,
            "trace_info": raw_data_info,
            "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        attendance.append(new_record)
        self._write_json(str(self.attendance_file), attendance)
        return True

    def delete_attendance_record(self, record_id: int) -> bool:
        """Delete specific record and its physical snapshot image."""
        records = self._read_json(str(self.attendance_file))
        initial_len = len(records)

        for r in records:
            if r.get("record_id") == int(record_id):
                img_path = r.get("pic_path")
                if img_path:
                    full_path = Path(__file__).parent.parent / img_path
                    if full_path.exists():
                        try:
                            os.remove(full_path)
                        except:
                            pass

        records = [r for r in records if r.get("record_id") != int(record_id)]
        if len(records) < initial_len:
            self._write_json(str(self.attendance_file), records)
            return True
        return False

    def clear_all_attendance(self) -> bool:
        """Wipe all attendance records and physical snapshot images."""
        records = self._read_json(str(self.attendance_file))
        for r in records:
            img_path = r.get("pic_path")
            if img_path:
                full_path = Path(__file__).parent.parent / img_path
                if full_path.exists():
                    try:
                        os.remove(full_path)
                    except:
                        pass

        return self._write_json(str(self.attendance_file), [])

    def get_employee_by_id(self, emp_id: str) -> Optional[Dict]:
        """Fetch employee detail by ID."""
        employees = self._read_json(str(self.employees_file))
        emp_id_str = str(emp_id)
        for emp in employees:
            if emp.get("emp_id") == emp_id_str: return emp
        return None


# Global instance
db = SimpleDatabase()