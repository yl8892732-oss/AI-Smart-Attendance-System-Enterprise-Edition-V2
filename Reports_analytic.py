import pandas as pd
import platform
import matplotlib

# Force 'Agg' backend for non-interactive web environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ===================== 1. Font Configuration =====================
# Optimized for cross-platform English rendering
system_os = platform.system()
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


# ===================== 2. Data Processing =====================

def read_attendance_data(db_obj):
    """Reads and merges attendance logs with employee profiles."""
    attendance_list = db_obj._read_json(str(db_obj.attendance_file))
    employee_list = db_obj._read_json(str(db_obj.employees_file))

    if not attendance_list:
        return pd.DataFrame()

    df_att = pd.DataFrame(attendance_list)
    df_emp = pd.DataFrame(employee_list)

    # Standardize ID types for merging
    df_att['emp_id'] = df_att['emp_id'].astype(str)
    df_emp['emp_id'] = df_emp['emp_id'].astype(str)

    # Merge to get employee names
    df = pd.merge(df_att, df_emp[['emp_id', 'name']], on='emp_id', how='left')
    df["check_time"] = pd.to_datetime(df["check_time"])
    df["date"] = df["check_time"].dt.date
    return df.fillna({"name": "Unknown"})


def analyze_attendance(df):
    if df.empty: return df

    late_threshold = datetime.strptime("09:00:00", "%H:%M:%S").time()

    df["status"] = df["check_time"].apply(
        lambda x: "On Time" if x.time() <= late_threshold else "Late"
    )
    df["score"] = df["status"].apply(lambda x: 1.0 if x == "On Time" else 0.5)

    return df


# ===================== 3. Visualization Logic =====================

def visualize_attendance(df):
    """Generates analytical charts. Clears/Placeholds if data is empty."""
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    plt.close('all')  # Clean up previous instances

    # Handle Empty Data Scenario (After 'Clear All History')
    if df.empty:
        _generate_placeholder_charts(static_dir)
        return

    # --- Chart 1: Individual Attendance Rate (%) ---
    plt.figure(figsize=(10, 6))
    attendance_rate = df.groupby("name")["score"].mean() * 100
    attendance_rate.plot(kind="bar", color="#4f46e5", edgecolor="none", alpha=0.8)
    plt.title("Employee Attendance Rate Analysis", fontsize=14, pad=15)
    plt.xlabel("Employee Name")
    plt.ylabel("Attendance Rate (%)")
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, "attendance_rate_bar.png"), dpi=200)
    plt.close()

    # --- Chart 2: Status Distribution (Pie Chart) ---
    plt.figure(figsize=(8, 8))
    status_count = df["status"].value_counts()
    colors = ['#10b981', '#f59e0b', '#3b82f6', '#ef4444']
    plt.pie(status_count.values, labels=status_count.index,
            autopct="%1.1f%%", startangle=140, colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    plt.title("Attendance Status Distribution", fontsize=14)
    plt.savefig(os.path.join(static_dir, "attendance_pie.png"), dpi=200)
    plt.close()

    # --- Chart 3: Daily Attendance Trend ---
    plt.figure(figsize=(12, 6))
    trend = df.groupby("date")["emp_id"].nunique()
    trend.plot(kind="line", marker='o', linewidth=2, color='#4f46e5', markersize=8)
    plt.title("Daily Attendance Trend", fontsize=14, pad=15)
    plt.xlabel("Date")
    plt.ylabel("Unique Check-ins")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, "attendance_trend.png"), dpi=200)
    plt.close()


def _generate_placeholder_charts(directory):
    """Generates placeholder images when no data is available."""
    placeholders = [
        "attendance_rate_bar.png",
        "attendance_pie.png",
        "attendance_trend.png"
    ]
    for filename in placeholders:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, 'No Data Available\nPlease complete clock-in first',
                 ha='center', va='center', fontsize=12, color='gray')
        plt.axis('off')
        plt.savefig(os.path.join(directory, filename), bbox_inches='tight')
        plt.close()