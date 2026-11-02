# 本地运行与测试

双击本目录的 `start.bat`，等待浏览器打开 <http://127.0.0.1:5010>。
允许浏览器使用摄像头；关闭启动窗口或双击 `stop.bat` 可停止服务。
如果服务已经在后台运行，再次双击 `start.bat` 会打开现有页面。
本地 localhost 的 HTTP 可以使用摄像头，不需要自签名 HTTPS 证书。

1. 打开 Register，输入员工编号、姓名、职位并正对摄像头注册。
2. 打开 Clock In，同一人再次面对镜头，识别成功后生成打卡记录。
3. 在 Employees 查看档案，在 Reports 查看记录、筛选与图表。
4. 连续打卡有 30 秒冷却；测试完成后可删除自己建立的测试档案和记录。

## 这次补齐了什么

现已配置独立 `.venv`、真实人脸检测与五点对齐、公开预训练模型、
一键启动入口，并修复路径、空数据图表、清空返回值及输入错误处理。

**默认本地演示后端为 OpenCV YuNet + SFace。**
这些模型从 OpenCV 官方 Zoo 下载，并对照 Git LFS 的 SHA-256 与文件大小校验。
模型来源、哈希和各自许可保存在 `models/SOURCES.txt` 和 `models/LICENSE-*.txt`。
SFace 使用 128 维特征；默认余弦阈值 0.363 来自 OpenCV 官方示例，
实际部署前应使用自己的验证集评估阈值。本地测试未实现活体检测。

ArcFace 训练基础为 [peteryuX/arcface-tf2 的指定提交](https://github.com/peteryuX/arcface-tf2/commit/d0d9fc7ada4701001d7c5b233b820d8ee9d35b52)，
相关模型源码位于 `ArcFace_model/`。不同模型的特征不能混用，切换后应重新注册。

## 环境重建

使用 Python 3.12 创建虚拟环境，再执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_models.py
.\start.bat
```

所有检测与识别均在本机执行，摄像头图像不会上传到外部识别服务。
当前服务只监听 `127.0.0.1`；员工照片、特征与打卡记录保存在本地，
并已通过 `.gitignore` 排除，以便后续整理 GitHub 展示内容。

## 已完成的验证

使用 OpenCV 的公开样例图片与隔离的临时数据库，完成了以下真实模型测试：

- 首页、注册、员工、报表、图库页面正常返回，空数据图表正常生成。
- 错误请求、无脸图片、多人人脸图片得到明确处理。
- 注册、重复编号拒绝、同一人经亮度调整后识别成功、不同人拒绝匹配。
- 30 秒重复打卡冷却、记录与照片生成、报表筛选、记录和档案删除、关联图片清理。
- 删除全部员工后，历史考勤记录仍能生成报表。

这些验证没有打开你的摄像头；请在浏览器中完成自己的摄像头测试。
