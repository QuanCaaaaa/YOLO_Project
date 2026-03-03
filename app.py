import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

# Cấu hình giao diện Web
st.set_page_config(page_title="Nhận Diện Người Thời Gian Thực", layout="wide")
st.title("Hệ Thống Nhận Diện Người Thời Gian Thực")

# Tải mô hình AI (Sử dụng cache để không phải load lại nhiều lần)
@st.cache_resource
def load_model():
    # Đảm bảo file best.pt nằm cùng thư mục với file app.py này
    return YOLO("best.pt")

model = load_model()

# Tạo khu vực tải file
uploaded_video = st.file_uploader("Chọn video để phân tích (MP4, AVI)", type=["mp4", "avi"])

if uploaded_video is not None:
    # Lưu video tạm thời để hệ thống có thể đọc từng khung hình
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    
    # Khởi tạo OpenCV để đọc video
    cap = cv2.VideoCapture(tfile.name)
    
    # Chia giao diện làm 2 cột: 1 bên chiếu video, 1 bên hiện thông báo
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Luồng Video Trực Tiếp")
        stframe = st.empty() # Khung chứa video
        
    with col2:
        st.markdown("### Cảnh báo")
        alert_box = st.empty() # Khung chứa cảnh báo phát hiện người
    
    # Lặp qua từng khung hình của video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Kết thúc video
            
        # Đưa khung hình vào mô hình YOLO để nhận diện
        results = model(frame)
        
        # Lấy ảnh đã được vẽ sẵn hộp giới hạn (bounding box)
        annotated_frame = results[0].plot()
        # Chuyển hệ màu để Streamlit hiển thị đúng
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Cập nhật video lên web
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)
        
        # Xử lý logic cảnh báo đơn giản: Nếu có bất kỳ đối tượng nào được nhận diện
        boxes = results[0].boxes
        if len(boxes) > 0:
            alert_box.error(f"⚠️ Phát hiện {len(boxes)} đối tượng trong khung hình!")
        else:
            alert_box.success("✅ Khung hình an toàn.")
            
    cap.release()
    st.info("Đã xử lý xong toàn bộ video.")