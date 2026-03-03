import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from ultralytics import YOLO

# 1. Cấu hình giao diện Web
st.set_page_config(page_title="Hệ Thống Nhận Diện Người", layout="wide")
st.title("Hệ Thống Nhận Diện Người Tích Hợp Đa Nguồn")

# 2. Tải mô hình AI
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
PERSON_CLASS_ID = 0 # ID của class Người

# 3. Tạo Menu chọn chế độ ở thanh bên (Sidebar)
st.sidebar.title("Cài đặt Nguồn dữ liệu")
app_mode = st.sidebar.selectbox("Chọn chế độ đầu vào:", ["Tải Video", "Tải Ảnh", "Camera Web"])

# ==========================================
# CHẾ ĐỘ 1: TẢI ẢNH TĨNH
# ==========================================
if app_mode == "Tải Ảnh":
    st.markdown("### Nhận diện trên Ảnh tĩnh")
    uploaded_image = st.file_uploader("Tải ảnh lên (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Đọc ảnh bằng thư viện PIL
        image = Image.open(uploaded_image)
        # Chuyển ảnh sang mảng ma trận để AI đọc hiểu
        img_array = np.array(image)
        
        # Đưa qua YOLO xử lý
        results = model(img_array, classes=[PERSON_CLASS_ID])
        
        # Lấy ảnh kết quả và đổi hệ màu hiển thị
        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        st.image(annotated_img, caption="Kết quả nhận diện", use_container_width=True)
        st.info(f"Phát hiện {len(results[0].boxes)} người trong ảnh.")

# ==========================================
# CHẾ ĐỘ 2: CAMERA WEB
# ==========================================
elif app_mode == "Camera Web":
    st.markdown("### Nhận diện trực tiếp qua Camera")
    st.write("Hãy cấp quyền truy cập Camera cho trình duyệt và chụp một bức ảnh.")
    
    # Kích hoạt Camera của trình duyệt
    camera_image = st.camera_input("Chụp ảnh")
    
    if camera_image is not None:
        # Xử lý tương tự như tải ảnh
        image = Image.open(camera_image)
        img_array = np.array(image)
        
        results = model(img_array, classes=[PERSON_CLASS_ID])
        
        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        st.image(annotated_img, caption="Kết quả từ Camera", use_container_width=True)
        
        boxes = results[0].boxes
        if len(boxes) > 0:
            st.error(f"⚠️ Phát hiện {len(boxes)} người!")
        else:
            st.success("✅ Không có người.")

# ==========================================
# CHẾ ĐỘ 3: TẢI VIDEO (Giữ nguyên như cũ)
# ==========================================
elif app_mode == "Tải Video":
    st.markdown("### Nhận diện trên Video")
    uploaded_video = st.file_uploader("Tải video lên (MP4, AVI)", type=["mp4", "avi"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        col1, col2 = st.columns([3, 1])
        
        with col1:
            stframe = st.empty()
            
        with col2:
            alert_box = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model(frame, classes=[PERSON_CLASS_ID])
            
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
            
            boxes = results[0].boxes
            if len(boxes) > 0:
                alert_box.error(f"⚠️ Phát hiện {len(boxes)} người!")
            else:
                alert_box.success("✅ Khung hình an toàn.")
                
        cap.release()
        st.info("Đã xử lý xong toàn bộ video.")
