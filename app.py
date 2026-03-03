import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from ultralytics import YOLO

# 1. Cấu hình giao diện
st.set_page_config(page_title="Nhận Diện Người Đa Nguồn", layout="wide")
st.title("Phân Tích & Nhận Diện Người Thời Gian Thực")

# 2. Tải mô hình
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
PERSON_CLASS_ID = 1 # CẬP NHẬT THEO YÊU CẦU: Class ID là 1

# 3. Sidebar điều hướng
st.sidebar.title("Điều hướng hệ thống")
app_mode = st.sidebar.radio(
    "Vui lòng chọn nguồn dữ liệu đầu vào:",
    ("Tải Ảnh Tĩnh", "Tải Video", "Camera Web") # Thay đổi từ selectbox sang radio để dễ nhìn
)

st.write(f"Đang sử dụng chế độ: **{app_mode}**")
st.markdown("---")

# ==========================================
# CHẾ ĐỘ 1: TẢI ẢNH TĨNH
# ==========================================
if app_mode == "Tải Ảnh Tĩnh":
    st.markdown("### Nhận diện trên Ảnh Tĩnh")
    # THÊM KEY = 'image_uploader' ĐỂ TÁCH BIỆT VỚI VIDEO
    uploaded_image = st.file_uploader("Chọn một bức ảnh (JPG, PNG)", type=["jpg", "jpeg", "png"], key="image_uploader")
    
    if uploaded_image is not None:
        try:
            # Đọc file ảnh an toàn
            image = Image.open(uploaded_image)
            img_array = np.array(image)
            
            # Xử lý nếu ảnh có kênh alpha (RGBA) từ PNG
            if img_array.shape[-1] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
            with st.spinner('Đang phân tích ảnh...'):
                results = model(img_array, classes=[PERSON_CLASS_ID])
                
                annotated_img = results[0].plot()
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                st.image(annotated_img, caption="Kết quả Phân Tích", use_container_width=True)
                
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.success(f"✅ Đã tìm thấy {len(boxes)} người trong khung hình.")
                else:
                    st.warning("⚠️ Không tìm thấy người nào.")
        except Exception as e:
            st.error(f"Có lỗi khi đọc ảnh: {e}")

# ==========================================
# CHẾ ĐỘ 2: TẢI VIDEO
# ==========================================
elif app_mode == "Tải Video":
    st.markdown("### Nhận diện trên Video")
    # THÊM KEY = 'video_uploader'
    uploaded_video = st.file_uploader("Chọn một đoạn video (MP4, AVI)", type=["mp4", "avi"], key="video_uploader")

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            stframe = st.empty()
        with col2:
            alert_box = st.empty()
            
        stop_button = st.button("Dừng phát video")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model(frame, classes=[PERSON_CLASS_ID])
            
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
            
            boxes = results[0].boxes
            if len(boxes) > 0:
                alert_box.error(f"🚨 CẢNH BÁO: {len(boxes)} người!")
            else:
                alert_box.success("✅ Khu vực an toàn.")
                
        cap.release()
        st.info("Luồng video đã kết thúc.")

# ==========================================
# CHẾ ĐỘ 3: CAMERA WEB (CHỤP ẢNH)
# ==========================================
elif app_mode == "Camera Web":
    st.markdown("### Quét qua Camera")
    st.info("Vui lòng cho phép trình duyệt truy cập Camera. Khung camera sẽ hiện bên dưới.")
    
    # Kích hoạt camera
    camera_image = st.camera_input("Bấm để chụp ảnh phân tích", key="camera_input")
    
    if camera_image is not None:
        try:
            image = Image.open(camera_image)
            img_array = np.array(image)
            
            with st.spinner('Đang phân tích hình ảnh từ Camera...'):
                results = model(img_array, classes=[PERSON_CLASS_ID])
                
                annotated_img = results[0].plot()
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                st.image(annotated_img, caption="Kết quả từ hệ thống", use_container_width=True)
                
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.error(f"🚨 Phát hiện {len(boxes)} đối tượng tình nghi!")
                else:
                    st.success("✅ Tầm nhìn quang đãng.")
        except Exception as e:
            st.error(f"Có lỗi khi xử lý camera: {e}")
