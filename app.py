import os
import pickle
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN, fixed_image_standardization
from scipy.spatial.distance import cosine
import torch
import cv2

RECOGNITION_THRESHOLD = 0.35
DB_FILE = 'face_db.pkl'
FACE_SIZE = (160, 160)
MIN_FACE_CONFIDENCE = 0.99
REJECT_ANGLE = 20
BOX_WIDTH = 4
TEXT_SIZE = 18
FRAME_SKIP = 5
st.set_page_config(page_title="Face Recognition System", layout="wide")


@st.cache_resource
def load_models():
    mtcnn = MTCNN(
        keep_all=True,
        thresholds=[0.85, 0.95, 0.98],
        min_face_size=80,
        device='cpu'
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, resnet

mtcnn, resnet = load_models()

def load_embeddings():
    try:
        if os.path.exists(DB_FILE) and os.path.getsize(DB_FILE) > 0:
            with open(DB_FILE, 'rb') as f:
                data = pickle.load(f)
            return [np.array(e) for e in data['embeddings']], data['names']
        return [], []
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return [], []

def save_embeddings(embeddings, names):
    try:
        with open(DB_FILE, 'wb') as f:
            pickle.dump({
                'embeddings': [e.tolist() for e in embeddings],
                'names': names
            }, f)
    except Exception as e:
        st.error(f"Save failed: {str(e)}")

def validate_face_geometry(box, landmarks):
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    if abs(angle) > REJECT_ANGLE:
        return False
    
    width = box[2] - box[0]
    height = box[3] - box[1]
    aspect_ratio = width / height
    
    if aspect_ratio < 0.75 or aspect_ratio > 1.25:
        return False
    
    return True

def get_aligned_face(image, box, landmarks, prob):
    if prob < MIN_FACE_CONFIDENCE:
        return None, None
    if not validate_face_geometry(box, landmarks):
        return None, None
    
    # Crop and resize face
    face = image.crop(box).resize(FACE_SIZE)
    
    # Convert to numpy array and standardize
    face_array = np.array(face, dtype=np.float32)
    standardized_face = fixed_image_standardization(face_array)
    
    # Create display version
    display_face = Image.fromarray((standardized_face * 128 + 127.5).astype(np.uint8))
    
    return standardized_face, display_face

def get_embedding(aligned_face):
    # Convert standardized face to tensor
    img_tensor = torch.from_numpy(aligned_face).permute(2, 0, 1).float()
    
    # Generate embedding
    embedding = resnet(img_tensor.unsqueeze(0)).detach().numpy().flatten()
    
    # Normalize embedding
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    return embedding

def recognize_face(embedding, known_embeddings, known_names):
    if not known_embeddings:
        return "Unknown", 1.0
    
    best_distance = float('inf')
    best_name = "Unknown"
    
    for known_embed, name in zip(known_embeddings, known_names):
        distance = cosine(embedding, known_embed)
        if distance < best_distance:
            best_distance = distance
            best_name = name
    
    if best_distance < RECOGNITION_THRESHOLD:
        return best_name, best_distance
    return "Unknown", best_distance

def recognize_from_image(image):
    img_rgb = np.ascontiguousarray(np.array(image.convert('RGB')))
    boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)
    
    recognized_faces = []
    known_embeddings, known_names = load_embeddings()
    draw = ImageDraw.Draw(image)
    
    if boxes is not None:
        for box, prob, landmark in zip(boxes, probs, landmarks):
            standardized_face, display_face = get_aligned_face(image, box, landmark, prob)
            if standardized_face is None:
                continue

            embedding = get_embedding(standardized_face)
            name, confidence = recognize_face(embedding, known_embeddings, known_names)
            
            # Only draw boxes for recognized faces
            if name != "Unknown":
                color = 'lime'
                label = name
                
                # Draw bounding box
                draw.rectangle(box.tolist(), outline=color, width=BOX_WIDTH)
                
                # Prepare text position
                text_position = (box[0], box[1] - 35)
                
                # Draw text with shadow
                for offset in [(-1,-1), (1,1)]:
                    draw.text(
                        (text_position[0]+offset[0], text_position[1]+offset[1]),
                        label,
                        fill='black' if offset == (-1,-1) else color,
                        font=ImageFont.load_default(size=TEXT_SIZE)
                    )

            recognized_faces.append((box, name, confidence))
    
    return image, recognized_faces

def process_video(uploaded_video):
    """Process uploaded video file for face recognition"""
    # Save video to temp file
    temp_video = "temp_video.mp4"
    with open(temp_video, "wb") as f:
        f.write(uploaded_video.read())
    
    # Open video capture
    cap = cv2.VideoCapture(temp_video)
    processed_frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Skip frames for faster processing
        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue
            
        # Convert OpenCV frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Process frame
        processed_img, _ = recognize_from_image(pil_image)
        processed_frames.append(processed_img)
        
        frame_count += 1
    
    cap.release()
    os.remove(temp_video)
    return processed_frames

st.title('Face Recognition System')
st.caption("Secure biometric identification with advanced machine learning")

# Parameters Section
with st.expander("SYSTEM PARAMETERS", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Recognition Threshold**  \n`0.35`  \n*Lower values = stricter matching*")
    with col2:
        st.markdown("**Min Face Confidence**  \n`0.99`  \n*Detection quality threshold*")
    with col3:
        st.markdown("**Face Size**  \n`160×160`  \n*Standardized template size*")
    with col4:
        st.markdown("**Frame Skip**  \n`5`  \n*For video processing*")

st.divider()

# Recognition Interface
st.subheader("Recognition Interface")
input_type = st.radio("Select Input Type:", ("Image", "Video"), horizontal=True)

if input_type == "Image":
    uploaded_file = st.file_uploader("**Upload Image for Analysis**", type=["jpg", "jpeg", "png"], 
                                   help="Upload a clear frontal face image for best results")
    if uploaded_file:
        image = Image.open(uploaded_file)
        processed_img, results = recognize_from_image(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Input Image', use_container_width=True)
        with col2:
            st.image(processed_img, caption='Analysis Results', use_container_width=True)
        
        if results:
            st.subheader("Recognition Report")
            for i, (box, name, confidence) in enumerate(results):
                st.markdown(f"""
                **Face {i+1}**  
                Identity: `{name}`  
                Confidence: `{confidence:.4f}`  
                Bounding Box: `{box.tolist()}`
                """)
        else:
            st.warning("No faces meeting quality standards detected")

elif input_type == "Video":
    uploaded_video = st.file_uploader("**Upload Video for Analysis**", type=["mp4", "avi", "mov"])
    if uploaded_video:
        with st.status("Processing video...", expanded=True) as status:
            st.write("Initializing video processing")
            processed_frames = process_video(uploaded_video)
            status.update(label="Processing complete!", state="complete", expanded=False)
        
        if processed_frames:
            st.success(f"Processed {len(processed_frames)} frames")
            st.subheader("Sample Processed Frames")
            
            cols = st.columns(3)
            for idx, frame in enumerate(processed_frames[:10]):
                with cols[idx % 3]:
                    st.image(frame, caption=f"Frame {idx*FRAME_SKIP}", use_container_width=True)
            
            if st.button("Export Processed Video", help="Coming soon!"):
                st.warning("Video export feature not implemented yet")

st.divider()

# Enrollment System
st.subheader("Biometric Enrollment")
with st.form("enrollment_form"):
    cols = st.columns(2)
    with cols[0]:
        new_name = st.text_input("**Full Name**", placeholder="Enter name")
    with cols[1]:
        new_face_file = st.file_uploader("**Face Image**", type=["jpg", "jpeg", "png"], 
                                       help="High-quality frontal face image")
    
    st.markdown("**Enrollment Requirements:**")
    st.markdown("""
    - Face must be clearly visible and well-lit
    - Neutral expression, looking straight at camera
    - No face coverings, glasses may affect recognition
    """)
    
    submitted = st.form_submit_button("Enroll Identity", use_container_width=True)
    
    if submitted and new_name and new_face_file:
        face_image = Image.open(new_face_file)
        img_rgb = np.ascontiguousarray(np.array(face_image.convert('RGB')))
        boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)
        
        if boxes is None or len(boxes) != 1 or probs[0] < MIN_FACE_CONFIDENCE:
            st.error("Enrollment Failed: Strict Requirements Not Met")
        else:
            standardized_face, display_face = get_aligned_face(face_image, boxes[0], landmarks[0], probs[0])
            if standardized_face is None:
                st.error("Enrollment Failed: Invalid Face Geometry")
            else:
                embedding = get_embedding(standardized_face)
                embeddings, names = load_embeddings()
                
                distances = [cosine(embedding, e) for e in embeddings]
                if any(d < 0.25 for d in distances):
                    st.error("Biometric Conflict: Similar face already enrolled")
                else:
                    embeddings.append(embedding)
                    names.append(new_name.strip())
                    save_embeddings(embeddings, names)
                    st.success(f"Successfully Enrolled: {new_name}")
                    st.image(display_face, caption="Enrolled Face Template", width=200)

st.divider()

# Database Management
st.subheader("Database Administration")
cols = st.columns(2)
with cols[0]:
    if st.button("Initialize Database", help="WARNING: This will erase all data!"):
        try:
            if os.path.exists(DB_FILE):
                os.remove(DB_FILE)
            save_embeddings([], [])
            st.success("Database initialized successfully")
        except Exception as e:
            st.error(f"Error: {str(e)}")

with cols[1]:
    if st.button("Show Database Analysis"):
        embeddings, names = load_embeddings()
        if names:
            st.subheader("Database Insights")
            tab1, tab2 = st.tabs(["Enrolled Identities", "Similarity Matrix"])
            
            with tab1:
                unique_names = list(set(names))
                for name in unique_names:
                    count = names.count(name)
                    st.markdown(f"- **{name}** (Entries: `{count}`)")
            
            with tab2:
                similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
                for i in range(len(embeddings)):
                    for j in range(len(embeddings)):
                        similarity_matrix[i,j] = 1 - cosine(embeddings[i], embeddings[j])
                st.dataframe(similarity_matrix, use_container_width=True)
        else:
            st.info("Database is empty")