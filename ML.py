import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import base64
from datetime import datetime
import streamlit.components.v1 as components
from PIL import Image

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(
    page_title="SILVICS EYE | Mission Control",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🌲"
)

# --- 1. "CRAZY" UI STYLING (CSS INJECTION) ---
def inject_custom_css():
    st.markdown("""
        <style>
        /* BASE THEME - Dark Glassmorphism */
        .stApp {
            background-color: #050505;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(16, 185, 129, 0.1) 0%, transparent 20%),
                radial-gradient(circle at 90% 80%, rgba(59, 130, 246, 0.1) 0%, transparent 20%);
        }
        
        /* TEXT COLORS */
        h1, h2, h3, p, div { color: #e0e0e0; font-family: 'Courier New', monospace; }
        .highlight-green { color: #00FF80; text-shadow: 0 0 10px rgba(0,255,128,0.5); }
        .highlight-red { color: #FF4B4B; text-shadow: 0 0 10px rgba(255,75,75,0.5); }

        /* METRIC CARDS */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
            transition: transform 0.2s;
        }
        div[data-testid="stMetric"]:hover {
            transform: scale(1.02);
            border-color: rgba(0, 255, 128, 0.3);
        }
        div[data-testid="stMetricLabel"] { color: #888; font-size: 0.8rem; letter-spacing: 1px; }
        div[data-testid="stMetricValue"] { color: #fff; font-weight: bold; }

        /* DATAFRAME / TABLE */
        div[data-testid="stDataFrame"] {
            border: 1px solid #333;
            border-radius: 8px;
        }

        /* CUSTOM SCANNING ANIMATION */
        @keyframes scan {
            0% { top: 0%; opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { top: 100%; opacity: 0; }
        }
        .scan-line {
            position: absolute;
            width: 100%;
            height: 4px;
            background: #FF4B4B;
            box-shadow: 0 0 15px #FF4B4B;
            animation: scan 3s linear infinite;
            z-index: 99;
            pointer-events: none;
        }
        
        /* REMOVE STREAMLIT BRANDING */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

# --- 2. AI & PROCESSING LOGIC ---
class PlantationAnalyzer:
    def __init__(self):
        # Configuration for pit detection
        self.PIT_RADIUS_PX = 15  
        # HSV Thresholds for Vegetation (Calibrated for Drone Imagery)
        self.GREEN_LOWER = np.array([35, 40, 40])
        self.GREEN_UPPER = np.array([85, 255, 255])
        # HSV Thresholds for "Cleared Soil" (Reddish/Brownish)
        self.SOIL_LOWER = np.array([10, 50, 50])
        self.SOIL_UPPER = np.array([30, 255, 255])

    def load_image_from_upload(self, uploaded_file):
        """Converts uploaded streamlit file to OpenCV format"""
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        return img

    def align_images_orb(self, img_ref, img_target):
        """
        Real alignment using ORB. 
        Note: This assumes images are roughly same scale/rotation.
        """
        try:
            gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
            gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
            
            # ORB Detector
            orb = cv2.ORB_create(nfeatures=2000)
            kp1, des1 = orb.detectAndCompute(gray_ref, None)
            kp2, des2 = orb.detectAndCompute(gray_target, None)
            
            # Matching
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            matches = matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Take top 15% matches
            good_matches = matches[:int(len(matches) * 0.15)]
            
            if len(good_matches) < 4:
                return img_target # Not enough matches, return original

            points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
            points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
            
            for i, match in enumerate(good_matches):
                points1[i, :] = kp1[match.queryIdx].pt
                points2[i, :] = kp2[match.trainIdx].pt
            
            h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
            height, width, _ = img_ref.shape
            aligned_img = cv2.warpPerspective(img_target, h, (width, height))
            
            return aligned_img
        except Exception as e:
            st.error(f"Alignment Failed: {e}")
            return img_target

    def generate_mock_data(self, seed=42, n_pits=200):
        """Generates synthetic data for demo purposes"""
        np.random.seed(seed)
        w, h = 800, 600
        
        op1 = np.full((h, w, 3), (100, 120, 160), dtype=np.uint8) # BGR Brown
        noise = np.random.normal(0, 15, (h, w, 3)).astype(np.uint8)
        op1 = cv2.add(op1, noise)
        op3 = op1.copy()
        
        pits = []
        casualties = []
        
        for _ in range(n_pits):
            cx = np.random.randint(20, w-20)
            cy = np.random.randint(20, h-20)
            
            # OP1: Pit
            cv2.circle(op1, (cx, cy), 8, (60, 70, 90), -1) 
            
            # OP3: Sapling
            cv2.circle(op3, (cx, cy), 18, (120, 140, 180), -1) # Soil clearing
            
            is_dead = np.random.random() > 0.85 
            if not is_dead:
                cv2.circle(op3, (cx, cy), 6, (50, 200, 50), -1) # Green
            else:
                cv2.circle(op3, (cx, cy), 2, (40, 50, 60), -1) # Dead
                casualties.append({'id': len(casualties), 'x': cx, 'y': cy})
                
            pits.append((cx, cy))
            
        return op1, op3, pits

    def detect_pits_blob(self, img_op1):
        """Detects pits in real OP1 imagery using Blob Detection"""
        gray = cv2.cvtColor(img_op1, cv2.COLOR_BGR2GRAY)
        
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 50   # Adjust based on resolution
        params.maxArea = 500
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        
        pits = [(int(k.pt[0]), int(k.pt[1])) for k in keypoints]
        return pits

    def process_patch(self, op1, op3, pits):
        """Main Logic Pipeline"""
        hsv = cv2.cvtColor(op3, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER)
        
        results = []
        
        for idx, (cx, cy) in enumerate(pits):
            # 1. ROI Extraction (The 1m circle)
            r = 25 # Approx 1m radius in pixels (adjust per resolution)
            x1, y1 = max(0, cx-r), max(0, cy-r)
            x2, y2 = min(op3.shape[1], cx+r), min(op3.shape[0], cy+r)
            
            if x2 <= x1 or y2 <= y1: continue

            roi = mask_green[y1:y2, x1:x2]
            
            # 2. Check for Green Center
            green_pixels = cv2.countNonZero(roi)
            survival = green_pixels > 15 # Threshold
            
            status = "Alive" if survival else "Dead"
            
            # Mock GPS (Center of Debadihi VF)
            lat = 21.4930 + (cy * 0.000005)
            lng = 83.9000 + (cx * 0.000005)
            
            results.append({
                "ID": f"P-{idx:04d}",
                "Pixel_X": cx,
                "Pixel_Y": cy,
                "Latitude": lat,
                "Longitude": lng,
                "Status": status,
                "Green_Score": green_pixels
            })
            
        return pd.DataFrame(results)

# --- 3. CUSTOM JS COMPARISON SLIDER ---
def render_comparison_slider(img1, img2):
    """
    Injects a custom HTML/JS Before/After slider.
    Expects BGR numpy arrays. Resizes for performance.
    """
    # Resize for web display to avoid lag with massive drone tiffs
    display_size = (800, 600)
    img1_s = cv2.resize(img1, display_size)
    img2_s = cv2.resize(img2, display_size)

    _, buf1 = cv2.imencode('.jpg', img1_s)
    _, buf2 = cv2.imencode('.jpg', img2_s)
    b64_1 = base64.b64encode(buf1).decode('utf-8')
    b64_2 = base64.b64encode(buf2).decode('utf-8')
    
    html_code = f"""
    <div class="slider-container" style="position: relative; width: 100%; height: 500px; overflow: hidden; border-radius: 12px; border: 1px solid #444;">
        <div class="img-background" style="position: absolute; top:0; left:0; width: 100%; height: 100%; background-image: url('data:image/jpeg;base64,{b64_2}'); background-size: cover;">
            <div style="position: absolute; bottom: 10px; right: 10px; color: #00FF80; background: rgba(0,0,0,0.7); padding: 5px; font-family: monospace; font-size: 12px;">OP3: NOV 2025 (SAPLINGS)</div>
        </div>
        <div class="img-foreground" id="img-foreground" style="position: absolute; top:0; left:0; width: 50%; height: 100%; background-image: url('data:image/jpeg;base64,{b64_1}'); background-size: cover; border-right: 2px solid white; box-shadow: 5px 0 15px rgba(0,0,0,0.5);">
            <div style="position: absolute; bottom: 10px; left: 10px; color: #FFD700; background: rgba(0,0,0,0.7); padding: 5px; font-family: monospace; font-size: 12px;">OP1: MAY 2025 (PITS)</div>
        </div>
        <input type="range" min="0" max="100" value="50" id="slider" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; cursor: ew-resize;">
    </div>
    
    <script>
        const slider = document.getElementById('slider');
        const foreground = document.getElementById('img-foreground');
        slider.addEventListener('input', (e) => {{
            foreground.style.width = e.target.value + '%';
        }});
    </script>
    """
    components.html(html_code, height=520)

# --- 4. MAIN APP LOGIC ---
def main():
    inject_custom_css()
    analyzer = PlantationAnalyzer()
    
    # -- HEADER --
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("# 🛰️ SPECTRAL <span class='highlight-green'>SCOUT</span>", unsafe_allow_html=True)
        st.markdown("### AUTOMATED PLANTATION MONITORING SYSTEM | ODISHA FOREST DEPARTMENT")
    with col_h2:
        st.caption(f"SYSTEM TIME: {datetime.now().strftime('%H:%M:%S UTC')}")

    st.markdown("---")

    # -- SIDEBAR CONTROLS --
    with st.sidebar:
        st.title("⚙️ PARAMETERS")
        
        # DATA SOURCE TOGGLE
        data_source = st.radio("Data Source", ["Synthetic Demo", "Upload Local Files"])
        
        op1_img = None
        op3_img = None
        pits = []
        
        if data_source == "Upload Local Files":
            st.info("Upload imagery from your Drive download:")
            file_op1 = st.file_uploader("Upload OP1 (Pits)", type=['png', 'jpg', 'jpeg', 'tif'])
            file_op3 = st.file_uploader("Upload OP3 (Saplings)", type=['png', 'jpg', 'jpeg', 'tif'])
            
            if file_op1 and file_op3:
                op1_img = analyzer.load_image_from_upload(file_op1)
                op3_img = analyzer.load_image_from_upload(file_op3)
                
                if st.button("RUN ALIGNMENT & DETECTION"):
                    with st.spinner("Aligning Orthomosaics (ORB)..."):
                        op3_img = analyzer.align_images_orb(op1_img, op3_img)
                    with st.spinner("Detecting Pits (Blob Analysis)..."):
                        pits = analyzer.detect_pits_blob(op1_img)
                        st.success(f"Detected {len(pits)} pits.")
        
        else:
            selected_site = st.selectbox("Select Patch", ["Debadihi VF", "Benkmura VF"])
            # Generate Mock
            seed = 101 if selected_site == "Debadihi VF" else 202
            op1_img, op3_img, pits = analyzer.generate_mock_data(seed=seed, n_pits=250)

        st.markdown("---")
        st.write("Detection Thresholds")
        green_sens = st.slider("Green Sensitivity", 0.0, 1.0, 0.65)
        st.markdown("---")
        st.info("Resolution: 2.5cm/px\nSource: DJI Mavic 3M")

    # -- MAIN EXECUTION --
    if op1_img is not None and op3_img is not None:
        
        # Run Analysis
        with st.spinner("Analyzing Vegetation Health..."):
            df_results = analyzer.process_patch(op1_img, op3_img, pits)

        # -- METRICS ROW --
        if not df_results.empty:
            total_saplings = len(df_results)
            alive_count = len(df_results[df_results['Status'] == 'Alive'])
            dead_count = len(df_results[df_results['Status'] == 'Dead'])
            survival_rate = (alive_count / total_saplings) * 100 if total_saplings > 0 else 0

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("SURVIVAL RATE", f"{survival_rate:.1f}%", f"{survival_rate - 90:.1f}% vs Target")
            with m2:
                st.metric("TOTAL PLANTED", f"{total_saplings}", "Detected Pits")
            with m3:
                st.metric("CONFIRMED ALIVE", f"{alive_count}", "High Confidence")
            with m4:
                st.metric("CASUALTIES DETECTED", f"{dead_count}", "Action Required", delta_color="inverse")

            # -- VISUALIZATION --
            st.markdown("### 👁️ AERIAL RECONNAISSANCE")
            
            tab1, tab2 = st.tabs(["COMPARISON SLIDER", "CASUALTY HEATMAP"])
            
            with tab1:
                st.markdown(f"**Comparing OP1 (Pit Dug) vs OP3 (Current Status)**")
                render_comparison_slider(op1_img, op3_img)

            with tab2:
                # Draw Red Circles on OP3 for Dead saplings
                heatmap_img = op3_img.copy()
                dead_rows = df_results[df_results['Status'] == 'Dead']
                
                for _, row in dead_rows.iterrows():
                    cv2.circle(heatmap_img, (int(row['Pixel_X']), int(row['Pixel_Y'])), 25, (0, 0, 255), 2)
                    cv2.putText(heatmap_img, "DEAD", (int(row['Pixel_X'])-20, int(row['Pixel_Y'])-30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Convert BGR to RGB for streamlit display
                heatmap_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
                st.image(heatmap_rgb, use_column_width=True, caption="🔴 Red Circles indicate missing vegetation.")

            # -- DATA TABLE & EXPORT --
            st.markdown("### 📋 CASUALTY MANIFEST")
            
            col_table, col_actions = st.columns([3, 1])
            
            with col_table:
                dead_df = df_results[df_results['Status'] == 'Dead'][['ID', 'Latitude', 'Longitude', 'Green_Score']]
                st.dataframe(dead_df, use_container_width=True, height=300)

            with col_actions:
                st.markdown("#### ACTIONS")
                st.write("Export precise GPS coordinates for ground field staff.")
                
                csv = dead_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 DOWNLOAD REPORT (CSV)",
                    data=csv,
                    file_name=f'casualty_report_mission_data.csv',
                    mime='text/csv',
                )
        else:
            st.warning("No pits detected. Adjust thresholds.")
    else:
        st.info("👈 Please select a Data Source from the Sidebar to begin.")

if __name__ == "__main__":
    main()