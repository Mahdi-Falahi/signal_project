import cv2
import numpy as np
from ultralytics import YOLO
import time

class AdvancedTracker:
    """
    A sophisticated object tracker that uses YOLO for initial detection and
    a combination of a correlation filter and a Kalman filter for robust tracking.
    """

    def __init__(self, yolo_model_path='yolo11n.pt', target_class="person"):
        # 1. INITIALIZE MODELS AND PARAMETERS
        self.model = YOLO(yolo_model_path)
        self.target_class = target_class

        # 2. TRACKING STATE
        self.tracking = False
        self.current_pos = (0, 0)
        self.current_size = (0, 0)
        self.fixed_roi_size = (64, 128) # WxH for feature extraction

        # 3. CORRELATION AND SCALE FILTER MODELS
        self.model_A, self.model_B = None, None
        self.scale_model_A, self.scale_model_B = None, None
        self.scale_factors = np.array([0.95, 0.98, 1.0, 1.02, 1.05])
        self.lambda_trust = 0.01
        self.learning_rate = 0.02

        # 4. KALMAN FILTER PARAMETERS
        dt = 1/20  # Initial assumption for delta time
        self.F = np.array([[1, dt], [0, 1]])
        self.H_kalman = np.array([[1, 0]])
        self.Q = np.array([[(dt**4)/4, (dt**3)/2], [(dt**3)/2, dt**2]]) * 1.0
        self.R = np.array([[25.0]])
        self.kf_x_state, self.kf_y_state = np.zeros((2, 1)), np.zeros((2, 1))
        self.kf_x_p, self.kf_y_p = np.eye(2) * 500, np.eye(2) * 500

    ### ----------------------------------------------------
    ### SECTION A: FEATURE EXTRACTION
    ### ----------------------------------------------------

    def _hog_scaling(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_descriptor = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        resized_image = cv2.resize(img_gray, (64, 64))
        result = hog_descriptor.compute(resized_image)
        return result if result is not None else np.zeros((1764,))

    def _hog_channel(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        win_size = (image.shape[1], image.shape[0])
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, 9)
        result = hog.compute(img_gray)
        
        h_blocks = (win_size[1] - block_size[1]) // block_stride[1] + 1
        w_blocks = (win_size[0] - block_size[0]) // block_stride[0] + 1
        features_per_block = (block_size[0] // cell_size[0]) * (block_size[1] // cell_size[1]) * 9
        
        hog_features = result.reshape((h_blocks, w_blocks, features_per_block))
        return cv2.resize(hog_features, (win_size[0] // block_stride[0], win_size[1] // block_stride[1]))

    def _extract_channels(self, image):
        hog_features = self._hog_channel(image)
        colors = np.array([
            [0.00, 0.00, 0.00], [45.37, -4.33, -33.43], [43.08, 17.51, 37.53],
            [53.59, 0.00, 0.00], [47.31, -45.33, 41.35], [65.75, 71.45, 63.32],
            [76.08, 22.25, -21.46], [32.30, 79.19, -107.86], [52.23, 75.43, 37.36],
            [100.00, 0.00, 0.00], [92.13, -16.53, 93.35]
        ], dtype=np.float32)
        
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        pixels = image_lab.reshape(-1, 3).astype(np.float32)
        distances = np.sum((pixels[:, np.newaxis, :] - colors[np.newaxis, :, :]) ** 2, axis=2)
        closest_color_indices = np.argmin(distances, axis=1)
        
        cn_features_flat = np.zeros((pixels.shape[0], colors.shape[0]), dtype=np.float32)
        cn_features_flat[np.arange(pixels.shape[0]), closest_color_indices] = 1
        cn_features = cn_features_flat.reshape(image.shape[0], image.shape[1], -1)

        hog_resized = cv2.resize(hog_features, (image.shape[1], image.shape[0]))
        return np.dstack((hog_resized, cn_features))

    ### ----------------------------------------------------
    ### SECTION B: CORE TRACKING ALGORITHMS
    ### ----------------------------------------------------

    def _teach_filter(self, f, g):
        X = np.fft.fft2(f, axes=(0, 1))
        G = np.fft.fft2(g)
        G1 = np.expand_dims(G, axis=2)
        num = np.conj(X) * G1
        denom = np.sum(np.conj(X) * X, axis=2) + self.lambda_trust
        return num, denom

    def _kalman_prediction(self, F, X_k_1, P_k_1, Q_k):
        x_k = np.dot(F, X_k_1)
        p_k = np.dot(F, np.dot(P_k_1, F.T)) + Q_k
        return x_k, p_k

    def _kalman_updating(self, x_k, p_k, H_k, z_k, R_k):
        k_1 = np.dot(np.dot(H_k, p_k), H_k.T) + R_k
        k_2 = np.dot(p_k, H_k.T)
        K = np.dot(k_2, np.linalg.inv(k_1))
        P_k_new = p_k - np.dot(np.dot(K, H_k), p_k)
        x_k_new = x_k + np.dot(K, (z_k - np.dot(H_k, x_k)))
        return x_k_new, P_k_new

    def _filter_update(self, H_new, H_old, alpha):
        return alpha * H_new + (1 - alpha) * H_old

    def _check_scale(self, frame, pos, base_size):
        scale_features = []
        for scale in self.scale_factors:
            w_s, h_s = int(base_size[0] * scale), int(base_size[1] * scale)
            x_s, y_s = max(0, int(pos[0] - w_s / 2)), max(0, int(pos[1] - h_s / 2))
            patch_s = frame[y_s:y_s+h_s, x_s:x_s+w_s]
            if patch_s.size == 0: continue
            scale_features.append(self._hog_scaling(patch_s))
        
        if not scale_features: return 1.0

        SF = np.fft.fft(np.array(scale_features), axis=0)
        scale_H = self.scale_model_A / (self.scale_model_B[:, np.newaxis] + self.lambda_trust)
        response_f = np.sum(np.conj(scale_H) * SF, axis=1)
        response = np.real(np.fft.ifft(response_f))
        
        return self.scale_factors[np.argmax(response)]

    ### ----------------------------------------------------
    ### SECTION C: MAIN PROCESSING METHOD
    ### ----------------------------------------------------

    def process_frame(self, frame, dt):
        """Processes a single video frame to detect or track the object."""
        self.F[0, 1] = dt # Update delta time in Kalman model

        if not self.tracking:
            # --- DETECTION PHASE ---
            results = self.model(frame, verbose=False)[0]
            for box in results.boxes:
                if self.model.names[int(box.cls[0].item())] == self.target_class:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    w, h = x2 - x1, y2 - y1
                    self.current_pos = (x1 + w/2, y1 + h/2)
                    self.current_size = (w, h)
                    
                    # Initialize Kalman Filter
                    self.kf_x_state[0], self.kf_y_state[0] = self.current_pos
                    
                    # Train Position Filter
                    patch = cv2.resize(frame[y1:y2, x1:x2], self.fixed_roi_size)
                    features = self._extract_channels(patch)
                    k_y = cv2.getGaussianKernel(self.fixed_roi_size[1], 18, cv2.CV_32F)
                    k_x = cv2.getGaussianKernel(self.fixed_roi_size[0], 18, cv2.CV_32F).T
                    target_y_2d = np.fft.ifft2(np.fft.fft2(k_y) * np.fft.fft2(k_x)).real
                    self.model_A, self.model_B = self._teach_filter(features, target_y_2d)

                    # Train Scale Filter
                    target_y_1d = np.fft.ifft(np.fft.fft(cv2.getGaussianKernel(len(self.scale_factors), 1, cv2.CV_32F))).real.flatten()
                    scale_features_init = [self._hog_scaling(cv2.resize(frame[max(0, int(self.current_pos[1] - (h * s)/2)):int(self.current_pos[1] + (h*s)/2), max(0, int(self.current_pos[0] - (w*s)/2)):int(self.current_pos[0] + (w*s)/2)], (64,64))) for s in self.scale_factors]
                    SF = np.fft.fft(np.array(scale_features_init), axis=0)
                    self.scale_model_A = np.conj(SF) * target_y_1d[:, np.newaxis]
                    self.scale_model_B = np.sum(np.conj(SF) * SF, axis=1)

                    self.tracking = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    break # Track the first detected person
        else:
            # --- TRACKING PHASE ---
            # Predict
            self.kf_x_state, self.kf_x_p = self._kalman_prediction(self.F, self.kf_x_state, self.kf_x_p, self.Q)
            self.kf_y_state, self.kf_y_p = self._kalman_prediction(self.F, self.kf_y_state, self.kf_y_p, self.Q)
            pred_pos = (self.kf_x_state[0, 0], self.kf_y_state[0, 0])

            # Scale Estimation
            current_scale = self._check_scale(frame, pred_pos, self.current_size)
            self.current_size = (self.current_size[0] * current_scale, self.current_size[1] * current_scale)

            # Position Localization
            search_area_scale = 1.5
            w_s, h_s = int(self.current_size[0] * search_area_scale), int(self.current_size[1] * search_area_scale)
            x_s, y_s = max(0, int(pred_pos[0] - w_s/2)), max(0, int(pred_pos[1] - h_s/2))
            search_patch = frame[y_s:y_s+h_s, x_s:x_s+w_s]

            if search_patch.shape[0] > 0 and search_patch.shape[1] > 0:
                resized_patch = cv2.resize(search_patch, self.fixed_roi_size)
                features = self._extract_channels(resized_patch)
                Z = np.fft.fft2(features, axes=(0, 1))
                
                H_filter = self.model_A / (np.expand_dims(self.model_B, axis=2) + self.lambda_trust)
                response = np.real(np.fft.ifft2(np.sum(np.conj(H_filter) * Z, axis=2)))
                
                peak_y, peak_x = np.unravel_index(np.argmax(response), response.shape)
                peak_y -= self.fixed_roi_size[1] if peak_y > self.fixed_roi_size[1] / 2 else 0
                peak_x -= self.fixed_roi_size[0] if peak_x > self.fixed_roi_size[0] / 2 else 0
                
                dx = (peak_x / self.fixed_roi_size[0]) * w_s
                dy = (peak_y / self.fixed_roi_size[1]) * h_s
                
                # Update Kalman
                measured_pos = (pred_pos[0] + dx, pred_pos[1] + dy)
                self.kf_x_state, self.kf_x_p = self._kalman_updating(self.kf_x_state, self.kf_x_p, self.H_kalman, measured_pos[0], self.R)
                self.kf_y_state, self.kf_y_p = self._kalman_updating(self.kf_y_state, self.kf_y_p, self.H_kalman, measured_pos[1], self.R)
                self.current_pos = (self.kf_x_state[0, 0], self.kf_y_state[0, 0])
                
                # --- MODEL UPDATE ---
                x1_up, y1_up = int(self.current_pos[0] - self.current_size[0]/2), int(self.current_pos[1] - self.current_size[1]/2)
                update_patch = frame[y1_up : y1_up + int(self.current_size[1]), x1_up : x1_up + int(self.current_size[0])]
                if update_patch.shape[0] > 0 and update_patch.shape[1] > 0:
                    # Update position filter
                    resized_patch_up = cv2.resize(update_patch, self.fixed_roi_size)
                    features_new = self._extract_channels(resized_patch_up)
                    k_y = cv2.getGaussianKernel(self.fixed_roi_size[1], 18, cv2.CV_32F)
                    k_x = cv2.getGaussianKernel(self.fixed_roi_size[0], 18, cv2.CV_32F).T
                    target_y_2d = np.fft.ifft2(np.fft.fft2(k_y) * np.fft.fft2(k_x)).real
                    new_A, new_B = self._teach_filter(features_new, target_y_2d)
                    self.model_A = self._filter_update(new_A, self.model_A, self.learning_rate)
                    self.model_B = self._filter_update(new_B, self.model_B, self.learning_rate)
                    
                    # Update scale filter
                    target_y_1d_up = np.fft.ifft(np.fft.fft(cv2.getGaussianKernel(len(self.scale_factors), 1, cv2.CV_32F))).real.flatten()
                    scale_features_new = [self._hog_scaling(cv2.resize(update_patch, (64,64)))]
                    SF_new = np.fft.fft(np.array(scale_features_new), axis=0)
                    new_scale_A = np.conj(SF_new) * target_y_1d_up[len(self.scale_factors)//2]
                    new_scale_B = np.sum(np.conj(SF_new) * SF_new, axis=1)
                    self.scale_model_A = self._filter_update(new_scale_A, self.scale_model_A, self.learning_rate)
                    self.scale_model_B = self._filter_update(new_scale_B, self.scale_model_B, self.learning_rate)

            # Draw tracking box
            x, y, w, h = int(self.current_pos[0]-self.current_size[0]/2), int(self.current_pos[1]-self.current_size[1]/2), int(self.current_size[0]), int(self.current_size[1])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, self.target_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame


if __name__ == "__main__":
    # Main execution block to run the tracker on a video file.
    
    video_path = 'person1.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        exit()

    tracker = AdvancedTracker(target_class="person")

    last_time = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate delta time for Kalman filter and FPS
        current_time = time.time()
        dt = current_time - last_time if last_time > 0 else 1 / 30
        last_time = current_time

        # Process the frame using the tracker instance
        processed_frame = tracker.process_frame(frame, dt)
        
        # Calculate and display FPS
        fps_frame_count += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
        
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(processed_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Advanced Tracker", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break
            
    cap.release()
    cv2.destroyAllWindows()