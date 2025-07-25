class CUSTOM_Tracker:
    """
    A headless, self-contained object tracker class based on the signal_amirreza.ipynb notebook.

    This tracker combines a correlation filter (using HOG and Color Name features) with a Kalman
    filter for motion prediction and uses YOLO for robust re-detection when tracking confidence is low.
    It's designed for easy integration into other applications.
    """

    def __init__(self, yolo_model_path='yolo11n.pt', confidence_threshold=0.5, psr_threshold=5.5, max_frames_to_skip=50, redetection_threshold=2):
        # --- Models and Main Parameters ---
        self.model = YOLO(yolo_model_path)
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.PSR_THRESHOLD = psr_threshold
        self.MAX_FRAMES_TO_SKIP = max_frames_to_skip
        self.RE_DETECTION_THRESHOLD = redetection_threshold

        # --- Internal State Variables ---
        self.model_A, self.model_B = None, None
        self.current_pos, self.current_size = (0, 0), (0, 0)
        self.fixed_roi_size = (64, 128)  # (width, height)
        self.is_tracking = False
        self.psr_score = 0.0
        self.bbox = (0, 0, 0, 0)
        self.frames_since_update = 0

        # --- Kalman Filter Initialization ---
        dt = 1/30.0  # Initial assumption for delta-time
        self.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 1e-2
        self.R = np.eye(2, dtype=np.float32) * 25
        self.P = np.eye(4, dtype=np.float32) * 100
        self.kalman_state = np.zeros(4, dtype=np.float32)
        self.last_time = 0

    # --- Private Helper Methods ---

    def _hog_channel(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        win_size = (image.shape[1], image.shape[0])
        # Using fixed parameters as in the notebook
        hog = cv2.HOGDescriptor(win_size, (16, 16), (8, 8), (8, 8), 9)
        result = hog.compute(img)
        if result is None:
            # Return a zero array with the expected feature dimensions if HOG fails
            h, w = win_size[1] // 8, win_size[0] // 8
            features_per_block = 36 # 2*2 cells * 9 bins
            return np.zeros((h, w, features_per_block), dtype=np.float32)
        h_blocks = (win_size[1] - 16) // 8 + 1
        w_blocks = (win_size[0] - 16) // 8 + 1
        features_per_block = 36 # 2*2 cells * 9 bins
        hog_features = result.reshape((h_blocks, w_blocks, features_per_block))
        return cv2.resize(hog_features, (win_size[0] // 8, win_size[1] // 8))

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

    def _teaching(self, features, target, lambda_trust=0.01):
        F = np.fft.fft2(features, axes=(0, 1))
        G = np.fft.fft2(target)
        G_expanded = np.expand_dims(G, axis=2)
        numerator = np.conj(F) * G_expanded
        denominator = np.sum(np.conj(F) * F, axis=2) + lambda_trust
        return numerator, denominator

    def _create_gaussian_target(self, size):
        height, width = size
        center_x, center_y = width // 2, height // 2
        sigma = np.sqrt(width * height) * 0.1  # Heuristic for sigma
        yy, xx = np.mgrid[0:height, 0:width]
        gaussian = np.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma**2))
        return np.roll(np.roll(gaussian, -center_y, axis=0), -center_x, axis=1)

    def _filter_updating(self, new_model, old_model, alpha=0.025):
        return alpha * new_model + (1 - alpha) * old_model

    def _calculate_psr(self, response_map):
        if not isinstance(response_map, np.ndarray) or response_map.size == 0:
            return 0.0
        peak_loc = np.unravel_index(np.argmax(response_map), response_map.shape)
        peak_value = response_map[peak_loc]
        sidelobe_window = 11
        h, w = response_map.shape
        y_start = max(0, peak_loc[0] - sidelobe_window // 2)
        y_end = min(h, peak_loc[0] + sidelobe_window // 2 + 1)
        x_start = max(0, peak_loc[1] - sidelobe_window // 2)
        x_end = min(w, peak_loc[1] + sidelobe_window // 2 + 1)
        mask = np.ones_like(response_map, dtype=bool)
        mask[y_start:y_end, x_start:x_end] = False
        sidelobe = response_map[mask]
        if sidelobe.size > 0:
            mean_sidelobe = np.mean(sidelobe)
            std_sidelobe = np.std(sidelobe)
            if std_sidelobe > 1e-5:
                return (peak_value - mean_sidelobe) / std_sidelobe
        return 0.0

    # --- Public API Methods ---

    def init(self, frame, bbox):
        """
        Initializes or re-initializes the tracker with the first frame and the object's bounding box.

        Args:
            frame (np.ndarray): The first video frame.
            bbox (tuple): The initial bounding box in (x, y, w, h) format.
        """
        clean_frame = cv2.medianBlur(frame, 3)
        x, y, w, h = map(int, bbox)

        self.current_pos = (x + w / 2, y + h / 2)
        self.current_size = (w, h)
        self.bbox = (x, y, w, h)
        self.kalman_state = np.array([self.current_pos[0], 0, self.current_pos[1], 0], dtype=np.float32)
        self.last_time = time.time()
        
        patch = cv2.resize(clean_frame[y:y+h, x:x+w], self.fixed_roi_size)
        features = self._extract_channels(patch)
        target_y_2d = self._create_gaussian_target((self.fixed_roi_size[1], self.fixed_roi_size[0]))
        
        self.model_A, self.model_B = self._teaching(features, target_y_2d)
        self.is_tracking = True
        self.frames_since_update = 0

    def update(self, frame):
        """
        Updates the tracker on a new frame to find the object's new position.

        Args:
            frame (np.ndarray): The new video frame.

        Returns:
            tuple: A tuple containing:
                   - success (bool): True if tracking is ongoing, False if lost.
                   - new_bbox (tuple): The new bounding box (x, y, w, h).
        """
        if not self.is_tracking:
            return False, self.bbox

        clean_frame = cv2.medianBlur(frame, 3)
        dt = time.time() - self.last_time
        self.last_time = time.time()

        # Kalman Prediction
        self.F[0, 1], self.F[2, 3] = dt, dt
        self.kalman_state = self.F @ self.kalman_state
        self.P = self.F @ self.P @ self.F.T + self.Q
        pred_pos = (self.kalman_state[0], self.kalman_state[2])
        pos_to_return = pred_pos

        # Correlation Filter Tracking
        w_search, h_search = int(self.current_size[0] * 2.5), int(self.current_size[1] * 2.5)
        x_search, y_search = int(pred_pos[0] - w_search / 2), int(pred_pos[1] - h_search / 2)
        search_patch = clean_frame[max(0, y_search):y_search + h_search, max(0, x_search):x_search + w_search]

        is_confident = False
        if search_patch.size > 0:
            resized_patch = cv2.resize(search_patch, self.fixed_roi_size)
            features = self._extract_channels(resized_patch)
            Z = np.fft.fft2(features, axes=(0, 1))
            H_filter = self.model_A / (np.expand_dims(self.model_B, axis=2) + 0.01)
            response = np.real(np.fft.ifft2(np.sum(np.conj(H_filter) * Z, axis=2)))
            self.psr_score = self._calculate_psr(response)

            if self.psr_score > self.PSR_THRESHOLD:
                is_confident = True
                self.frames_since_update = 0
                
                # Update position based on filter response
                peak_y, peak_x = np.unravel_index(np.argmax(response), response.shape)
                if peak_y > self.fixed_roi_size[1] / 2: peak_y -= self.fixed_roi_size[1]
                if peak_x > self.fixed_roi_size[0] / 2: peak_x -= self.fixed_roi_size[0]
                dx = (peak_x / self.fixed_roi_size[0]) * w_search
                dy = (peak_y / self.fixed_roi_size[1]) * h_search
                measured_pos = (pred_pos[0] + dx, pred_pos[1] + dy)

                # Kalman Update
                z = np.array([measured_pos[0], measured_pos[1]], dtype=np.float32)
                K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
                self.kalman_state = self.kalman_state + K @ (z - self.H @ kalman_state)
                self.P = (np.eye(4) - K @ self.H) @ self.P
                self.current_pos = (self.kalman_state[0], self.kalman_state[2])
                pos_to_return = self.current_pos
                
                # Model Update
                x_up, y_up, w_up, h_up = self.bbox
                update_patch = clean_frame[y_up:y_up+h_up, x_up:x_up+w_up]
                if update_patch.size > 0:
                    features_new = self._extract_channels(cv2.resize(update_patch, self.fixed_roi_size))
                    target_new = self._create_gaussian_target((self.fixed_roi_size[1], self.fixed_roi_size[0]))
                    new_A, new_B = self._teaching(features_new, target_new)
                    self.model_A = self._filter_updating(new_A, self.model_A)
                    self.model_B = self._filter_updating(new_B, self.model_B)

        if not is_confident:
            self.frames_since_update += 1
            self.current_pos = pred_pos # Trust Kalman prediction if confidence is low
            
            # Attempt re-detection if tracker has been uncertain for a few frames
            if self.RE_DETECTION_THRESHOLD < self.frames_since_update <= self.MAX_FRAMES_TO_SKIP:
                w, h = self.current_size
                # Using a corrected, centered RoI for better re-detection
                roi_x1, roi_y1 = max(0, int(pred_pos[0]-w)), max(0, int(pred_pos[1]-h))
                roi_x2, roi_y2 = min(frame.shape[1], int(pred_pos[0]+w)), min(frame.shape[0], int(pred_pos[1]+h))
                roi_frame = clean_frame[roi_y1:roi_y2, roi_x1:roi_x2]

                if roi_frame.size > 0:
                    results = self.model(roi_frame, verbose=False, classes=[0], conf=self.CONFIDENCE_THRESHOLD)[0]
                    if len(results.boxes) > 0:
                        box = results.boxes[0].xyxy[0]
                        gx, gy = int(box[0] + roi_x1), int(box[1] + roi_y1)
                        gw, gh = int(box[2] - box[0]), int(box[3] - box[1])
                        
                        last_area = self.current_size[0] * self.current_size[1]
                        if last_area > 0 and (gw * gh) > 0.25 * last_area:
                            self.init(frame, (gx, gy, gw, gh))
                            return True, self.bbox # Return immediately after successful re-initialization

        if self.frames_since_update > self.MAX_FRAMES_TO_SKIP:
            self.is_tracking = False
            return False, self.bbox

        # Finalize bounding box for this frame
        x, y = int(pos_to_return[0] - self.current_size[0] / 2), int(pos_to_return[1] - self.current_size[1] / 2)
        w, h = int(self.current_size[0]), int(self.current_size[1])
        self.bbox = (x, y, w, h)
        return True, self.bbox