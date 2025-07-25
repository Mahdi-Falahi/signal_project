class CUSTOM_Tracker:
    """
    A headless, self-contained object tracker class designed for integration.
    This tracker uses a combination of a correlation filter (with HOG and Color Name features),
    a Kalman filter for motion prediction, and YOLO for robust re-detection.
    It is designed to be used like standard trackers (e.g., CSRT).
    """

    def __init__(self, yolo_model_path='yolo11n.pt', confidence_threshold=0.5):
        # --- Models and Main Parameters ---
        self.model = YOLO(yolo_model_path)
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.model_A, self.model_B = None, None
        self.current_pos, self.current_size = (0, 0), (0, 0)
        self.fixed_roi_size = (64, 128)  # (width, height)
        self.is_tracking = False
        self.psr_score = 0.0
        self.bbox = (0, 0, 0, 0)
        
        # --- Tracking Logic Parameters ---
        self.frames_since_update = 0
        self.MAX_FRAMES_TO_SKIP = 30
        self.RE_DETECTION_FRAME_THRESHOLD = 5
        self.LEARNING_RATE = 0.025
        self.LAMBDA_TRUST = 0.01
        self.PSR_THRESHOLD = 5.5

        # --- Kalman Filter Initialization ---
        dt = 1/30.0
        self.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 1e-2
        self.R = np.eye(2, dtype=np.float32) * 25
        self.P = np.eye(4, dtype=np.float32) * 100
        self.kalman_state = np.zeros(4, dtype=np.float32)
        self.last_time = 0

    def _hog_channel(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        win_size = (image.shape[1], image.shape[0])
        hog = cv2.HOGDescriptor(win_size, (16, 16), (8, 8), (8, 8), 9)
        result = hog.compute(img)
        if result is None: return np.zeros((image.shape[0] // 8, image.shape[1] // 8, 36))
        h_blocks = (win_size[1] - 16) // 8 + 1
        w_blocks = (win_size[0] - 16) // 8 + 1
        hog_features = result.reshape((h_blocks, w_blocks, 4 * 9))
        return cv2.resize(hog_features, (win_size[0] // 8, win_size[1] // 8))

    def _extract_channels(self, image):
        hog_features = self._hog_channel(image)
        colors = np.array([[0,0,0], [45.3,-4.3,-33.4], [43,17.5,37.5], [53.6,0,0], [47.3,-45.3,41.3], [65.7,71.4,63.3], [76,22.2,-21.4], [32.3,79.1,-107.8], [52.2,75.4,37.3], [100,0,0], [92.1,-16.5,93.3]], dtype=np.float32)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        pixels = image_lab.reshape(-1, 3).astype(np.float32)
        distances = np.sum((pixels[:, np.newaxis, :] - colors[np.newaxis, :, :])**2, axis=2)
        closest_color_indices = np.argmin(distances, axis=1)
        cn_features_flat = np.zeros((pixels.shape[0], colors.shape[0]), dtype=np.float32)
        cn_features_flat[np.arange(pixels.shape[0]), closest_color_indices] = 1
        cn_features = cn_features_flat.reshape(image.shape[0], image.shape[1], -1)
        hog_resized = cv2.resize(hog_features, (image.shape[1], image.shape[0]))
        return np.dstack((hog_resized, cn_features))

    def _teaching(self, f, g, lambda_trust):
        X = np.fft.fft2(f, axes=(0, 1))
        G = np.fft.fft2(g)
        num = np.conj(X) * np.expand_dims(G, axis=2)
        denom = np.sum(np.conj(X) * X, axis=2) + lambda_trust
        return num, denom

    def _create_gaussian_target(self, size): # size is (height, width)
        height, width = size
        center_x, center_y = width // 2, height // 2
        sigma = np.sqrt(width * height) * 0.1
        yy, xx = np.mgrid[0:height, 0:width]
        gaussian = np.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma**2))
        return np.roll(np.roll(gaussian, -center_y, axis=0), -center_x, axis=1)

    def _filter_updating(self, H_new, H_old, alpha):
        return alpha * H_new + (1 - alpha) * H_old

    def _calculate_psr(self, response_map):
        if not isinstance(response_map, np.ndarray) or response_map.size == 0: return 0.0
        peak_loc = np.unravel_index(np.argmax(response_map), response_map.shape)
        peak_value = response_map[peak_loc]
        sidelobe_window = 11; h, w = response_map.shape
        y_start=max(0, peak_loc[0]-sidelobe_window//2); y_end=min(h, peak_loc[0]+sidelobe_window//2+1)
        x_start=max(0, peak_loc[1]-sidelobe_window//2); x_end=min(w, peak_loc[1]+sidelobe_window//2+1)
        mask = np.ones_like(response_map, dtype=bool); mask[y_start:y_end, x_start:x_end] = False
        sidelobe = response_map[mask]
        if sidelobe.size > 0:
            std_sidelobe = np.std(sidelobe)
            if std_sidelobe > 1e-5: return (peak_value - np.mean(sidelobe)) / std_sidelobe
        return 0.0
        
    def init(self, frame, bbox):
        """
        Initializes the tracker with the first frame and the object's bounding box.
        Args:
            frame (np.ndarray): The first video frame.
            bbox (tuple): The initial bounding box (x, y, w, h).
        """
        x, y, w, h = map(int, bbox)
        self.current_pos = (x + w / 2, y + h / 2)
        self.current_size = (w, h)
        self.bbox = (x, y, w, h)
        self.kalman_state = np.array([self.current_pos[0], 0, self.current_pos[1], 0], dtype=np.float32)
        self.last_time = time.time()
        patch = cv2.resize(frame[y:y+h, x:x+w], self.fixed_roi_size)
        features = self._extract_channels(patch)
        target_y_2d = self._create_gaussian_target((self.fixed_roi_size[1], self.fixed_roi_size[0]))
        self.model_A, self.model_B = self._teaching(features, target_y_2d, self.LAMBDA_TRUST)
        self.is_tracking = True
        self.frames_since_update = 0

    def update(self, frame):
        """
        Updates the tracker on a new frame to find the object.
        Args:
            frame (np.ndarray): The new video frame.
        Returns:
            tuple: A tuple containing:
                   - success (bool): True if tracking is successful, False otherwise.
                   - new_bbox (tuple): The new bounding box (x, y, w, h).
        """
        if not self.is_tracking: return False, self.bbox

        dt = (time.time() - self.last_time) if self.last_time > 0 else 1/30.0
        self.last_time = time.time()
        self.F[0,1], self.F[2,3] = dt, dt
        self.kalman_state = self.F @ self.kalman_state
        self.P = self.F @ self.P @ self.F.T + self.Q
        pred_pos = (self.kalman_state[0], self.kalman_state[2])
        
        w_search, h_search = int(self.current_size[0]*2.0), int(self.current_size[1]*2.0)
        x_search, y_search = int(pred_pos[0]-w_search/2), int(pred_pos[1]-h_search/2)
        search_patch = frame[max(0,y_search):y_search+h_search, max(0,x_search):x_search+w_search]
        
        if search_patch.size > 0:
            resized_patch = cv2.resize(search_patch, self.fixed_roi_size)
            features = self._extract_channels(resized_patch)
            Z = np.fft.fft2(features, axes=(0, 1))
            H_filter = self.model_A / (np.expand_dims(self.model_B, axis=2) + self.LAMBDA_TRUST)
            response = np.real(np.fft.ifft2(np.sum(np.conj(H_filter) * Z, axis=2)))
            self.psr_score = self._calculate_psr(response)

            if self.psr_score > self.PSR_THRESHOLD:
                self.frames_since_update = 0
                peak_y, peak_x = np.unravel_index(np.argmax(response), response.shape)
                if peak_y > self.fixed_roi_size[1]/2: peak_y -= self.fixed_roi_size[1]
                if peak_x > self.fixed_roi_size[0]/2: peak_x -= self.fixed_roi_size[0]
                dx, dy = (peak_x/self.fixed_roi_size[0])*w_search, (peak_y/self.fixed_roi_size[1])*h_search
                measured_pos = (pred_pos[0]+dx, pred_pos[1]+dy)
                z = np.array([measured_pos[0], measured_pos[1]], dtype=np.float32)
                K = self.P@self.H.T@np.linalg.inv(self.H@self.P@self.H.T+self.R)
                self.kalman_state += K@(z-self.H@self.kalman_state)
                self.P = (np.eye(4)-K@self.H)@self.P
                self.current_pos = (self.kalman_state[0], self.kalman_state[2])
                
                x1_up, y1_up = int(self.current_pos[0]-self.current_size[0]/2), int(self.current_pos[1]-self.current_size[1]/2)
                w_up, h_up = int(self.current_size[0]), int(self.current_size[1])
                update_patch = frame[max(0,y1_up):y1_up+h_up, max(0,x1_up):x1_up+w_up]
                if update_patch.size > 0:
                    resized_patch_up = cv2.resize(update_patch, self.fixed_roi_size)
                    features_new = self._extract_channels(resized_patch_up)
                    target_y_2d = self._create_gaussian_target((self.fixed_roi_size[1], self.fixed_roi_size[0]))
                    new_A, new_B = self._teaching(features_new, target_y_2d, self.LAMBDA_TRUST)
                    self.model_A = self._filter_updating(new_A, self.model_A, self.LEARNING_RATE)
                    self.model_B = self._filter_updating(new_B, self.model_B, self.LEARNING_RATE)
            else:
                self.frames_since_update += 1

            if self.RE_DETECTION_FRAME_THRESHOLD < self.frames_since_update <= self.MAX_FRAMES_TO_SKIP:
                roi_x1 = max(0, int(pred_pos[0]-self.current_size[0]*1.5)); roi_y1 = max(0, int(pred_pos[1]-self.current_size[1]*1.5))
                roi_x2 = min(frame.shape[1], int(pred_pos[0]+self.current_size[0]*1.5)); roi_y2 = min(frame.shape[0], int(pred_pos[1]+self.current_size[1]*1.5))
                roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                
                found_good_detection = False
                if roi_frame.size > 0:
                    roi_results = self.model(roi_frame, verbose=False, classes=[0], conf=self.CONFIDENCE_THRESHOLD)[0]
                    if len(roi_results.boxes) > 0:
                        best_det_box, max_area = None, 0
                        for r in roi_results.boxes:
                            box, area = r.xyxy[0], (r.xyxy[0][2]-r.xyxy[0][0])*(r.xyxy[0][3]-r.xyxy[0][1])
                            if area > max_area: max_area, best_det_box = area, box
                        
                        if best_det_box is not None:
                            gw, gh = int(best_det_box[2]-best_det_box[0]), int(best_det_box[3]-best_det_box[1])
                            last_area = self.current_size[0] * self.current_size[1]
                            if last_area > 0 and (gw * gh) > 0.25 * last_area:
                                gx1, gy1 = int(best_det_box[0]+roi_x1), int(best_det_box[1]+roi_y1)
                                self.init(frame, (gx1, gy1, gw, gh))
                                found_good_detection = True
                                return True, self.bbox
                
                if not found_good_detection: self.current_pos = pred_pos
        
        if self.frames_since_update > self.MAX_FRAMES_TO_SKIP:
            self.is_tracking = False
            return False, self.bbox
        
        if self.frames_since_update > 0: self.current_pos = pred_pos

        w, h = self.current_size
        x, y = int(self.current_pos[0] - w/2), int(self.current_pos[1] - h/2)
        self.bbox = (x, y, int(w), int(h))
        return True, self.bbox