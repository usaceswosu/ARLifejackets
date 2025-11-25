import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from scipy.spatial import distance

# --- CONFIG ---
OUTFIT_SCALE = 1.5

# Mouse click tracking for coordinate transformation
window_click_x = 0
window_click_y = 0

model = YOLO("yolov8n.pt")

outfit_sets = [
    {'torso': 'lifejacket.png', 'head': None},
    {'torso': 'ranger_vest.png', 'head': 'ranger_hat.png'},
    {'torso': 'volunteer_vest.png', 'head': 'volunteer_hat.png'}
]

current_set_index = 0

def load_current_set():
    set_data = outfit_sets[current_set_index]
    torso_img = cv2.imread(set_data['torso'], cv2.IMREAD_UNCHANGED) if set_data['torso'] else None
    head_img = cv2.imread(set_data['head'], cv2.IMREAD_UNCHANGED) if set_data['head'] else None
    return torso_img, head_img

torso_img, head_img = load_current_set()

button_size = (80, 80)
button_positions = [(550, 100), (550, 190), (550, 280)]

processed_buttons = []
for outfit in outfit_sets:
    img_path = outfit['torso'] if outfit['torso'] else outfit['head']
    if img_path:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, button_size, interpolation=cv2.INTER_AREA)
    else:
        img = np.zeros((button_size[1], button_size[0], 3), dtype=np.uint8)
    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        overlay = cv2.merge((b, g, r))
        mask = a / 255.0
        processed_buttons.append((overlay, mask))
    else:
        processed_buttons.append((img, None))

def switch_set(event, x, y, flags, param):
    global window_click_x, window_click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        window_click_x = x
        window_click_y = y
        # Coordinate transformation happens in main loop

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# --- Camera Auto-Detection ---
def find_camera(max_tested=10):
    print("Searching for available camera...")
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            cap.release()
            return i
        cap.release()
    print("No usable camera detected.")
    return None

camera_index = find_camera()
if camera_index is None:
    print("ERROR: No camera available")
    exit()

cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()


prev_coords = {}
person_counter = 0
alpha = 0.3

cv2.namedWindow('AR Outfit Overlay', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('AR Outfit Overlay', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback('AR Outfit Overlay', switch_set)

def match_person(x_center, y_center, prev_coords):
    for pid, coords in prev_coords.items():
        px_center = (coords['x_start'] + coords['x_end']) / 2
        py_center = (coords['y_start'] + coords['y_end']) / 2
        if distance.euclidean((x_center, y_center), (px_center, py_center)) < 50:
            return pid
    return None

def overlay_image(frame, overlay_img, x_start, y_start, x_end, y_end):
    if overlay_img is None:
        return
    w = x_end - x_start
    h = y_end - y_start
    if w <= 0 or h <= 0:
        return
    resized = cv2.resize(overlay_img, (w, h), interpolation=cv2.INTER_AREA)
    if resized.shape[2] == 4:
        b, g, r, a = cv2.split(resized)
        overlay_color = cv2.merge((b, g, r))
        alpha_mask = a / 255.0
    else:
        overlay_color = resized
        alpha_mask = np.ones(resized.shape[:2], dtype=float)
    y_start_c = max(0, y_start)
    y_end_c = min(frame.shape[0], y_end)
    x_start_c = max(0, x_start)
    x_end_c = min(frame.shape[1], x_end)
    roi = frame[y_start_c:y_end_c, x_start_c:x_end_c]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return
    alpha_resized = cv2.resize(alpha_mask, (roi.shape[1], roi.shape[0]))
    overlay_resized = cv2.resize(overlay_color, (roi.shape[1], roi.shape[0]))
    for c in range(3):
        roi[:, :, c] = (alpha_resized * overlay_resized[:, :, c] +
                        (1 - alpha_resized) * roi[:, :, c]).astype(np.uint8)
    frame[y_start_c:y_end_c, x_start_c:x_end_c] = roi

def scale_to_screen(frame, screen_w, screen_h):
    h, w = frame.shape[:2]
    frame_ratio = w / h
    screen_ratio = screen_w / screen_h
    if frame_ratio > screen_ratio:
        new_w = screen_w
        new_h = int(screen_w / frame_ratio)
    else:
        new_h = screen_h
        new_w = int(screen_h * frame_ratio)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x_offset = (screen_w - new_w) // 2
    y_offset = (screen_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

screen_w = 1920
screen_h = 1080

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    boxes = results[0].boxes
    current_coords = {}

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_person = frame[y1:y2, x1:x2]
            if cropped_person.size == 0:
                continue

            rgb_crop = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_crop)
            if not pose_results.pose_landmarks:
                continue

            lm = pose_results.pose_landmarks.landmark

            if torso_img is not None:
                ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                lh = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
                rh = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

                ls_x = int(ls.x * (x2 - x1)) + x1
                rs_x = int(rs.x * (x2 - x1)) + x1
                ls_y = int(ls.y * (y2 - y1)) + y1
                rs_y = int(rs.y * (y2 - y1)) + y1
                lh_y = int(lh.y * (y2 - y1)) + y1
                rh_y = int(rh.y * (y2 - y1)) + y1

                x_start_torso = min(ls_x, rs_x)
                x_end_torso = max(ls_x, rs_x)
                y_start_torso = min(ls_y, rs_y)
                y_end_torso = max(lh_y, rh_y)

                cx = (x_start_torso + x_end_torso) // 2
                cy = (y_start_torso + y_end_torso) // 2
                w = int((x_end_torso - x_start_torso) * OUTFIT_SCALE)
                h = int((y_end_torso - y_start_torso) * OUTFIT_SCALE)
                x_start_torso = cx - w // 2
                x_end_torso = cx + w // 2
                y_start_torso = cy - h // 2
                y_end_torso = cy + h // 2

            head_coords_defined = False
            if head_img is not None:
                torso_top = y_start_torso if torso_img is not None else y1
                head_center_x = (x_start_torso + x_end_torso) // 2 if torso_img is not None else (x1 + x2) // 2
                torso_width = x_end_torso - x_start_torso if torso_img is not None else 50

                hat_width = int(torso_width * 0.8)
                hat_height = int(hat_width * head_img.shape[0] / head_img.shape[1])

                x_start_head = head_center_x - hat_width // 2
                y_end_head = torso_top + int(hat_height * 0.1)
                y_start_head = y_end_head - hat_height
                x_end_head = x_start_head + hat_width

                head_coords_defined = True

            # Determine person ID for smoothing
            x_center = (x_start_torso + x_end_torso) / 2 if torso_img is not None else (x_start_head + x_end_head) / 2
            y_center = (y_start_torso + y_end_torso) / 2 if torso_img is not None else (y_start_head + y_end_head) / 2

            person_id = match_person(x_center, y_center, prev_coords)
            if person_id is None:
                person_id = person_counter
                person_counter += 1

            # Smoothing coordinates
            if person_id in prev_coords:
                if torso_img is not None:
                    x_start_torso = int(alpha * x_start_torso + (1 - alpha) * prev_coords[person_id]['x_start'])
                    x_end_torso = int(alpha * x_end_torso + (1 - alpha) * prev_coords[person_id]['x_end'])
                    y_start_torso = int(alpha * y_start_torso + (1 - alpha) * prev_coords[person_id]['y_start'])
                    y_end_torso = int(alpha * y_end_torso + (1 - alpha) * prev_coords[person_id]['y_end'])

                if head_img is not None and head_coords_defined:
                    hx_start_prev = prev_coords[person_id].get('hx_start') or x_start_head
                    hx_end_prev = prev_coords[person_id].get('hx_end') or x_end_head
                    hy_start_prev = prev_coords[person_id].get('hy_start') or y_start_head
                    hy_end_prev = prev_coords[person_id].get('hy_end') or y_end_head

                    x_start_head = int(alpha * x_start_head + (1 - alpha) * hx_start_prev)
                    x_end_head = int(alpha * x_end_head + (1 - alpha) * hx_end_prev)
                    y_start_head = int(alpha * y_start_head + (1 - alpha) * hy_start_prev)
                    y_end_head = int(alpha * y_end_head + (1 - alpha) * hy_end_prev)

            # Save smoothed coordinates
            current_coords[person_id] = {
                'x_start': x_start_torso if torso_img is not None else x_start_head,
                'x_end': x_end_torso if torso_img is not None else x_end_head,
                'y_start': y_start_torso if torso_img is not None else y_start_head,
                'y_end': y_end_torso if torso_img is not None else y_end_head,
                'hx_start': x_start_head if head_img is not None and head_coords_defined else None,
                'hx_end': x_end_head if head_img is not None and head_coords_defined else None,
                'hy_start': y_start_head if head_img is not None and head_coords_defined else None,
                'hy_end': y_end_head if head_img is not None and head_coords_defined else None
            }

            # Apply overlays
            if torso_img is not None:
                overlay_image(frame, torso_img, x_start_torso, y_start_torso, x_end_torso, y_end_torso)

            if head_img is not None and head_coords_defined:
                overlay_image(frame, head_img, x_start_head, y_start_head, x_end_head, y_end_head)

    # Update previous coordinates for smoothing
    prev_coords = current_coords

    # Draw buttons
    for i, (bx, by) in enumerate(button_positions):
        overlay, mask = processed_buttons[i]
        h, w = overlay.shape[:2]

        if mask is not None:
            for c in range(3):
                frame[by:by+h, bx:bx+w, c] = (
                    mask * overlay[:, :, c] +
                    (1 - mask) * frame[by:by+h, bx:bx+w, c]
                ).astype(np.uint8)
        else:
            frame[by:by+h, bx:bx+w] = overlay

    # Scale frame to fullscreen with correct aspect ratio
    display_frame = scale_to_screen(frame, screen_w, screen_h)

    # Reverse-scale UI interactions
    # Compute offsets from scale_to_screen()
    h, w = frame.shape[:2]
    frame_ratio = w / h
    screen_ratio = screen_w / screen_h

    if frame_ratio > screen_ratio:
        new_w = screen_w
        new_h = int(screen_w / frame_ratio)
    else:
        new_h = screen_h
        new_w = int(screen_h * frame_ratio)

    x_offset = (screen_w - new_w) // 2
    y_offset = (screen_h - new_h) // 2

    # Transform click coords → frame coords
    if window_click_x != 0 or window_click_y != 0:  # Only process if there was a click
        fx = (window_click_x - x_offset) * (w / new_w)
        fy = (window_click_y - y_offset) * (h / new_h)

        # If click is inside actual frame region
        if 0 <= fx < w and 0 <= fy < h:
            # Now check buttons using transformed coords
            for i, (bx, by) in enumerate(button_positions):
                bw, bh = button_size
                if bx <= fx <= bx + bw and by <= fy <= by + bh:
                    current_set_index = i
                    torso_img, head_img = load_current_set()
                    print(f"Switched outfit: {outfit_sets[i]}")
                    break
        
        # Reset click coordinates after processing
        window_click_x = 0
        window_click_y = 0

    cv2.imshow('AR Outfit Overlay', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


