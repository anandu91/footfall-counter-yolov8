import cv2

def draw_id_box(
    frame,
    bb,                 # (x1, y1, x2, y2)
    tid=None,           # track id (int or None)
    label=None,         # optional extra text (e.g., "0.86" or "person")
    color=(0, 255, 0),  # box color
    thickness=2,
    show_centroid=True
):
    x1, y1, x2, y2 = map(int, bb)
    h, w = frame.shape[:2]

    # clip to frame (avoids draw errors at borders)
    x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))

    # draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # centroid
    if show_centroid:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    # label text (ID + optional label)
    if tid is not None or label is not None:
        text = f"ID {tid}" if tid is not None else ""
        if label:
            text = f"{text} {label}".strip()

        # text size + background box
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.55, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        tx1, ty1 = x1, max(0, y1 - th - 6)
        tx2, ty2 = x1 + tw + 6, y1

        # black bg with slight transparency look
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 0, 0), -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 6), font, scale, (0, 200, 255), thick)

    # function mutates the frame in-place; no return needed
