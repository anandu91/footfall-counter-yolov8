# footfall_counter.py
import argparse, math, os
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------- Args -----------------
def parse_args():
    ap = argparse.ArgumentParser("Footfall Counter (YOLOv8 + ByteTrack)")
    ap.add_argument("--source", required=True, help="Video path or webcam index (0)")
    ap.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLOv8 weights")
    ap.add_argument("--conf", type=float, default=0.5, help="Detection confidence [0..1]")

    # MODE: line-cross vs frame-boundary
    ap.add_argument("--count_mode", choices=["line", "frame"], default="line",
                    help="Counting mode: 'line' (ROI crossing) or 'frame' (appear/leave by edges)")

    # ----- line-cross params -----
    ap.add_argument("--line", nargs=4, type=float, default=[0.5, 0.1, 0.5, 0.9],
                    help="Normalized ROI line x1 y1 x2 y2 (0..1)")
    ap.add_argument("--direction", choices=["lr","rl","tb","bt"], default="lr",
                    help="Entry direction for line mode")
    ap.add_argument("--margin", type=float, default=0.03,
                    help="Outer margin band around line (fraction of max(H,W))")
    ap.add_argument("--inner", type=float, default=0.015,
                    help="Inner band around line (fraction of max(H,W))")
    ap.add_argument("--mintravel", type=float, default=12.0,
                    help="Min centroid travel (px) between frames to accept a crossing")
    ap.add_argument("--cooldown", type=int, default=25, help="Frames to ignore after a count (per-ID)")
    ap.add_argument("--minframesoff", type=int, default=6,
                    help="Frames an ID must stay off the line before eligible again")
    ap.add_argument("--smooth", type=float, default=0.35, help="EMA smoothing factor for centroid")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame")

    # ----- frame-boundary params -----
    ap.add_argument("--edge_band", type=float, default=0.06,
                    help="Edge band width as fraction of max(H,W)")
    ap.add_argument("--disappear_frames", type=int, default=15,
                    help="Consecutive missed frames to declare EXIT")
    ap.add_argument("--spawn_frames", type=int, default=2,
                    help="Count ENTRY only if first seen within edge_band during first N frames")
    ap.add_argument("--edge_hold", type=int, default=6,
                    help="Require last K frames to be near the same edge before EXIT")
    ap.add_argument("--vel_min", type=float, default=1.2,
                    help="Min outward velocity (px/frame) to accept EXIT")

    ap.add_argument("--maxw", type=int, default=1280, help="Resize if width>maxw")
    ap.add_argument("--save", default="assets/example_output.mp4", help="Output MP4 path")
    ap.add_argument("--show", action="store_true", help="Show preview window")
    return ap.parse_args()

# ----------------- Geometry helpers (line mode) -----------------
def unit_normal(ax, ay, bx, by):
    vx, vy = (bx - ax, by - ay)
    nx, ny = (-vy, vx)
    nrm = math.hypot(nx, ny) + 1e-9
    return nx / nrm, ny / nrm

def signed_distance(px, py, ax, ay, bx, by):
    nx, ny = unit_normal(ax, ay, bx, by)
    apx, apy = (px - ax, py - ay)
    return apx * nx + apy * ny

def vel_along_normal(p0, p1, ax, ay, bx, by):
    nx, ny = unit_normal(ax, ay, bx, by)
    dx, dy = (p1[0] - p0[0], p1[1] - p0[1])
    return dx * nx + dy * ny

def entry_from_vel(vn, direction):
    if direction == "lr":  return vn > 0
    if direction == "rl":  return vn < 0
    if direction == "tb":  return vn > 0
    if direction == "bt":  return vn < 0

# ----------------- Frame-edge helpers -----------------
LEFT, RIGHT, TOP, BOTTOM, NONE = "L","R","T","B","N"

def nearest_edge(p, W, H):
    x, y = p
    dL, dR, dT, dB = x, (W - x), y, (H - y)
    m = min(dL, dR, dT, dB)
    if m == dL: return LEFT
    if m == dR: return RIGHT
    if m == dT: return TOP
    return BOTTOM

def dist_to_edge(p, W, H, which):
    x, y = p
    if which == LEFT:   return x
    if which == RIGHT:  return W - x
    if which == TOP:    return y
    if which == BOTTOM: return H - y
    return 1e9

def outward_component(v, edge):
    vx, vy = v
    if edge == LEFT:   return -vx
    if edge == RIGHT:  return  vx
    if edge == TOP:    return -vy
    if edge == BOTTOM: return  vy
    return 0.0

def near_edge(p, W, H, band_px, edge):
    d = dist_to_edge(p, W, H, edge)
    return d <= band_px

# ----------------- Drawing helpers -----------------
def draw_hud(frame, cin, cout, mode, tracking=True):
    """Top-left HUD with mode and counts; bottom labels like your screenshot."""
    cv2.rectangle(frame, (10,10), (420,120), (0,0,0), -1)
    cv2.putText(frame, f"Mode : {mode}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Entry: {cin}", (20,75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Exit : {cout}", (20,110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)

    h, w = frame.shape[:2]
    status = "Tracking" if tracking else "Waiting"
    tot_inside = max(0, cin - cout)
    # bottom-left status
    cv2.putText(frame, f"Status: {status}", (20, h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3, cv2.LINE_AA)
    # bottom-right total inside
    txt = f"Total people inside: [{tot_inside}]"
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    cv2.putText(frame, txt, (w - tw - 20, h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3, cv2.LINE_AA)

def draw_edge_bands(frame, band_px):
    """Frame-mode visualization: translucent edge bands."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (int(band_px), h), (0,255,255), -1)
    cv2.rectangle(overlay, (w-int(band_px),0), (w, h), (0,255,255), -1)
    cv2.rectangle(overlay, (0,0), (w, int(band_px)), (0,255,255), -1)
    cv2.rectangle(overlay, (0, h-int(band_px)), (w, h), (0,255,255), -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

def draw_line_bands(frame, a, b, outer_m, inner_m):
    """Line-mode visualization: line + inner/outer hysteresis."""
    ax, ay = a; bx, by = b
    nx, ny = unit_normal(ax, ay, bx, by)
    overlay = frame.copy()
    p1 = (int(ax + nx*outer_m), int(ay + ny*outer_m))
    p2 = (int(bx + nx*outer_m), int(by + ny*outer_m))
    p3 = (int(bx - nx*outer_m), int(by - ny*outer_m))
    p4 = (int(ax - nx*outer_m), int(ay - ny*outer_m))
    cv2.line(overlay, p1, p2, (0,255,255), 1, cv2.LINE_AA)
    cv2.line(overlay, p3, p4, (0,255,255), 1, cv2.LINE_AA)
    q1 = (int(ax + nx*inner_m), int(ay + ny*inner_m))
    q2 = (int(bx + nx*inner_m), int(by + ny*inner_m))
    q3 = (int(bx - nx*inner_m), int(by - ny*inner_m))
    q4 = (int(ax - nx*inner_m), int(ay - ny*inner_m))
    cv2.line(overlay, q1, q2, (0,180,255), 1, cv2.LINE_AA)
    cv2.line(overlay, q3, q4, (0,180,255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    # main line
    cv2.line(frame, (ax,ay), (bx,by), (0,255,255), 2, cv2.LINE_AA)

# ----------------- Main -----------------
def main():
    args = parse_args()
    src = 0 if (str(args.source).isdigit()) else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    ok, frame = cap.read()
    if not ok: raise RuntimeError("Empty stream/video.")
    H0, W0 = frame.shape[:2]

    scale = 1.0
    if W0 > args.maxw:
        scale = args.maxw / W0
    if scale != 1.0:
        frame = cv2.resize(frame, (int(W0*scale), int(H0*scale)))
        H0, W0 = frame.shape[:2]

    # writer
    writer = None
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps/max(1,args.stride), (W0, H0))

    # model
    model = YOLO(args.model)
    classes = [0]  # person

    # line mode setup
    x1n,y1n,x2n,y2n = args.line
    Lx1, Ly1 = int(x1n * W0), int(y1n * H0)
    Lx2, Ly2 = int(x2n * W0), int(y2n * H0)
    outer_m = args.margin * max(W0, H0)
    inner_m = min(args.inner * max(W0, H0), 0.5*outer_m)

    # frame mode setup
    edge_band_px = args.edge_band * max(W0, H0)

    # state
    last_raw = {}
    last_smooth = {}
    last_dist = {}
    dist_hist = defaultdict(lambda: deque(maxlen=16))
    cooldown = defaultdict(int)

    # frame-mode state
    seen_frames = defaultdict(int)
    missing = defaultdict(int)
    pos_hist = defaultdict(lambda: deque(maxlen=20))     # recent positions
    vel_hist = defaultdict(lambda: deque(maxlen=10))     # recent velocities
    exited_recent = set()                                # prevents duplicate exits
    seen_once = set()                                    # entries already counted

    count_in = 0
    count_out = 0
    frame_idx = 0

    while True:
        ok, f = cap.read()
        if not ok: break
        if scale != 1.0:
            f = cv2.resize(f, (W0, H0))
        frame_idx += 1
        if args.stride > 1 and (frame_idx % args.stride):
            continue

        # track
        res = model.track(f, persist=True, tracker="bytetrack.yaml",
                          conf=args.conf, classes=classes, verbose=False)
        det = res[0]

        # decay line-mode cooldown
        for k in list(cooldown.keys()):
            if cooldown[k] > 0: cooldown[k] -= 1

        current_ids = set()

        if det.boxes is not None:
            ids = (det.boxes.id.int().cpu().tolist() if det.boxes.id is not None else [])
            xyxy = det.boxes.xyxy.cpu().numpy().astype(int).tolist()

            for i, bb in enumerate(xyxy):
                tid = ids[i] if i < len(ids) else None
                x1,y1,x2,y2 = bb
                cx, cy = ( (x1+x2)//2, (y1+y2)//2 )
                current_ids.add(tid)

                # draw box + id + centroid
                cv2.rectangle(f, (x1,y1), (x2,y2), (0,255,0), 2)
                if tid is not None:
                    cv2.putText(f, f"ID {tid}", (x1, max(20,y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2, cv2.LINE_AA)
                cv2.circle(f, (cx,cy), 3, (0,0,255), -1)

                # smooth (line mode)
                prev_sm = last_smooth.get(tid, (cx,cy))
                smx = int((1 - args.smooth) * cx + args.smooth * prev_sm[0])
                smy = int((1 - args.smooth) * cy + args.smooth * prev_sm[1])
                last_smooth[tid] = (smx, smy)

                # velocity (frame mode)
                px, py = last_raw.get(tid, (cx, cy))
                vx, vy = (cx - px, cy - py)
                vel_hist[tid].append((vx, vy))
                pos_hist[tid].append((cx, cy))
                last_raw[tid] = (cx, cy)

                # reset missing if seen
                missing[tid] = 0

                if args.count_mode == "frame":
                    # ENTRY: first time we see this ID, within edge band, and in first few frames of its life
                    seen_frames[tid] += 1
                    if tid not in seen_once:
                        e = nearest_edge((cx, cy), W0, H0)
                        if near_edge((cx, cy), W0, H0, edge_band_px, e) and seen_frames[tid] <= args.spawn_frames:
                            count_in += 1
                            seen_once.add(tid)
                            if tid in exited_recent:
                                exited_recent.remove(tid)

                else:
                    # LINE MODE (robust)
                    d_now = signed_distance(smx, smy, Lx1, Ly1, Lx2, Ly2)
                    d_prev = last_dist.get(tid, d_now)
                    dist_hist[tid].append(d_now)
                    last_dist[tid] = d_now

                    travel = math.hypot(cx - px, cy - py)
                    crossed = (d_prev * d_now) < 0
                    outside_prev = abs(d_prev) > inner_m
                    outside_now  = abs(d_now)  > inner_m
                    near_inner   = any(abs(x) < inner_m for x in dist_hist[tid])
                    can_count    = (cooldown[tid] == 0)

                    if crossed and outside_prev and outside_now and near_inner and travel >= args.mintravel and can_count:
                        vn = vel_along_normal((px,py), (cx,cy), Lx1, Ly1, Lx2, Ly2)
                        if entry_from_vel(vn, args.direction):
                            count_in += 1
                        else:
                            count_out += 1
                        cooldown[tid] = args.cooldown
                        dist_hist[tid].clear()

        # FRAME MODE: check for exits (IDs not seen this frame)
        if args.count_mode == "frame":
            vanished = [tid for tid in set(list(seen_frames.keys()) + list(missing.keys())) if tid not in current_ids]
            for tid in vanished:
                missing[tid] += 1
                if missing[tid] == args.disappear_frames and tid not in exited_recent and len(pos_hist[tid]) >= max(3, args.edge_hold):
                    # examine last K positions and average velocity
                    K = min(args.edge_hold, len(pos_hist[tid]))
                    last_positions = list(pos_hist[tid])[-K:]
                    last_vels = list(vel_hist[tid])[-K:]
                    last_pos = last_positions[-1]
                    edge = nearest_edge(last_pos, W0, H0)

                    # must be near same edge for K frames
                    near_counts = sum(1 for p in last_positions if near_edge(p, W0, H0, edge_band_px, edge))
                    # average outward speed along that edge normal
                    if last_vels:
                        avg_vx = sum(v[0] for v in last_vels) / len(last_vels)
                        avg_vy = sum(v[1] for v in last_vels) / len(last_vels)
                    else:
                        avg_vx = avg_vy = 0.0
                    out_speed = outward_component((avg_vx, avg_vy), edge)

                    # distance to edge at last sighting (60% of band)
                    d_edge = dist_to_edge(last_pos, W0, H0, edge)
                    near_last = d_edge <= 0.6 * edge_band_px

                    if near_counts >= K and near_last and out_speed >= args.vel_min:
                        count_out += 1
                        exited_recent.add(tid)
                    # cleanup to avoid multiple triggers
                    del missing[tid]

        # overlays (mutually exclusive)
        if args.count_mode == "line":
            draw_line_bands(f, (Lx1, Ly1), (Lx2, Ly2), outer_m, inner_m)
            mode_name = "LINE"
        else:
            draw_edge_bands(f, edge_band_px)
            mode_name = "FRAME"

        tracking = len(current_ids) > 0
        draw_hud(f, count_in, count_out, mode_name, tracking=tracking)

        if writer: writer.write(f)
        if args.show:
            cv2.imshow("Footfall Counter", f)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print(f"\nFinal Counts -> Entry: {count_in} | Exit: {count_out}")

# ----------------- Entrypoint -----------------
if __name__ == "__main__":
    main()
