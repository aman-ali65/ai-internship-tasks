"""
YOLO Object Detector + Tracker (menu driven).
Pre-trained YOLOv4-tiny runs inside OpenCV. Each object gets an ID, and per-frame metrics are saved.
Install: pip install opencv-python numpy matplotlib      Run: python detector.py
"""
import csv, os, time, urllib.request
from collections import Counter
import cv2
import numpy as np
import matplotlib.pyplot as plt

OUT = "cv_outputs"
os.makedirs(OUT, exist_ok=True)
URLS = {
    "video": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi",
    "cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
    "names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
}
FILES = {"video": "sample_video.avi", "cfg": "yolov4-tiny.cfg", "weights": "yolov4-tiny.weights", "names": "coco.names"}


def get_file(key):
    """Download a file once and return its local path."""
    path = f"{OUT}/{FILES[key]}"
    if not os.path.exists(path):
        print(f"Downloading {FILES[key]} ...")
        urllib.request.urlretrieve(URLS[key], path)
    return path


def load_yolo():
    model = cv2.dnn_DetectionModel(cv2.dnn.readNetFromDarknet(get_file("cfg"), get_file("weights")))
    model.setInputParams(size=(416, 416), scale=1 / 255.0, swapRB=True)
    return model, open(get_file("names")).read().split("\n")


def detect(model, names, frame):
    """Returns a list of (label, (x, y, w, h))."""
    ids, _, boxes = model.detect(frame, 0.4, 0.4)
    return [(names[int(i)], tuple(int(v) for v in b)) for i, b in zip(np.array(ids).flatten(), boxes)]


class Tracker:
    """Gives each object an ID by matching it to the nearest object of the previous frame."""

    def __init__(self, max_dist=60):
        self.objs, self.next_id, self.max_dist = {}, 0, max_dist   # id -> (centroid, frames_missing)

    def update(self, centroids):
        free, ids = dict(self.objs), []
        for c in centroids:
            dist = lambda k: np.hypot(c[0] - free[k][0][0], c[1] - free[k][0][1])
            oid = min(free, key=dist, default=None)
            if oid is None or dist(oid) > self.max_dist:          # no close match -> new object
                oid, self.next_id = self.next_id, self.next_id + 1
            else:
                del free[oid]
            self.objs[oid] = (c, 0)
            ids.append(oid)
        for oid, (c, age) in free.items():                        # unmatched: keep 5 frames, then drop
            if age >= 5:
                del self.objs[oid]
            else:
                self.objs[oid] = (c, age + 1)
        return ids


def draw(frame, dets, ids, counts, fps):
    for (label, (x, y, w, h)), oid in zip(dets, ids):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(frame, f"#{oid} {label}", (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    text = "  ".join(f"{k}: {v}" for k, v in counts.most_common(5)) or "no objects"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(frame, f"{text} | total: {sum(counts.values())} | {fps:.0f} FPS", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def save_results(rows, tracker):
    with open(f"{OUT}/metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    ms = np.array([r["inference_ms"] for r in rows])
    objs = np.array([r["total_objects"] for r in rows])
    print(f"Frames: {len(rows)} | Avg time: {ms.mean():.1f} ms ({1000 / ms.mean():.1f} FPS) | "
          f"95th pct: {np.percentile(ms, 95):.1f} ms")
    print(f"Avg objects/frame: {objs.mean():.2f} | Unique objects tracked: {tracker.next_id}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, data, title, label in [(ax[0], ms, "Inference time per frame", "ms"),
                                  (ax[1], objs, "Objects detected per frame", "count")]:
        a.plot(data)
        a.set(title=title, xlabel="Frame", ylabel=label)
    plt.tight_layout()
    fig.savefig(f"{OUT}/metrics_plot.png", dpi=120)
    plt.close(fig)
    print(f"Saved in {OUT}/: annotated_yolo.mp4, metrics.csv, metrics_plot.png")


def run(source, show, max_frames):
    model, names = load_yolo()
    detect(model, names, np.zeros((416, 416, 3), np.uint8))     # warm-up, so timing is fair
    cap = cv2.VideoCapture(source)
    size = (int(cap.get(3)), int(cap.get(4)))
    writer = cv2.VideoWriter(f"{OUT}/annotated_yolo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), cap.get(5) or 30, size)
    tracker, rows = Tracker(), []

    while cap.isOpened() and (not max_frames or len(rows) < max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        dets = detect(model, names, frame)
        ids = tracker.update([(x + w / 2, y + h / 2) for _, (x, y, w, h) in dets])
        ms = (time.perf_counter() - t0) * 1000

        counts = Counter(label for label, _ in dets)
        rows.append({"frame": len(rows), "inference_ms": round(ms, 2), "total_objects": len(dets),
                     "counts": ";".join(f"{k}:{v}" for k, v in counts.items())})
        draw(frame, dets, ids, counts, 1000 / ms)
        writer.write(frame)
        if show:
            try:
                cv2.imshow("YOLO detector (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                print("Live window not supported (headless OpenCV installed). Continuing without it.\n"
                      "Fix: pip uninstall opencv-python-headless -y, then pip install opencv-python")
                show = False

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    save_results(rows, tracker)


def main():
    while True:
        print("\n===== YOLO Detector =====")
        print("1. Download sample video from internet and run YOLO")
        print("2. Use my own video file and run YOLO")
        print("3. Use webcam and run YOLO")
        print("0. Exit")
        choice = input("Choose: ").strip()
        if choice == "0":
            break

        if choice == "1":
            source = get_file("video")
        elif choice == "2":
            source = input("Video file path: ").strip().strip("\"'")
            if not os.path.exists(source):
                print("File not found.")
                continue
        elif choice == "3":
            source = 0
        else:
            print("Invalid option.")
            continue

        show = source == 0 or input("Show live window? (y/n): ").lower() == "y"
        limit = input("Max frames (Enter = all): ").strip()
        run(source, show, int(limit) if limit.isdigit() else (300 if source == 0 else None))


if __name__ == "__main__":
    main()