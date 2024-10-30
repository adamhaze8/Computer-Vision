import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from fisheye_camera import ELP


class Object_Detector:
    def __init__(self, model_path, labels_path, min_conf_threshold=0.25):
        self.model_path = model_path
        self.labels_path = labels_path
        self.min_conf_threshold = min_conf_threshold
        self.interpreter = None
        self.labels_dict = {}
        self.input_details = None
        self.output_details = None
        self.floating_model = None

        self.initialize_model()

    def initialize_model(self):
        # Load the label map into a dictionary
        with open(self.labels_path, "r") as f:
            for line in f:
                parts = line.strip().split("  ")
                if len(parts) >= 2:
                    self.labels_dict[int(parts[0])] = " ".join(parts[1:])

        # Load the TensorFlow Lite model.
        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]["shape"][1]
        self.width = self.input_details[0]["shape"][2]

        self.floating_model = self.input_details[0]["dtype"] == np.float32

        if self.floating_model:
            print("Model is floating or non-quantized")

        outname = self.output_details[0]["name"]
        if "StatefulPartitionedCall" in outname:  # This is a TF2 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 1, 3, 0
        else:  # This is a TF1 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 0, 1, 2

    def locate(self, item, frame):

        # Grab frame from video stream
        imH, imW, _ = frame.shape

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        # Retrieve detection results
        boxes = self.interpreter.get_tensor(
            self.output_details[self.boxes_idx]["index"]
        )[0]
        classes = self.interpreter.get_tensor(
            self.output_details[self.classes_idx]["index"]
        )[0]
        scores = self.interpreter.get_tensor(
            self.output_details[self.scores_idx]["index"]
        )[0]

        # Loop over all detections and find item of interest
        for i in range(len(scores)):
            if scores[i] > self.min_conf_threshold and scores[i] <= 1.0:
                class_id = int(classes[i])
                if self.labels_dict[class_id] == item:
                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))
                    width = xmax - xmin
                    height = ymax - ymin
                    center_x = int((xmin + xmax) / 2)
                    center_y = int((ymin + ymax) / 2)
                    if center_x:
                        return (center_x, center_y, width, height)
                    else:
                        return None


# Example Usage

if __name__ == "__main__":

    model_path = "model/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
    labels_path = "model/coco_labels.txt"
    detector = Object_Detector(model_path, labels_path)
    videostream = cv2.VideoCapture(1)
    # videostream.set(cv2.CAP_PROP_FRAME_WIDTH, 1040)
    # videostream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera_height = videostream.get(cv2.CAP_PROP_FRAME_HEIGHT)
    camera_width = videostream.get(cv2.CAP_PROP_FRAME_WIDTH)
    videostream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    fisheye_camera = ELP()

    def detect_object(object):
        # Initialize video stream
        while True:
            ret, frame = videostream.read()
            if not ret:
                print("Error: Failed to receive frame from video stream")
                continue
            frame = fisheye_camera.undistort_fisheye(frame)
            try:
                x, y, w, h = detector.locate(object, frame)
                print(x, y)
                if x is not None:
                    cv2.rectangle(
                        frame,
                        (int(x - w / 2), int(y - h / 2)),
                        (int(x + w / 2), int(y + h / 2)),
                        (0, 255, 0),
                        2,
                    )
                    print(
                        "object location: ",
                        x - camera_width / 2,
                        " ",
                        y - camera_height / 2,
                    )
            except TypeError:
                pass
            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        videostream.release()
        cv2.destroyAllWindows()

    detect_object("cup")
