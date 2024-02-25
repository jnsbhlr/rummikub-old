import os
import cv2
from ultralytics import YOLO

class Vision():

  def __init__(self):
    self.model = YOLO(os.path.join(os.path.dirname(__file__), 'model/rummikub.pt'))

  def predict(self, img_path, conf=0.5, iou=0.25):
    results = self.model.predict(
      source = img_path,
      conf = conf,
      iou = iou,
      agnostic_nms = True,
      verbose = False,
    )
    self.result = self.Result(results[0])
    return self.result

  class Result():

    def __init__(self, result):
      self.img_orig = result.orig_img.copy()
      self.img = result.orig_img.copy()
      self.names = result.names
      self.boxes = [self.BoundingBox(box, self) for box in result.boxes]
      self.box_sets = self.getBoxSets()

    def getBoxSets(self, margin=5):

      boxes = self.boxes
      used = []
      box_sets = []

      for box in boxes:

        if not box in used:

          existing_set = False
          neighbours = box.findNeighbours(boxes, margin=margin)
          box_set = neighbours.copy()

          if any([box in used for box in neighbours]):
            for _box_set in box_sets:
              if any([box in _box_set for box in neighbours]):
                existing_set = True
                box_set = _box_set

          if not existing_set:
            box_set.append(box)
            box_sets.append(box_set)
            used.extend(box_set)

          else:
            new_boxes = [box for box in neighbours + [box] if box not in box_set]
            box_set.extend(new_boxes)
            used.extend(new_boxes)

        else:
          continue

      return box_sets


    def resetImg(self):
      self.img = self.img_orig.copy()

    def drawBoundingBoxes(self, level='box', color=(255, 0, 0), thickness=1, margin=0):
      match level:
        case 'box':
          for box in self.boxes:
            box.drawBoundingBox(color=color, thickness=thickness, margin=margin)
        case 'box_set':
          for box_set in self.box_sets:
            x1 = [box.xyxyWithMargin(margin=margin)[0] for box in box_set]
            y1 = [box.xyxyWithMargin(margin=margin)[1] for box in box_set]
            x2 = [box.xyxyWithMargin(margin=margin)[2] for box in box_set]
            y2 = [box.xyxyWithMargin(margin=margin)[3] for box in box_set]
            cv2.rectangle(self.img, (min(x1), min(y1)), (max(x2), max(y2)), color, thickness)

    def drawMarkers(self, color=(0, 255, 0), thickness=1):
      for box in self.boxes:
        box.drawMarker(color=(0, 255, 0), thickness=1)

    def drawLabels(self):
      for box in self.boxes:
        box.drawLabel()

    class BoundingBox():

      def __init__(self, box, result):
        self.result = result
        self.x = int(box.xyxy.tolist()[0][0])
        self.y = int(box.xyxy.tolist()[0][1])
        self.width = int(box.xywh.tolist()[0][2])
        self.height = int(box.xywh.tolist()[0][3])
        self.xywh = [self.x, self.y, self.width, self.height]
        self.xyxy = [self.x, self.y, self.x+self.width, self.y+self.height]
        self.center_x = int(self.x + self.width / 2)
        self.center_y = int(self.y + self.height / 2)
        self.aspect_ratio = self.width / self.height
        self.orientation = 'portrait' if self.aspect_ratio > 0 else 'landscape'
        self.cls = [int(box.cls.tolist()[0]), result.names[int(box.cls.tolist()[0])]]

      def __str__(self):
        return f"Bounding Box\nX: {self.x}\nY:{self.y}\nWidth: {self.width}\nHeight: {self.height}\nAspect Ratio: {self.aspect_ratio}\nOrientation: {self.orientation}\nClass: {self.cls}"

      def xyxyWithMargin(self, margin=0):

        if self.orientation == 'portrait':
          return [self.x - margin, self.y, self.x + self.width + margin, self.y + self.height]

        return [self.x, self.y - margin, self.x + self.width, self.y + self.height + margin]

      def findNeighbours(self, boxes, margin=10):
        neighbours = []
        for box in boxes:
          if self != box:
            if self.intersectsWith(box, margin=margin):
              neighbours.append(box)
        return neighbours

      def intersectsWith(self, box, margin=0):
        Axyxy = self.xyxyWithMargin(margin)
        Bxyxy = box.xyxy
        interX1 = max(Axyxy[0], Bxyxy[0])
        interY1 = max(Axyxy[1], Bxyxy[1])
        interX2 = min(Axyxy[2], Bxyxy[2])
        interY2 = min(Axyxy[3], Bxyxy[3])
        return interX1 < interX2 and interY1 < interY2

      def drawBoundingBox(self, color=(255, 0, 0), thickness=1, margin=0):
        x1, y1, x2, y2 = self.xyxyWithMargin(margin=margin)
        start_point = (x1, y1)
        end_point = (x2, y2)
        cv2.rectangle(self.result.img, start_point, end_point, color, thickness)

      def drawMarker(self, color=(0, 255, 0), thickness=1):
        cv2.drawMarker(self.result.img, (self.center_x, self.center_y), color, cv2.MARKER_CROSS)

      def drawLabel(self):
        color, *value = self.cls[1].upper().split('_', 1)
        label = color[0] + color[-1] + (value[0].zfill(2) if value else '')
        cv2.putText(self.result.img, label, (self.x+10, int(self.y+self.height*0.8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)