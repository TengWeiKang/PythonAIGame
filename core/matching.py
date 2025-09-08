"""Matching engine using IoU + centroid distance + angle tolerance."""
from __future__ import annotations
from typing import List, Tuple
from app.core.entities import Detection, MasterObject, MatchResult
from app.utils.geometry import iou_xyxy, centroid_distance, angle_delta

class MatchingEngine:
    def __init__(self, iou_thr:float, tol_px:int, angle_tol:float):
        self.iou_thr=iou_thr; self.tol_px=tol_px; self.angle_tol=angle_tol

    def match(self, detections:List[Detection], masters:List[MasterObject], image_shape:Tuple[int,int]) -> List[MatchResult]:
        h,w=image_shape
        # Convert master normalized boxes to absolute xyxy
        master_abs=[]
        for m in masters:
            cx,cy,bw,bh = m.bbox_norm
            x1 = int((cx - bw/2)*w); y1=int((cy - bh/2)*h)
            x2 = int((cx + bw/2)*w); y2=int((cy + bh/2)*h)
            master_abs.append((m,(x1,y1,x2,y2)))
        used=set()
        results:List[MatchResult]=[]
        for m, mbox in master_abs:
            # find best detection of same class
            best=None; best_iou=0.0; best_idx=None
            for i,d in enumerate(detections):
                if d.class_id != m.class_id or i in used: continue
                iou = iou_xyxy(d.bbox, mbox)
                if iou > best_iou:
                    best_iou=iou; best=(d, iou); best_idx=i
            if not best:
                results.append(MatchResult(master=m,detection=None,verdict='missing',iou=0.0,offset_px=0.0,angle_delta=None));
                continue
            det, iou = best
            used.add(best_idx)
            dist = centroid_distance(det.bbox, mbox)
            ang_d = angle_delta(det.angle, m.angle)
            if iou >= self.iou_thr and dist <= self.tol_px and (ang_d is None or ang_d <= self.angle_tol):
                verdict='match'
            elif dist <= self.tol_px*2:
                verdict='near'
            else:
                verdict='misaligned'
            results.append(MatchResult(master=m,detection=det,verdict=verdict,iou=iou,offset_px=dist,angle_delta=ang_d))
        # extras
        for i,d in enumerate(detections):
            if i not in used:
                results.append(MatchResult(master=MasterObject(class_id=d.class_id,name=f"extra_{i}",bbox_norm=(0,0,0,0)),detection=d,verdict='extra',iou=0.0,offset_px=0.0,angle_delta=None))
        return results
