import numpy as np
import cv2
import contextlib
from insightface.utils import face_align
from insightface.app import FaceAnalysis
from tqdm import tqdm

_fa = None
def FaceDetector_get(frames_data):
    global _fa
    if _fa is None:
        with contextlib.redirect_stdout(None):
            _fa = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
            _fa.prepare(ctx_id=0, det_thresh=0.5)
    if isinstance(frames_data, list):
        dets = [_fa.get(frame) for frame in tqdm(frames_data, desc='Detect Faces')]
    else:
        dets = _fa.get(frames_data)
    return dets

def FaceDetector_release():
    global _fa
    del _fa
    _fa = None
    
class Face:
    def __init__(self, frame, det, name) -> None:
        self.name = name
        self.frame = frame

        self.det_score = det.det_score
        self.pose = det.pose
        
        self.bbox = det.bbox
        self.kps = det.kps
        self.landmark_2d = det.landmark_2d_106
        self.landmark_3d = det.landmark_3d_68

        self.embedding = det.embedding
        self.embedding_norm = det.embedding_norm
        self.normed_embedding = det.normed_embedding

        self.gender = det.gender
        self.sex = det.sex
        self.age = det.age

    @property
    def area(self):
        l, t, r, b = self.bbox
        return abs(l-r) * abs(t-b)
    
    def crop(self, size, padding):
        return cv2.warpAffine(self.frame.data, self.M(size-2*padding), (size, size), borderValue=0.0)

    def M(self, size=512):
        if '_M' not in self.__dict__:
            self.__set_M()
        return (size / 512) * self._M

    def iM(self, size=512):
        return cv2.invertAffineTransform((size / 512) * self.M())
    
    def __set_M(self):
        l, t, r, b = self.bbox
        self.corners = np.array([[l, t], [r, t], [r, b], [l, b]])

        if abs(self.pose[2]) > 90:
        # else:
            center = ((l+r)/2, (t+b)/2)
            M = cv2.getRotationMatrix2D(center, self.pose[2], 1)
            corners = np.array([[l, t, 1], [r, t, 1], [l, b, 1], [r, b, 1]]) @ M.T
            l, t, r, b = np.array([corners.min(axis=0), corners.max(axis=0)]).flatten().astype(int)

            rw = r-l
            rh = b-t
            A = np.array(
                [[0, 0, 50 - l + max(0, (rh - rw) / 2)],
                [0, 0, 50 - t + max(0, (rw - rh) / 2)]]
            ).astype(np.float64)

            rotimg = cv2.warpAffine(self.frame.data, M+A, (max(rw,rh)+100, max(rw,rh)+100))
            dets = FaceDetector_get(rotimg)
            if len(dets) > 0:
                def o(m):
                    return np.hstack((m, np.ones((m.shape[0],1))))
                det = dets[0]
                iM = cv2.invertAffineTransform(M+A)
                self.kps = np.insert(det.kps, 2, 1, axis=1) @ iM.T
                self.landmark_2d = np.insert(det.landmark_2d_106, 2, 1, axis=1) @ iM.T
                self.landmark_3d = np.insert(np.insert(det.landmark_3d_68[:,:2], 2, 1, axis=1) @ iM.T, 2, det.landmark_3d_68[:,2], axis=1)

                l, t, r, b = det.bbox
                self.corners = np.array([[l, t, 1], [r, t, 1], [r, b, 1], [l, b, 1]]) @ iM.T

        M = face_align.estimate_norm(self.kps, 224, mode=None)
        self._M = 512 / 224 * M
        