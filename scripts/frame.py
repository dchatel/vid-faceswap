import numpy as np
from scripts.face import Face

class Frame:
    def __init__(self, name, data, dets) -> None:
        self.name = name
        self.data = data
        self.dets = dets
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def faces(self):
        if '_faces' not in self.__dict__:
            self._faces = sorted(
                filter(
                    lambda face: face.area / (np.prod(self.shape)) > 0.005,
                    [
                        Face(self, det, name=f'{self.name}_{i}')
                        for i, det in enumerate(self.dets)
                    ]
                ),
                key=lambda face: face.area, reverse=True
            )
        
        return self._faces