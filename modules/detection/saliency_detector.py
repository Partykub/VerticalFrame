import cv2
import numpy as np

class SaliencyDetector:
    def __init__(self, algorithm="spectral"):
        """
        Initialize Saliency Detector.
        algorithm: 'spectral' (Spectral Residual) or 'fine_grained' (Fine Grained)
        """
        self.algorithm = algorithm
        self.saliency = None
        
        if self.algorithm == "spectral":
            print("Initializing Spectral Residual Saliency...")
            self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        elif self.algorithm == "fine_grained":
            print("Initializing Fine Grained Saliency...")
            self.saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def detect(self, frame):
        """
        Compute saliency map for the frame.
        Returns a saliency map (grayscale images, float [0,1] or uint8 [0,255]).
        """
        if frame is None:
            return None
        
        # Compute Saliency
        (success, saliencyMap) = self.saliency.computeSaliency(frame)
        
        if success:
            # Normalize to 0-255 and convert to uint8
            saliencyMap = (saliencyMap * 255).astype("uint8")
            return saliencyMap
        else:
            return None
