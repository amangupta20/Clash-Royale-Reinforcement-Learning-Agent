import cv2 as cv
import time
import numpy as np

class DeckMatcherGPU:
    def __init__(self, deck=None):
        if deck is None:
            raise ValueError("Deck is required")
        self.deck = deck
        
        # Check if CUDA is available
        self.use_gpu = cv.cuda.getCudaEnabledDeviceCount() > 0
        print(f"CUDA devices available: {cv.cuda.getCudaEnabledDeviceCount()}")
        
        if self.use_gpu:
            self.templates_gpu = self.load_templates_gpu()
            print("Using GPU acceleration")
        else:
            self.templates = self.load_templates()
            print("GPU not available, using CPU")

    def load_templates_gpu(self):
        """Load templates to GPU memory"""
        templates = []
        for card in self.deck:
            template_cpu = cv.imread(f"assets/cards/{card}.png", cv.IMREAD_REDUCED_GRAYSCALE_2)
            if template_cpu is not None:
                template_gpu = cv.cuda_GpuMat()
                template_gpu.upload(template_cpu)
                templates.append((card, template_gpu))
        return templates

    def load_templates(self):
        """Fallback CPU template loading"""
        templates = []
        for card in self.deck:
            template = cv.imread(f"assets/cards/{card}.png", cv.IMREAD_REDUCED_GRAYSCALE_2)
            if template is not None:
                templates.append((card, template))
        return templates

    def detect_slots_gpu(self, frame):
        start = time.time()
        
        # Upload frame to GPU
        frame_gpu = cv.cuda_GpuMat()
        frame_gpu.upload(frame)
        
        # GPU preprocessing pipeline
        cropped_gpu = self._crop_gpu(frame_gpu)
        if cropped_gpu is None:
            return {1: None, 2: None, 3: None, 4: None}
        
        # GPU color conversion and resize
        gray_gpu = cv.cuda.cvtColor(cropped_gpu, cv.COLOR_BGR2GRAY)
        resized_gpu = cv.cuda.resize(gray_gpu, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        
        detected_slots = {1: None, 2: None, 3: None, 4: None}
        threshold = 0.8
        best_scores = {1: 0, 2: 0, 3: 0, 4: 0}
        
        # GPU template matching
        for card_name, template_gpu in self.templates_gpu:
            result_gpu = cv.cuda.matchTemplate(resized_gpu, template_gpu, cv.TM_CCOEFF_NORMED)
            
            # Download result for minMaxLoc (no GPU equivalent)
            result_cpu = result_gpu.download()
            _, max_val, _, max_loc = cv.minMaxLoc(result_cpu)
            
            if max_val >= threshold:
                slot = self._fast_which_slot(max_loc[0])
                if slot is not None and max_val > best_scores[slot]:
                    detected_slots[slot] = card_name
                    best_scores[slot] = max_val
        
        end = time.time()
        print(f"GPU slot detection took {(end-start)*1000:.1f}ms")
        return detected_slots

    def detect_slots(self, frame):
        if self.use_gpu:
            return self.detect_slots_gpu(frame)
        else:
            return self.detect_slots_cpu(frame)

    def detect_slots_cpu(self, frame):
        """Fallback CPU implementation"""
        start = time.time()
        
        cropped = self._fast_crop(frame)
        if cropped is None:
            return {1: None, 2: None, 3: None, 4: None}
        
        gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        game_image = cv.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        
        detected_slots = {1: None, 2: None, 3: None, 4: None}
        threshold = 0.8
        best_scores = {1: 0, 2: 0, 3: 0, 4: 0}
        
        for card_name, template in self.templates:
            result = cv.matchTemplate(game_image, template, cv.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv.minMaxLoc(result)
            
            if max_val >= threshold:
                slot = self._fast_which_slot(max_loc[0])
                if slot is not None and max_val > best_scores[slot]:
                    detected_slots[slot] = card_name
                    best_scores[slot] = max_val
        
        end = time.time()
        print(f"CPU slot detection took {(end-start)*1000:.1f}ms")
        return detected_slots

    def _fast_which_slot(self, x):
        if x < 40:
            return 1
        elif x < 80:
            return 2
        elif x < 120:
            return 3
        elif x < 160:
            return 4
        return None

    def _crop_gpu(self, frame_gpu):
        """GPU cropping (if supported)"""
        # Note: GPU ROI might not be faster than CPU for this small operation
        # Download, crop on CPU, re-upload might be faster
        frame_cpu = frame_gpu.download()
        cropped = self._fast_crop(frame_cpu)
        if cropped is None:
            return None
        
        cropped_gpu = cv.cuda_GpuMat()
        cropped_gpu.upload(cropped)
        return cropped_gpu

    def _fast_crop(self, frame):
        h, w = frame.shape[:2]
        if h < 891 or w < 1503:
            return None
        return frame[845:h-46, 783:w-720]
