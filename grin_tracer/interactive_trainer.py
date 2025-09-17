from grin_tracer.optic import Optic
from threading import Thread, Lock
import napari
import time

class InteractiveTrainer:
    def __init__(self, optic: Optic, sampler, n_rays: int = 4096, n_rays_visualized: int = 10):
        self.optic = optic
        self.viewer = None
        self.layer = None
        self._stop = False
        self.sampler = sampler
        self.n_rays = n_rays
        self.n_rays_visualized = n_rays_visualized
        self._update_needed = False
        self._ray_sequence = None
        self._lock = Lock()

    def _training_loop(self):
        while not self._stop:
            print(f"Iteration {self.optic._iteration}")
            self.optic.gradient_update(self.sampler, self.n_rays)

            if self.optic._iteration % 200 == 0:
                input_rays, output_rays = self.sampler(self.n_rays_visualized)
                ray_sequence, _ = self.optic.propagate_rays(input_rays, keep_paths=True)
                
                # Thread-safe update of visualization data
                with self._lock:
                    self._ray_sequence = ray_sequence
                    self._update_needed = True
    
    def _update_visualization(self):
        """Called from main thread to update visualization"""
        with self._lock:
            if self._update_needed and self._ray_sequence is not None:
                self.viewer.layers.clear()
                self.optic.visualize_rays(self._ray_sequence, self.viewer)
                self._update_needed = False
    
    def run(self):
        # Create viewer on main thread
        self.viewer = napari.Viewer()
        
        # Start training thread
        self._training_thread = Thread(target=self._training_loop)
        self._training_thread.start()

        # Set up timer to check for updates on main thread
        from qtpy.QtCore import QTimer
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_visualization)
        self._timer.start(100)  # Check every 100ms

        try:
            napari.run()
        finally:
            self._stop = True
            self._training_thread.join()
