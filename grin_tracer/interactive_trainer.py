from threading import Thread, Lock

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import napari

from grin_tracer.optic import Optic

class InteractiveTrainer:
    def __init__(self,
                 optic: Optic,
                 sampler,
                 n_rays: int = 4096,
                 n_rays_visualized: int = 10,
                 loss_func = None,
                 post_update_composition_regularizer = None):
        self.optic = optic
        self.sampler = sampler
        self.loss_func = loss_func
        self.post_update_composition_regularizer = post_update_composition_regularizer
        self.n_rays = n_rays
        self.n_rays_visualized = n_rays_visualized

        self.viewer = None
        self.layer = None
        self._stop = False
        self._update_needed = False
        self._ray_sequence = None
        self._lock = Lock()

    def _training_loop(self):
        while not self._stop:
            print(f"Iteration {self.optic._iteration}")
            self.optic.gradient_update(
                self.sampler,
                self.n_rays,
                loss_func=self.loss_func,
                post_update_composition_regularizer=self.post_update_composition_regularizer)

            if self.optic._iteration % 50 == 0:
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

        # Create a live loss plot using matplotlib
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        self.viewer.window.add_dock_widget(canvas, area='bottom', name='Loss Plot')
        def update_loss_plot():
            ax.clear()
            ax.plot(self.optic._losses)
            ax.set_yscale('log')
            ax.set_title('Training Loss')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            canvas.draw()
        self._loss_timer = QTimer()
        self._loss_timer.timeout.connect(update_loss_plot)
        self._loss_timer.start(1000)  # Update loss plot every second

        try:
            napari.run()
        finally:
            self._stop = True
            self._training_thread.join()
