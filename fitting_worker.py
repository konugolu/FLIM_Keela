import threading
import time

class FittingWorker(threading.Thread):
    def __init__(self, data_queue, get_settings_fn, irf=None, result_callback=None, interval=1.0):
        """
        :param data_queue: Queue containing latest measurement data (128 bins per array)
        :param get_settings_fn: Function to retrieve current model settings from GUI (e.g., model_panel.get_settings)
        :param irf: The loaded IRF array to use in the model fitting
        :param result_callback: Function to call with fitting results (optional)
        :param interval: Time interval between fits in seconds
        """
        super().__init__(daemon=True)
        self.data_queue = data_queue
        self.get_settings_fn = get_settings_fn
        self.irf = irf
        self.result_callback = result_callback
        self.interval = interval

        self.running = True

    def run(self):
        while self.running:
            try:
                latest_data = None
                while not self.data_queue.empty():
                    latest_data = self.data_queue.get()

                if latest_data is not None:
                    # place holder code for fitting logic
                    settings = self.get_settings_fn()
                    print("[FittingWorker] Performing fitting with settings:", settings)

                    fit_result = self.perform_fit(latest_data, settings, self.irf)
                    print(f"[FittingWorker] Fit Result: {fit_result}")

                    if self.result_callback:
                        self.result_callback(fit_result)

            except Exception as e:
                print(f"[FittingWorker] Error during fitting: {e}")

            time.sleep(self.interval)


    def perform_fit(self, measured_data, settings, irf):
        """
        Placeholder fitting logic.
        Replace this with your actual fitting using least_squares or other algorithms.
        """
        # Example: pretend result is mua and musp randomly guessed
        return {"mua": 0.1, "musp": 5.0}

    def stop(self):
        self.running = False
