import socket
import threading
import time

class DataReader(threading.Thread):
    def __init__(self, data_queue, selected_channels):
        super().__init__(daemon=True)
        self.data_queue = data_queue
        self.selected_channels = selected_channels

        self.running = True
        self.measuring = True
        self.sock = None

    def setup(self):
        self.sock = socket.create_connection(('127.0.0.1', 39998))
        try:
            # Read initial connection data
            init = read_all(self.sock, timeout=0.5)
            print("Connected. Initial data:")
            print(init)

            print("\nChanging iteration...")
            self.sock.sendall(b'(i275)')  # Trigger measurement
            time.sleep(0.1)        # Short wait before reading
            print("\nChanging Histogram Mode ")
            self.sock.sendall(b'(H3)')  # H0 to switch off, 1 for reference histogram, 2 for measuremnet histogram, 3 for both
            time.sleep(0.1)        # Short wait before reading
        except socket.error as e:
            print(f"Socket error during setup: {e}")
            self.running = False
        

    def run(self):     
        self.setup()     
        # Loop: trigger m0 and extract #HLONG01
        while self.running:
            if not self.measuring: # wait for user to start measurement
                time.sleep(0.1)
                continue
            print(f"Measuring on channels: {self.selected_channels}")
            self.sock.sendall(b'(m0)')  # Trigger measurement
            
            response = read_all(self.sock)
            for line in response.splitlines():
                for channel in self.selected_channels:
                    target = f'#HLONG{channel:02d}'
                    if line.startswith(target):
                        self.data_queue.put(line)
                        break  # stop checking other channels once matched
            # # Parse the HLONG01 line
            # target = f'#HLONG{self.channel_number:02d}'
            # for line in response.splitlines():
            #     if line.startswith(target):
            #         self.data_queue.put(line)
            #         break

    def stop(self):
        self.running = False
        self.sock.close()
        print("DataReader stopped.")


def read_all(sock, timeout=0.07):
    sock.settimeout(timeout)
    chunks = []
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    except socket.timeout:
        pass
    return b''.join(chunks).decode('utf-8', errors='ignore')

def start_measurement(self):
    self.measuring = True
    print( "Starting measurements ... ")

def stop_measurement(self):
    self.measuring = False
    print( "Stopping measurements ... ")
