import socket
import threading
import time

class DataReader(threading.Thread):
    def __init__(self, data_queue, channel_number=5):
        super().__init__(daemon=True)
        self.data_queue = data_queue
        self.channel_number = channel_number
        self.running = True

    def run(self):
        # Connect to the device
        with socket.create_connection(('127.0.0.1', 39998)) as sock:
            # Read initial connection data
            init = read_all(sock, timeout=0.5)
            print("Connected. Initial data:")
            print(init)

            print("\nChanging iteration...")
            sock.sendall(b'(i275)')  # Trigger measurement
            time.sleep(0.1)        # Short wait before reading
            print("\nChanging Histogram Mode ")
            sock.sendall(b'(H3)')  # H0 to switch off, 1 for reference histogram, 2 for measuremnet histogram, 3 for both
            time.sleep(0.1)        # Short wait before reading

            # Loop: trigger m0 and extract #HLONG01
            for i in range(100):  # 100 measurements
                print(f"\n--- Measurement {i+1} ---")
                sock.sendall(b'(m0)')  # Trigger measurement
                
                response = read_all(sock)
                # Parse the HLONG01 line
                target = f'#HLONG{self.channel_number:02d}'
                for line in response.splitlines():
                    if line.startswith(target):
                        self.data_queue.put(line)
                        break


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
