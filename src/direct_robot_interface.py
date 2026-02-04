"""
Direct Robot Interface (Bypasses SDK Server)

Connects directly to the robot at 192.168.4.1 instead of using the SDK server.
This avoids Puppeteer issues.

Usage:
    from direct_robot_interface import DirectRobotInterface
    
    robot = DirectRobotInterface()
    if robot.connect():
        lat, lon, heading = robot.get_pose()
        frame = robot.get_camera_frame()
        robot.send_control(0.3, 0.0)
"""

import base64
import math
import requests
import numpy as np
from io import BytesIO
from PIL import Image


class DirectRobotInterface:
    """Direct interface to EarthRover (bypasses SDK server)."""
    
    def __init__(self, robot_ip="192.168.4.1", timeout=5.0):
        """
        Initialize direct robot interface.
        
        Args:
            robot_ip: IP address of the robot (default 192.168.4.1)
            timeout: Request timeout in seconds
        """
        self.robot_ip = robot_ip
        self.base_url = f"http://{robot_ip}"
        self.timeout = timeout
        self.connected = False
        
        # Cache
        self._last_data = None
        self._last_data_time = 0
        self._data_cache_duration = 0.05
        
        # Last known pose
        self._last_lat = None
        self._last_lon = None
        self._last_heading = None
    
    def connect(self):
        """Test connection to robot."""
        try:
            response = requests.get(
                f"{self.base_url}/data",
                timeout=self.timeout
            )
            if response.status_code == 200:
                self.connected = True
                data = response.json()
                print(f"Connected to EarthRover at {self.base_url}")
                print(f"  Battery: {data.get('battery', 'N/A')}%")
                print(f"  GPS: ({data.get('latitude', 'N/A')}, {data.get('longitude', 'N/A')})")
                print(f"  Orientation: {data.get('orientation', 'N/A')}")
                return True
            else:
                print(f"Robot returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to robot at {self.base_url}")
            print("Make sure:")
            print("  1. Robot is powered on")
            print("  2. You're connected to robot's WiFi")
            print(f"  3. Robot IP is correct ({self.robot_ip})")
            return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def get_data(self):
        """Get raw sensor data from robot."""
        import time
        
        # Use cache if recent
        now = time.time()
        if self._last_data and (now - self._last_data_time) < self._data_cache_duration:
            return self._last_data
        
        try:
            response = requests.get(
                f"{self.base_url}/data",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                self._last_data = data
                self._last_data_time = now
                return data
            else:
                return None
                
        except Exception as e:
            print(f"Error getting data: {e}")
            return None
    
    def get_camera_frame(self):
        """Get camera frame as numpy array."""
        data = self.get_data()
        if data is None:
            return None
        
        # Get base64 encoded frame
        frame_b64 = data.get('frame')
        if not frame_b64:
            return None
        
        try:
            # Decode base64
            frame_bytes = base64.b64decode(frame_b64)
            
            # Convert to PIL Image
            image = Image.open(BytesIO(frame_bytes))
            
            # Convert to RGB numpy array
            frame = np.array(image.convert('RGB'))
            
            return frame
            
        except Exception as e:
            print(f"Error decoding frame: {e}")
            return None
    
    def get_pose(self):
        """
        Get current position and heading.
        
        Returns:
            (latitude, longitude, heading) where heading is in degrees
        """
        data = self.get_data()
        
        if data is None:
            return self._last_lat, self._last_lon, self._last_heading
        
        lat = data.get('latitude')
        lon = data.get('longitude')
        orientation = data.get('orientation')  # 0-360 degrees
        
        if lat is None or lon is None:
            return self._last_lat, self._last_lon, self._last_heading
        
        if orientation is not None:
            heading = orientation  # Keep in degrees
        else:
            heading = self._last_heading
        
        # Update cache
        self._last_lat = lat
        self._last_lon = lon
        self._last_heading = heading
        
        return lat, lon, heading
    
    def send_control(self, linear, angular, lamp=0):
        """
        Send velocity command to robot.
        
        Args:
            linear: Forward/backward speed (-1 to 1)
            angular: Rotation speed (-1 to 1), positive = left
            lamp: Lamp control (0 = off, 1 = on)
        
        Returns:
            True if successful, False otherwise
        """
        # Clamp values
        linear = max(-1.0, min(1.0, float(linear)))
        angular = max(-1.0, min(1.0, float(angular)))
        lamp = int(lamp)
        
        try:
            response = requests.post(
                f"{self.base_url}/control",
                json={
                    "command": {
                        "linear": linear,
                        "angular": angular,
                        "lamp": lamp
                    }
                },
                timeout=self.timeout
            )
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"Error sending control: {e}")
            return False
    
    def stop(self):
        """Stop the robot."""
        return self.send_control(0.0, 0.0)
    
    def get_battery(self):
        """Get battery percentage."""
        data = self.get_data()
        if data:
            return data.get('battery')
        return None


# Quick test
if __name__ == "__main__":
    print("Testing Direct Robot Interface...")
    print("=" * 60)
    
    robot = DirectRobotInterface()
    
    print("\n[1] Testing connection...")
    if not robot.connect():
        print("\nCannot connect to robot.")
        print("Make sure you're connected to the robot's WiFi.")
        exit(1)
    
    print("\n[2] Testing get_pose...")
    lat, lon, heading = robot.get_pose()
    print(f"  Latitude: {lat}")
    print(f"  Longitude: {lon}")
    print(f"  Heading: {heading}°")
    
    print("\n[3] Testing get_camera_frame...")
    frame = robot.get_camera_frame()
    if frame is not None:
        print(f"  Frame shape: {frame.shape}")
        print(f"  Frame dtype: {frame.dtype}")
    else:
        print("  No frame received")
    
    print("\n[4] Testing stop...")
    success = robot.stop()
    print(f"  Stop command: {'OK' if success else 'FAILED'}")
    
    print("\n" + "=" * 60)
    print("Direct Robot Interface test complete!")
