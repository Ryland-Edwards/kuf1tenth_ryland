import rclpy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header


# model class
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        
        # First layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=268, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second layer
        self.conv2 = nn.Conv1d(in_channels=268, out_channels=66, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Third layer
        self.conv3 = nn.Conv1d(in_channels=66, out_channels=64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Flatten
        self.flatten = nn.Flatten()
        # Dense layers
        self.fc1 = nn.Linear(960, 100) 
        self.fc2 = nn.Linear(100, 80)
        self.fc3 = nn.Linear(80, 50)
        self.fc4 = nn.Linear(50, 2)
    


    def forward(self, x):
        # First layer
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Second layer
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Third layer
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten
        x = self.flatten(x)  # Flatten all dimensions except the batch dimension
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # Output layer with tanh activation
        
        return x

class ModelNode(Node):  # Changed class name to ModelNode
    def __init__(self):
        super().__init__('model_node')  # Changed node name
        self.ackermann_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.get_logger().info('ModelNode has been started.')  # Updated message

        # PyTorch model loading setup
        self.model = PyTorchModel() 
        state_dict = torch.load("/sim_ws/src/kuf1tenth_ryland/kuf1tenth_ryland/models/pytorch_model.pth")
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set model to evaluation mode
        
        # Device configuration (CPU/GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Changed buffer to PyTorch tensor
        self.scan_buffer = torch.zeros((2, 20), dtype=torch.float32, device=self.device)

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

    def scan_callback(self, msg):
        # Original scan processing remains similar
        scans = np.array(msg.ranges)
        scans = np.append(scans, [20])
        self.get_logger().info(f'num scans:{len(scans)}')
        noise = np.random.normal(0, 0.5, scans.shape)
        scans = scans + noise
        scans[scans > 10] = 10
        scans = scans[::2]

        # Convert to PyTorch tensor and move to device
        scans = torch.from_numpy(scans.astype(np.float32)).to(self.device)
        scans = scans.unsqueeze(-1).unsqueeze(0)  # Equivalent to np.expand_dims

        # Inference with PyTorch
        with torch.no_grad():
            start_time = time.time()
            output = self.model(scans)
            inf_time = (time.time() - start_time) * 1000  # milliseconds
        
        self.get_logger().info(f'Inference time: {inf_time:.2f} ms')

        # Convert output to numpy array
        output = output.cpu().numpy()  # Move to CPU if using GPU
        steer = output[0, 0]
        speed = output[0, 1]

        # Apply linear mapping (preserved from original)
        min_speed = 1
        max_speed = 8
        speed = self.linear_map(speed, 0, 1, min_speed, max_speed)

        self.publish_ackermann_drive(speed, steer)

    def publish_ackermann_drive(self, speed, steering_angle):
        # Unchanged from original
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header = Header()
        ackermann_msg.header.stamp = self.get_clock().now().to_msg()
        ackermann_msg.drive.speed = float(speed)
        ackermann_msg.drive.steering_angle = float(steering_angle)

        self.ackermann_publisher.publish(ackermann_msg)
        self.get_logger().info(f'Published drive: speed={speed}, steering={steering_angle}')

def main(args=None):
    rclpy.init(args=args)
    node = ModelNode()  # Updated class name
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()