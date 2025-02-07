import rclpy
import numpy as np
import time
import torch  # Changed from tensorflow to torch
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

class ModelNode(Node):  # Changed class name to ModelNode
    def __init__(self):
        super().__init__('model_node')  # Changed node name
        self.ackermann_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.get_logger().info('ModelNode has been started.')  # Updated message

        # PyTorch model loading setup
        self.model_path = "models/pytorch_model.pth"  # Changed to .pth format
        self.model = torch.load(self.model_path)
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