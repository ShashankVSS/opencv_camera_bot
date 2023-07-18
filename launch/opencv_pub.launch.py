import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.cv_bridge = CvBridge()
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0
        self.interval = 3  # Number of frames to calculate average velocity over
        self.x_velocity_publisher = self.create_publisher(Float32, '/x_velocity', 10)  # Publish x velocity
        self.y_velocity_publisher = self.create_publisher(Float32, '/y_velocity', 10)  # Publish y velocity

    def calculate_average_velocity(self):
        if self.tracks and len(self.tracks[0]) >= self.track_len:
            velocities = []
            for tr in self.tracks:
                x_vel = tr[-1][0] - tr[0][0]
                y_vel = tr[-1][1] - tr[0][1]
                velocities.append((x_vel, y_vel))

            avg_vel = np.mean(velocities, axis=0)
            x_velocity_msg = Float32()
            x_velocity_msg.data = float(avg_vel[0])
            y_velocity_msg = Float32()
            y_velocity_msg.data = float(avg_vel[1])
            self.x_velocity_publisher.publish(x_velocity_msg)
            self.y_velocity_publisher.publish(y_velocity_msg)

    def image_callback(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        vis = cv_image.copy()

        if len(self.tracks) > 0:
            img0, img1 = self.prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
            self.tracks = new_tracks
            cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
            self.calculate_average_velocity()

        if self.frame_idx % self.detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])

        self.frame_idx += 1
        self.prev_gray = frame_gray
        cv2.imshow('lk_track', vis)
        cv2.waitKey(1)


def generate_launch_description(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    generate_launch_description()
