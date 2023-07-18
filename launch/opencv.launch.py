import rclpy
from rclpy.node import Node
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
        self.velocities = []  # List to store velocities of moving points

    def draw_str(self, dst, target, s):
        x, y = target
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

    def calculate_average_velocity(self):
        if len(self.velocities) > 0:
            avg_vel = np.mean(self.velocities, axis=0)
            print('Average Velocity: %.2f, %.2f' % (avg_vel[0], avg_vel[1]))

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
            velocities = []
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                # Calculate velocity using the difference between current and previous point
                vel = (tr[-1][0] - tr[-2][0], tr[-1][1] - tr[-2][1])
                velocities.append(vel)
            self.tracks = new_tracks
            cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
            self.draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            self.velocities.extend(velocities)  # Add velocities to the list

            # Calculate average velocity over the specified interval
            if self.frame_idx % self.interval == 0:
                self.calculate_average_velocity()
                # Clear velocities list after calculating average
                self.velocities = []

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


def generate_launch_description():
    rclpy.init()
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
