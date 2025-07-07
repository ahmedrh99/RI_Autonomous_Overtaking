#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Vector3Stamped

def main():
    # 1) Initialize ROS node
    rospy.init_node('drive')
    # 2) Publisher to the user_command topic the QCar node is listening on
    pub = rospy.Publisher('/qcar1/user_command',
                          Vector3Stamped,
                          queue_size=1)
    rate = rospy.Rate(10)  # 10 Hz

    # 3) Build the message one time
    cmd = Vector3Stamped()
    cmd.header.frame_id = 'command_input'
    cmd.vector.x = 0.0
    # throttle → forward speed
    cmd.vector.y = 0.00   # steering offset → (0.01 - 0.01) = 0 → straight
    cmd.vector.z = 0.0

    rospy.loginfo("🚗 [drive_forward_cmd_py3] Publishing x=%.2f, y=%.2f",
                  cmd.vector.x, cmd.vector.y)

    # 4) Loop until shutdown
    while not rospy.is_shutdown():
        cmd.header.stamp = rospy.Time.now()
        pub.publish(cmd)
        rate.sleep()

    # 5) On shutdown, send zero to stop
    stop = Vector3Stamped()
    stop.header.frame_id = 'command_input'
    stop.header.stamp = rospy.Time.now()
    pub.publish(stop)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

