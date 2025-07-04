import numpy as np

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])

def quat_inverse(q):
    # For unit quaternion the inverse is its conjagte
    # Conjuate of a pure quaternion is negatve of the vector part
    np.array(q[0], -q[1], -q[2], -q[3])

def quat_log(q):
    pass

def quat_exp(q):
    pass


def main():
    q1 = np.array([0, 1, 0, 0])
    q2 = np.array([np.cos(np.pi/8), 0, -1*np.sin(np.pi/8), 0])
    q_final = quat_multiply(q1, q2)

    print(f"{q_final[0]}, {q_final[1]}i, {q_final[2]}j, {q_final[3]}k")


main()
