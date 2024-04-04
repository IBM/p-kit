"""Utils method for visualization"""

def m_to_string(outputs, left_right=True):
    ret = ""
    states = reversed(outputs) if left_right else outputs
    for output in states:
        ret += "1" if output == 1 else "0"
    return ret
