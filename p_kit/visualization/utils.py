"""Utils method for visualization"""

def m_to_string(outputs):
    ret = ""
    for output in outputs:
        ret += "1" if output == 1 else "0"
    return ret
