import socket


def get_ip() -> str:
    """ attempts to get a non-localhost ip for server hosting """
    for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
        if not ip.startswith("127."):
            return ip
    for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]:
        s.connect(("8.8.8.8", 53))
        ip, port = s.getsockname()
        s.close()
        if not ip.startswith("127."):
            return ip
    raise ConnectionError("Can not get a suitable IP")
