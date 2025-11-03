# Simple server to enable online play.
# Uses TCP to ensure deterministic behaviour, but this does mean that it waits for both clients to send inputs.
# Hopefully nobody has a lag switch.

import os
import socket
import numpy as np

HOST = "0.0.0.0"
PORT = 8080

def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    i = 0
    while i < n:
        r = sock.recv_into(view[i:], n - i)
        if r == 0:
            print("Connection reset, Exiting!")
            exit(1)
        i += r
    return bytes(buf)

def send_exact(sock, n, buf):
    assert len(buf) == n
    sock.sendall(buf)

def run_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(2)
    print("Server Listening")
    seed = os.urandom(32)
    clients = []
    for i in range(2):
        conn, addr = srv.accept()
        clients.append(conn)
        send_exact(conn, 32, seed)
    for i in range(2):
        send_exact(clients[i], 1, (i+1).to_bytes())
    action_buf_len = len(np.zeros(10).astype(np.float32).tobytes(order="C"))
    while True:
        player1_mv = recv_exact(clients[0], action_buf_len)
        player2_mv = recv_exact(clients[1], action_buf_len)
        for i in range(2):
            send_exact(clients[i], action_buf_len, player1_mv)
            send_exact(clients[i], action_buf_len, player2_mv)
    


if __name__ == "__main__":
    run_server()