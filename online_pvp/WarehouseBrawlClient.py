import os
import socket
import random
from environment.environment import RenderMode, CameraResolution
from environment.agent import *
from environment.agent import run_real_time_match
import numpy as np
import pygame
pygame.init()

HOST = "127.0.0.1"
PORT = 8080

class OnlineAgent(Agent):
    def __init__(
            self,
            action,
            *args,
            **kwargs
    ):
        self.action = action
        self.predict_hook = None
        super().__init__(*args, **kwargs)
    
    def set_predict_hook(self, predict_hook):
        self.predict_hook = predict_hook

    def predict(self, obs):
        if self.predict_hook is not None:
            self.predict_hook(self)
        return np.frombuffer(self.action, dtype=np.float32)
    

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

def client():
    sock = socket.create_connection((HOST, PORT))
    seed = recv_exact(sock, 32)
    random.seed(seed)
    player_number = int.from_bytes(recv_exact(sock, 1))
    print("Connected as Player {}!".format(player_number))

    
    action_buf_len = len(np.zeros(10).astype(np.float32).tobytes(order="C"))
    player1_action = bytearray(action_buf_len)
    player2_action = bytearray(action_buf_len)
    player1_mv = memoryview(player1_action)
    player2_mv = memoryview(player2_action)

    player1 = OnlineAgent(player1_action)
    player2 = OnlineAgent(player2_action)   
    
    def update_actions(agent):
        action = agent.act_helper.zeros()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = agent.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = agent.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = agent.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = agent.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = agent.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = agent.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = agent.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = agent.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = agent.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = agent.act_helper.press_keys(['g'], action)
        send_exact(sock, action_buf_len, action.astype(np.float32).tobytes(order="C"))
        player1_mv[:] = recv_exact(sock, action_buf_len)
        player2_mv[:] = recv_exact(sock, action_buf_len)
    
    player1.set_predict_hook(update_actions)


    # Run a single real-time match
    run_real_time_match(
        agent_1=player1,
        agent_2=player2,
        max_timesteps=30 * 999990000,  # Match time in frames (adjust as needed)
        resolution=CameraResolution.LOW,
    )

if __name__ == "__main__":
    client()