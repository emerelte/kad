import base64

import math
from locust import HttpUser, TaskSet, task, constant, LoadTestShape
from locust import HttpLocust, task, HttpUser
from random import choice


class WebTasks(TaskSet):

    @task
    def load(self):
        self.client.get("/")
        self.client.get("/category.html")
        self.client.delete("/cart")
        self.client.get("/basket.html")
        self.client.post("/orders")


class WebsiteUser(HttpUser):
    wait_time = constant(0.5)
    tasks = [WebTasks]


class CyclicWithGapsLoadShape(LoadTestShape):
    """
    A step load shape
    Keyword arguments:
        min_users -- minimal number of users
        max_users -- maximal number of users
        step_load -- User increase amount at each step
        spawn_rate -- Users to stop/start per second at every step
        half_period -- half of the cycle
        gap_period -- period of gaps repetition
    """

    min_users = 0
    max_users = 10
    spawn_rate = 500
    half_period = 100
    gap_period = 500

    gap_time = gap_period
    slop_down_time = half_period

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.gap_time:
            if run_time > self.gap_time + self.gap_period:
                self.gap_time += 2 * self.gap_period
            return self.min_users, self.spawn_rate

        if run_time > self.slop_down_time:
            if run_time > self.slop_down_time + self.half_period:
                self.slop_down_time += 2 * self.half_period
            return self.min_users, self.spawn_rate
        else:
            return self.max_users, self.spawn_rate
