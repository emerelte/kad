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


class StepLoadShape(LoadTestShape):
    """
    A step load shape
    Keyword arguments:
        step_time -- Time between steps
        step_load -- User increase amount at each step
        spawn_rate -- Users to stop/start per second at every step
        time_limit -- Time limit in seconds
    """

    min_users = 0
    max_users = 10
    spawn_rate = 500
    original_time_limit = 100
    time_limit = original_time_limit

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            if run_time > self.time_limit + self.original_time_limit:
                self.time_limit += 2 * self.original_time_limit
            return self.min_users, self.spawn_rate
        else:
            return self.max_users, self.spawn_rate
