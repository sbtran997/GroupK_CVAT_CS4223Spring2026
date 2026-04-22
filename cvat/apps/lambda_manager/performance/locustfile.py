"""
Performance test: 50 concurrent users, batch annotation on 100-frame tasks.
Threshold: POST /api/lambda/requests must respond in < 200ms (p95).
Run with:
    locust -f locustfile.py --host=http://localhost:8080 \
           --users 50 --spawn-rate 5 --run-time 90s --headless
"""
from locust import HttpUser, task, between, events

# NOTE: Must point to a 100-frame task in the test environment (STP 7.2)
TEST_TASK_ID = 1
FUNCTION_ID = "test-openvino-omz-public-yolo-v3-tf"


class LambdaBatchUser(HttpUser):
    wait_time = between(0.5, 1.5)

    def on_start(self):
        """Authenticate before issuing requests."""
        self.client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin"},
        )

    @task(3)
    def queue_batch_annotation(self):
        """
        Primary load target: POST /api/lambda/requests
        STP threshold: < 200ms response time.
        """
        payload = {
            "function": FUNCTION_ID,
            "task": TEST_TASK_ID,
            "cleanup": False,
            "mapping": {"car": {"name": "car"}},
        }
        with self.client.post(
            "/api/lambda/requests",
            json=payload,
            catch_response=True,
            name="/api/lambda/requests [batch]",
        ) as response:
            if response.elapsed.total_seconds() * 1000 > 200:
                response.failure(
                    f"Response time {response.elapsed.total_seconds()*1000:.0f}ms "
                    f"exceeded 200ms threshold"
                )
            elif response.status_code not in (200, 201, 202):
                response.failure(f"Unexpected status: {response.status_code}")

    @task(1)
    def list_functions(self):
        """Secondary task: enumerate available functions."""
        self.client.get("/api/lambda/functions", name="/api/lambda/functions [list]")


@events.quitting.add_listener
def assert_thresholds(environment, **kwargs):
    """Fail the Locust run if 95th-percentile latency exceeds 200ms."""
    if not environment.runner:
        return

    # FIXED: entries is a dict keyed by (name, method) tuple
    stats = environment.runner.stats.entries.get(("/api/lambda/requests [batch]", "POST"))
    if stats is None:
        print("WARNING: No stats collected for batch endpoint — was any traffic generated?")
        return

    p95 = stats.get_response_time_percentile(0.95)
    if p95 > 200:
        print(f"THRESHOLD BREACH: p95 response time {p95:.0f}ms exceeds 200ms limit")
        environment.process_exit_code = 1
    else:
        print(f"Threshold OK: p95 response time {p95:.0f}ms")
