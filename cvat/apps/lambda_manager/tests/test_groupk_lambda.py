# CS4223 - Software Quality & Testing
# Group K - Lambda Manager Test Cases
# TC-002, TC-003, TC-006, TC-007, TC-011, TC-012, TC-014

import json
import os
from unittest import mock

from django.contrib.auth.models import Group, User
from rest_framework import status

from cvat.apps.engine.tests.utils import (
    ApiTestBase,
    ForceLogin,
    generate_image_file,
)

LAMBDA_ROOT_PATH = "/api/lambda"
LAMBDA_FUNCTIONS_PATH = f"{LAMBDA_ROOT_PATH}/functions"
LAMBDA_REQUESTS_PATH = f"{LAMBDA_ROOT_PATH}/requests"

# Function IDs matching assets/functions.json (same as test_lambda.py)
id_function_detector = "test-openvino-omz-public-yolo-v3-tf"
id_function_tracker = "test-pth-foolwood-siammask"
id_function_interactor = "test-openvino-dextr"

path = os.path.join(os.path.dirname(__file__), "assets", "tasks.json")
with open(path) as f:
    tasks = json.load(f)

path = os.path.join(os.path.dirname(__file__), "assets", "functions.json")
with open(path) as f:
    functions = json.load(f)
    
# Shared base - mirrors _LambdaTestCaseBase in test_lambda.py
class GroupKLambdaTestBase(ApiTestBase):
    """
    Base class for Group K test cases.
    Mocks LambdaGateway._http and LambdaGateway.invoke so no live
    Nuclio instance is required (pure unit / security tests).
    """

    def setUp(self):
        super().setUp()
        self.client = self.client_class(raise_request_exception=False)

        http_patcher = mock.patch(
            "cvat.apps.lambda_manager.views.LambdaGateway._http",
            side_effect=self._mock_http,
        )
        self.addCleanup(http_patcher.stop)
        http_patcher.start()

        invoke_patcher = mock.patch(
            "cvat.apps.lambda_manager.views.LambdaGateway.invoke",
            side_effect=self._mock_invoke,
        )
        self.addCleanup(invoke_patcher.stop)
        invoke_patcher.start()

    # Mock helpers

    def _mock_http(self, **kwargs):
        url = kwargs.get("url", "")
        if url == "/api/functions":
            return functions["positive"]
        func_id = url.split("/")[-1]
        if func_id in functions["positive"]:
            return functions["positive"][func_id]
        from django.http import HttpResponseNotFound
        import requests as req_lib
        err = req_lib.HTTPError()
        err.response = HttpResponseNotFound()
        raise err

    def _mock_invoke(self, func, payload):
        kind = func.kind.value
        if kind == "detector":
            return [
                {
                    "confidence": "0.99",
                    "label": "car",
                    "points": [3, 3, 15, 15],
                    "type": "rectangle",
                },
                {
                    "confidence": "0.85",
                    "label": "car",
                    "points": [10, 10, 20, 10, 20, 20, 10, 20],
                    "type": "polygon",
                },
                {
                    "confidence": "0.70",
                    "label": "unknown_label",   # not in task → should be filtered
                    "points": [1, 1, 5, 5],
                    "type": "rectangle",
                },
            ]
        if kind == "tracker":
            return {
                "shapes": [[12.0, 34.0, 56.0, 78.0]],
                "states": [{"key": "value"}],
            }
        if kind == "interactor":
            return [[8, 12], [34, 56], [77, 77]]
        return []

    # DB helpers

    @classmethod
    def _create_db_users(cls):
        admin_group, _ = Group.objects.get_or_create(name="admin")
        user_group, _ = Group.objects.get_or_create(name="user")

        cls.admin = User.objects.create_superuser(
            username="gk_admin", email="gk_admin@example.com", password="admin"
        )
        cls.admin.groups.add(admin_group)

        cls.user = User.objects.create_user(
            username="gk_user", email="gk_user@example.com", password="user"
        )
        cls.user.groups.add(user_group)

        cls.other_user = User.objects.create_user(
            username="gk_other", email="gk_other@example.com", password="other"
        )
        cls.other_user.groups.add(user_group)

    def _create_task(self, labels=None, owner=None, num_images=3):
        """Create a minimal task with generated images owned by `owner`."""
        if labels is None:
            labels = [{"name": "car"}]
        owner = owner or self.admin
        task_spec = {"name": "gk_test_task", "labels": labels}

        with ForceLogin(owner, self.client):
            response = self.client.post("/api/tasks", data=task_spec, format="json")
            self.assertEqual(response.status_code, status.HTTP_201_CREATED)
            tid = response.data["id"]

            images = {"client_files[%d]" % i: generate_image_file("img%d.jpg" % i)
                      for i in range(num_images)}
            images["image_quality"] = 70
            response = self.client.post(f"/api/tasks/{tid}/data", data=images)
            self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
            rq_id = response.json()["rq_id"]

            response = self.client.get(f"/api/requests/{rq_id}")
            self.assertEqual(response.status_code, status.HTTP_200_OK)

        return tid


# TC-002 - Payload serialised correctly; detector response deserialized with correct shape coordinates
class TC002_DetectorPayloadAndDeserialization(GroupKLambdaTestBase):
    """
    TC-002: Verify that the lambda_manager correctly serializes an image
    payload and deserializes the Nuclio detector response into LabeledData
    shapes with the correct shape type and coordinate values.

    Approach: call the online (interactive) detector endpoint for a single
    frame and inspect the returned shapes structure.
    """

    @classmethod
    def setUpTestData(cls):
        cls._create_db_users()

    def setUp(self):
        super().setUp()
        self.tid = self._create_task(labels=[{"name": "car"}], owner=self.admin)
        self.url = f"{LAMBDA_FUNCTIONS_PATH}/{id_function_detector}"

    def test_detector_response_contains_shapes_key(self):
        """Deserialized response must include a 'shapes' list."""
        payload = {"task": self.tid, "frame": 0, "mapping": {"car": {"name": "car"}}}
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.json())
        self.assertIn("shapes", response.json())

    def test_detector_rectangle_shape_has_correct_coordinates(self):
        """
        The rectangle annotation from the mock (points=[3,3,15,15]) must
        survive serialization round-trip with those exact coordinate values.
        """
        payload = {"task": self.tid, "frame": 0, "mapping": {"car": {"name": "car"}}}
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        shapes = response.json()["shapes"]
        rect_shapes = [s for s in shapes if s.get("type") == "rectangle"]
        self.assertGreater(len(rect_shapes), 0, "Expected at least one rectangle shape")
        pts = rect_shapes[0]["points"]
        self.assertEqual(pts, [3.0, 3.0, 15.0, 15.0])

    def test_detector_shape_has_correct_label(self):
        """Each returned shape must carry the mapped task label name."""
        payload = {"task": self.tid, "frame": 0, "mapping": {"car": {"name": "car"}}}
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        shapes = response.json()["shapes"]
        for shape in shapes:
            self.assertIn("label_id", shape)

# TC-003 - Response correctly parsed into CVAT polygon annotation objects
class TC003_DetectorPolygonParsing(GroupKLambdaTestBase):
    """
    TC-003: Verify that a polygon-type annotation returned by a Nuclio
    detector function is correctly parsed into a CVAT polygon shape object
    (correct type field and point count).
    """

    @classmethod
    def setUpTestData(cls):
        cls._create_db_users()

    def setUp(self):
        super().setUp()
        self.tid = self._create_task(labels=[{"name": "car"}], owner=self.admin)
        self.url = f"{LAMBDA_FUNCTIONS_PATH}/{id_function_detector}"

    def test_polygon_shape_type_is_preserved(self):
        """
        Mock returns one polygon (8 points). The deserialized shape must
        have type == 'polygon'.
        """
        payload = {"task": self.tid, "frame": 0, "mapping": {"car": {"name": "car"}}}
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        shapes = response.json()["shapes"]
        polygon_shapes = [s for s in shapes if s.get("type") == "polygon"]
        self.assertGreater(len(polygon_shapes), 0, "Expected at least one polygon shape")

    def test_polygon_point_count_matches_source(self):
        """
        Source polygon has 8 coordinate values (4 points x,y).
        The parsed shape must have exactly 8 coordinate values.
        """
        payload = {"task": self.tid, "frame": 0, "mapping": {"car": {"name": "car"}}}
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        shapes = response.json()["shapes"]
        polygon_shapes = [s for s in shapes if s.get("type") == "polygon"]
        self.assertEqual(len(polygon_shapes[0]["points"]), 8)

# TC-006 - Only matched labels produce annotations; unmatched are skipped
class TC006_LabelFiltering(GroupKLambdaTestBase):
    """
    TC-006: When the Nuclio response contains labels that are NOT present in
    the task label mapping, those items must be silently dropped — no
    exception, no extra shapes in the output.

    The mock returns 3 items: 2 'car' (matched) and 1 'unknown_label'
    (not in task). Only the 2 'car' shapes should appear.
    """

    @classmethod
    def setUpTestData(cls):
        cls._create_db_users()

    def setUp(self):
        super().setUp()
        # Task only has 'car'; mock also returns 'unknown_label'
        self.tid = self._create_task(labels=[{"name": "car"}], owner=self.admin)
        self.url = f"{LAMBDA_FUNCTIONS_PATH}/{id_function_detector}"

    def test_unmatched_label_shapes_are_excluded(self):
        """Shapes with labels not in the mapping must not appear in output."""
        payload = {"task": self.tid, "frame": 0, "mapping": {"car": {"name": "car"}}}
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        shapes = response.json()["shapes"]
        # All shapes must belong to a label_id that exists in the task
        self.assertGreater(len(shapes), 0)
        # Verify count: only 2 'car' shapes should be present (not 3)
        self.assertEqual(len(shapes), 2,
            "Expected 2 matched shapes; 'unknown_label' shape must be filtered out")

    def test_no_exception_raised_for_unmatched_label(self):
        """Request must succeed (HTTP 200) even when unmatched labels exist."""
        payload = {"task": self.tid, "frame": 0, "mapping": {"car": {"name": "car"}}}
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

# TC-007 - All lambda_manager endpoints return HTTP 401 for unauthenticated
class TC007_UnauthenticatedRequestsRejected(GroupKLambdaTestBase):
    """
    TC-007: Every lambda_manager REST endpoint must reject requests that carry
    no authentication credentials with HTTP 401 Unauthorized.
    """

    @classmethod
    def setUpTestData(cls):
        cls._create_db_users()

    def setUp(self):
        super().setUp()
        self.tid = self._create_task(labels=[{"name": "car"}], owner=self.admin)

    def test_list_functions_unauthenticated_returns_401(self):
        response = self.client.get(LAMBDA_FUNCTIONS_PATH)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_retrieve_function_unauthenticated_returns_401(self):
        url = f"{LAMBDA_FUNCTIONS_PATH}/{id_function_detector}"
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_call_function_online_unauthenticated_returns_401(self):
        url = f"{LAMBDA_FUNCTIONS_PATH}/{id_function_detector}"
        payload = {"task": self.tid, "frame": 0}
        response = self.client.post(url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_list_requests_unauthenticated_returns_401(self):
        response = self.client.get(LAMBDA_REQUESTS_PATH)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_create_request_unauthenticated_returns_401(self):
        payload = {
            "function": id_function_detector,
            "task": self.tid,
            "cleanup": False,
        }
        response = self.client.post(LAMBDA_REQUESTS_PATH, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

# TC-011 - Frame sequence produces track annotations with correct frame ranges
class TC011_TrackerFrameAnnotations(GroupKLambdaTestBase):
    """
    TC-011: An online tracker call must return 'shapes' and 'states' keys.
    The returned shape coordinates must match the mock tracker output
    (rectangle [12.0, 34.0, 56.0, 78.0]).
    """

    @classmethod
    def setUpTestData(cls):
        cls._create_db_users()

    def setUp(self):
        super().setUp()
        self.tid = self._create_task(labels=[{"name": "car"}], owner=self.admin)
        self.url = f"{LAMBDA_FUNCTIONS_PATH}/{id_function_tracker}"

    def test_tracker_response_has_shapes_and_states(self):
        """Tracker response must contain both 'shapes' and 'states' keys."""
        payload = {
            "task": self.tid,
            "frame": 0,
            "shapes": [{"type": "rectangle", "points": [0, 0, 10, 10]}],
            "states": [],
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn("shapes", data)
        self.assertIn("states", data)

    def test_tracker_returned_shape_coordinates_match_mock(self):
        """The tracker shape coordinates must reflect the mock output values."""
        payload = {
            "task": self.tid,
            "frame": 0,
            "shapes": [{"type": "rectangle", "points": [0, 0, 10, 10]}],
            "states": [],
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        shapes = response.json()["shapes"]
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["points"], [12.0, 34.0, 56.0, 78.0])

    def test_tracker_states_are_signed_strings(self):
        """
        Tracker states returned from lambda_manager must be signed strings
        (TimestampSigner format), not raw dicts.
        """
        payload = {
            "task": self.tid,
            "frame": 0,
            "shapes": [{"type": "rectangle", "points": [0, 0, 10, 10]}],
            "states": [],
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        states = response.json()["states"]
        self.assertEqual(len(states), 1)
        self.assertIsInstance(states[0], str,
            "Tracker state must be a signed string, not a raw dict")

# TC-012 - Single-frame interactive call returns shapes without launching RQ
class TC012_InteractiveFunctionNoRQJob(GroupKLambdaTestBase):
    """
    TC-012: An online (interactive) function call via
    POST /api/lambda/functions/<id> must return shape data directly and must
    NOT enqueue an RQ job (i.e., no entry in /api/lambda/requests).
    """

    @classmethod
    def setUpTestData(cls):
        cls._create_db_users()

    def setUp(self):
        super().setUp()
        self.tid = self._create_task(labels=[{"name": "car"}], owner=self.admin)
        self.url = f"{LAMBDA_FUNCTIONS_PATH}/{id_function_detector}"

    def test_interactive_call_returns_200_with_shapes(self):
        """Online call must return HTTP 200 and a non-empty shapes list."""
        payload = {"task": self.tid, "frame": 0, "mapping": {"car": {"name": "car"}}}
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("shapes", response.json())

    def test_interactive_call_does_not_create_rq_request(self):
        """
        After an online function call, the lambda requests queue must not
        contain a new job for this task — interactive calls are synchronous.
        """
        payload = {"task": self.tid, "frame": 0, "mapping": {"car": {"name": "car"}}}
        with ForceLogin(self.admin, self.client):
            before = self.client.get(LAMBDA_REQUESTS_PATH)
            self.assertEqual(before.status_code, status.HTTP_200_OK)
            count_before = len(before.json())

            self.client.post(self.url, data=payload, format="json")

            after = self.client.get(LAMBDA_REQUESTS_PATH)
            self.assertEqual(after.status_code, status.HTTP_200_OK)
            count_after = len(after.json())

        self.assertEqual(count_before, count_after,
            "Interactive call must not create an RQ job in /api/lambda/requests")

# TC-014 - Authenticated user cannot trigger annotation job on another user's task
class TC014_CrossUserAnnotationDenied(GroupKLambdaTestBase):
    """
    TC-014: An authenticated user must receive HTTP 403 Forbidden when they
    attempt to invoke a lambda function on a task they do not own and are not
    assigned to.
    """

    @classmethod
    def setUpTestData(cls):
        cls._create_db_users()

    def setUp(self):
        super().setUp()
        # Task owned by admin; gk_other is a different authenticated user
        self.tid = self._create_task(labels=[{"name": "car"}], owner=self.admin)
        self.url = f"{LAMBDA_FUNCTIONS_PATH}/{id_function_detector}"

    def test_other_user_cannot_call_online_function_on_admin_task(self):
        """
        gk_other is authenticated but has no rights on admin's task.
        Expect HTTP 403.
        """
        payload = {
            "task": self.tid,
            "frame": 0,
            "mapping": {"car": {"name": "car"}},
        }
        with ForceLogin(self.other_user, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertIn(response.status_code,
            [status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND],
            "Cross-user function call must be denied (403 or 404)")

    def test_other_user_cannot_create_batch_request_on_admin_task(self):
        """
        gk_other must not be able to enqueue a batch annotation job on a task
        owned by admin. Expect HTTP 403.
        """
        payload = {
            "function": id_function_detector,
            "task": self.tid,
            "cleanup": False,
            "mapping": {"car": {"name": "car"}},
        }
        with ForceLogin(self.other_user, self.client):
            response = self.client.post(LAMBDA_REQUESTS_PATH, data=payload, format="json")
        self.assertIn(response.status_code,
            [status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND],
            "Cross-user batch request must be denied (403 or 404)")
        
# TC-DI-01 - Nuclio crash mid-invocation must not corrupt existing annotations
class TC_DataIntegrity_RollbackOnFailure(GroupKLambdaTestBase):
    """
    7.2 Data Integrity: If the Nuclio model returns an error response,
    the database transaction must roll back cleanly. Existing valid
    annotations on the task must not be overwritten or deleted.
    """

    @classmethod
    def setUpTestData(cls):
        cls._create_db_users()

    def setUp(self):
        super().setUp()
        self.tid = self._create_task(labels=[{"name": "car"}], owner=self.admin)
        self.url = f"{LAMBDA_FUNCTIONS_PATH}/{id_function_detector}"

    def test_failed_invocation_does_not_overwrite_existing_annotations(self):
        annotation_payload = {
            "shapes": [{
                "type": "rectangle",
                "frame": 0,
                "points": [1.0, 1.0, 50.0, 50.0],
                "label_id": None,
                "group": 0,
                "source": "manual",
                "attributes": [],
                "occluded": False,
                "z_order": 0,
            }]
        }
        with ForceLogin(self.admin, self.client):
            # FIXED: use /api/labels?task_id= instead of task detail
            labels_response = self.client.get(f"/api/labels?task_id={self.tid}").json()
            self.assertGreater(len(labels_response["results"]), 0, f"No labels found: {labels_response}")
            label_id = labels_response["results"][0]["id"]
            annotation_payload["shapes"][0]["label_id"] = label_id
            patch_response = self.client.patch(
                f"/api/tasks/{self.tid}/annotations",
                data=annotation_payload,
                format="json",
            )
            self.assertEqual(patch_response.status_code, 200, patch_response.json())
            before = self.client.get(f"/api/tasks/{self.tid}/annotations").json()
            self.assertEqual(len(before["shapes"]), 1)
    
        with mock.patch(
            "cvat.apps.lambda_manager.views.LambdaGateway.invoke",
            side_effect=Exception("Simulated Nuclio container crash"),
        ):
            with ForceLogin(self.admin, self.client):
                payload = {
                    "task": self.tid,
                    "frame": 0,
                    "mapping": {"car": {"name": "car"}},
                }
                response = self.client.post(self.url, data=payload, format="json")
                self.assertNotEqual(response.status_code, 200)
                after = self.client.get(f"/api/tasks/{self.tid}/annotations").json()
                self.assertEqual(
                    len(after["shapes"]), 1,
                    "Existing annotations must survive a failed Nuclio invocation"
                )
