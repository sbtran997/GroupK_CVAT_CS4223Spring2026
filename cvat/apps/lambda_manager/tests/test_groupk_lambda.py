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
import requests as requests_lib

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
    def test_auto_mapping_used_when_mapping_omitted(self):
        """No mapping key triggers make_default_mapping — covers auto-mapping path."""
        payload = {"task": self.tid, "frame": 0}
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_invalid_model_label_in_mapping_returns_400(self):
        payload = {
            "task": self.tid,
            "frame": 0,
            "mapping": {"nonexistent_model_label": {"name": "car"}},
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    
    def test_invalid_db_label_in_mapping_returns_400(self):
        payload = {
            "task": self.tid,
            "frame": 0,
            "mapping": {"car": {"name": "nonexistent_task_label"}},
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

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
        
    def test_tracker_continuing_tracking_with_states_only(self):
        """'states' without 'shapes' hits the continuing-tracking branch."""
        payload = {
            "task": self.tid,
            "frame": 0,
            "states": [],
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_tracker_tampered_state_returns_400(self):
        """A tampered state string triggers BadSignature → 400."""
        payload = {
            "task": self.tid,
            "frame": 0,
            "states": ["this.is.not.a.valid.signed.state"],
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

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

class TC015_FunctionListAndRetrieve(GroupKLambdaTestBase):
    @classmethod
    def setUpTestData(cls):
        cls._create_db_users()

    def test_admin_can_list_functions(self):
        with ForceLogin(self.admin, self.client):
            response = self.client.get(LAMBDA_FUNCTIONS_PATH)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.json(), list)
        self.assertGreater(len(response.json()), 0)

    def test_user_can_list_functions(self):
        with ForceLogin(self.user, self.client):
            response = self.client.get(LAMBDA_FUNCTIONS_PATH)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_admin_can_retrieve_known_function(self):
        url = f"{LAMBDA_FUNCTIONS_PATH}/{id_function_detector}"
        with ForceLogin(self.admin, self.client):
            response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()["id"], id_function_detector)

    def test_retrieve_nonexistent_function_returns_404(self):
        url = f"{LAMBDA_FUNCTIONS_PATH}/does-not-exist"
        with ForceLogin(self.admin, self.client):
            response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

class TC016_BatchRequestLifecycle(GroupKLambdaTestBase):
    @classmethod
    def setUpTestData(cls):
        cls._create_db_users()

    def setUp(self):
        super().setUp()
        self.tid = self._create_task(labels=[{"name": "car"}], owner=self.admin)

    def test_admin_can_create_batch_request(self):
        payload = {
            "function": id_function_detector,
            "task": self.tid,
            "cleanup": False,
            "mapping": {"car": {"name": "car"}},
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(LAMBDA_REQUESTS_PATH, data=payload, format="json")
        self.assertIn(response.status_code,
            [status.HTTP_200_OK, status.HTTP_201_CREATED])

    def test_admin_can_list_requests(self):
        with ForceLogin(self.admin, self.client):
            response = self.client.get(LAMBDA_REQUESTS_PATH)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.json(), list)

    def test_admin_can_retrieve_request(self):
        payload = {
            "function": id_function_detector,
            "task": self.tid,
            "cleanup": False,
            "mapping": {"car": {"name": "car"}},
        }
        with ForceLogin(self.admin, self.client):
            create = self.client.post(LAMBDA_REQUESTS_PATH, data=payload, format="json")
            rid = create.json().get("id")
            response = self.client.get(f"{LAMBDA_REQUESTS_PATH}/{rid}")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()["id"], rid)

    def test_admin_can_delete_request(self):
        payload = {
            "function": id_function_detector,
            "task": self.tid,
            "cleanup": False,
            "mapping": {"car": {"name": "car"}},
        }
        with ForceLogin(self.admin, self.client):
            create = self.client.post(LAMBDA_REQUESTS_PATH, data=payload, format="json")
            rid = create.json().get("id")
            response = self.client.delete(f"{LAMBDA_REQUESTS_PATH}/{rid}")
        self.assertIn(response.status_code,
            [status.HTTP_204_NO_CONTENT, status.HTTP_200_OK])

    def test_batch_request_with_cleanup_true(self):
        payload = {
            "function": id_function_detector,
            "task": self.tid,
            "cleanup": True,
            "mapping": {"car": {"name": "car"}},
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(LAMBDA_REQUESTS_PATH, data=payload, format="json")
        self.assertIn(response.status_code,
            [status.HTTP_200_OK, status.HTTP_201_CREATED])

    def test_batch_request_with_invalid_function_returns_404(self):
        payload = {
            "function": "nonexistent-function-id",
            "task": self.tid,
            "cleanup": False,
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(LAMBDA_REQUESTS_PATH, data=payload, format="json")
        self.assertIn(response.status_code,
            [status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND])

    def test_batch_request_with_invalid_task_returns_400_or_404(self):
        payload = {
            "function": id_function_detector,
            "task": 99999,
            "cleanup": False,
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(LAMBDA_REQUESTS_PATH, data=payload, format="json")
        self.assertIn(response.status_code,
            [status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND])

    def test_retrieve_nonexistent_request_returns_404(self):
        with ForceLogin(self.admin, self.client):
            response = self.client.get(f"{LAMBDA_REQUESTS_PATH}/nonexistent-id")
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_duplicate_batch_request_returns_409(self):
        """Second request on same task while first is still queued must return 409."""
        payload = {
            "function": id_function_detector,
            "task": self.tid,
            "cleanup": False,
            "mapping": {"car": {"name": "car"}},
        }
        with ForceLogin(self.admin, self.client):
            first_resp = self.client.post(LAMBDA_REQUESTS_PATH, data=payload, format="json")
            self.assertIn(
                first_resp.status_code,
                [status.HTTP_200_OK, status.HTTP_201_CREATED],
                "First request should succeed",
            )
    
            # The job finishes synchronously in tests, so we must make it
            # appear still active (queued) to trigger the 409 conflict check.
            import rq
            with mock.patch(
                "rq.job.Job.get_status",
                return_value=rq.job.JobStatus.QUEUED,
            ):
                response = self.client.post(LAMBDA_REQUESTS_PATH, data=payload, format="json")
    
        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)
    
    def test_list_requests_with_queued_job_returns_results(self):
        """List requests after enqueuing a job — exercises the permission filter path."""
        payload = {
            "function": id_function_detector,
            "task": self.tid,
            "cleanup": False,
            "mapping": {"car": {"name": "car"}},
        }
        with ForceLogin(self.admin, self.client):
            self.client.post(LAMBDA_REQUESTS_PATH, data=payload, format="json")
            response = self.client.get(LAMBDA_REQUESTS_PATH)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreater(len(response.json()), 0)

class TC017_InteractorFunction(GroupKLambdaTestBase):
    @classmethod
    def setUpTestData(cls):
        cls._create_db_users()

    def setUp(self):
        super().setUp()
        self.tid = self._create_task(labels=[{"name": "car"}], owner=self.admin)
        self.url = f"{LAMBDA_FUNCTIONS_PATH}/{id_function_interactor}"

    def test_interactor_returns_200_with_points(self):
        payload = {
            "task": self.tid,
            "frame": 0,
            "pos_points": [[10, 10], [20, 20]],
            "neg_points": [],
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_interactor_missing_pos_points_returns_400(self):
        payload = {
            "task": self.tid,
            "frame": 0,
            "neg_points": [],
            # pos_points intentionally omitted
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_interactor_with_obj_bbox(self):
        payload = {
            "task": self.tid,
            "frame": 0,
            "pos_points": [[10, 10]],
            "neg_points": [],
            "obj_bbox": [0, 0, 50, 50],
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

class TC018_ErrorHandling(GroupKLambdaTestBase):
    @classmethod
    def setUpTestData(cls):
        cls._create_db_users()

    def test_nuclio_connection_error_returns_503(self):
        """ConnectionError from Nuclio must surface as 503."""
        with mock.patch(
            "cvat.apps.lambda_manager.views.LambdaGateway._http",
            side_effect=requests_lib.ConnectionError("connection refused"),
        ):
            with ForceLogin(self.admin, self.client):
                response = self.client.get(LAMBDA_FUNCTIONS_PATH)
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)

    def test_nuclio_timeout_returns_504(self):
        """Timeout from Nuclio must surface as 504."""
        with mock.patch(
            "cvat.apps.lambda_manager.views.LambdaGateway._http",
            side_effect=requests_lib.Timeout("timed out"),
        ):
            with ForceLogin(self.admin, self.client):
                response = self.client.get(LAMBDA_FUNCTIONS_PATH)
        self.assertEqual(response.status_code, status.HTTP_504_GATEWAY_TIMEOUT)

    def test_nuclio_request_exception_returns_500(self):
        """Generic RequestException from Nuclio must surface as 500."""
        with mock.patch(
            "cvat.apps.lambda_manager.views.LambdaGateway._http",
            side_effect=requests_lib.RequestException("generic error"),
        ):
            with ForceLogin(self.admin, self.client):
                response = self.client.get(LAMBDA_FUNCTIONS_PATH)
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)

    def test_call_without_task_or_job_returns_400(self):
        """Omitting both 'task' and 'job' from the call payload must return 400."""
        url = f"{LAMBDA_FUNCTIONS_PATH}/{id_function_detector}"
        payload = {"frame": 0}  # task and job both missing
        with ForceLogin(self.admin, self.client):
            response = self.client.post(url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_call_with_nonexistent_task_returns_400(self):
        """A task id that doesn't exist must return 400."""
        url = f"{LAMBDA_FUNCTIONS_PATH}/{id_function_detector}"
        payload = {"task": 99999, "frame": 0}
        with ForceLogin(self.admin, self.client):
            response = self.client.post(url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

class TC019_TagAnnotations(GroupKLambdaTestBase):
    @classmethod
    def setUpTestData(cls):
        cls._create_db_users()

    def setUp(self):
        super().setUp()
        self.tid = self._create_task(labels=[{"name": "car"}], owner=self.admin)
        self.url = f"{LAMBDA_FUNCTIONS_PATH}/{id_function_detector}"

    def _mock_invoke(self, func, payload):
        # Override to return a tag-type annotation
        return [
            {
                "confidence": "0.99",
                "label": "car",
                "type": "tag",
            }
        ]

    def test_tag_annotation_is_returned_in_tags_list(self):
        """A tag-type result from the model must appear in the 'tags' key, not 'shapes'."""
        payload = {"task": self.tid, "frame": 0, "mapping": {"car": {"name": "car"}}}
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn("tags", data)
        self.assertGreater(len(data["tags"]), 0)
        self.assertEqual(len(data.get("shapes", [])), 0)

class TC020_MaskAnnotations(GroupKLambdaTestBase):
    @classmethod
    def setUpTestData(cls):
        cls._create_db_users()

    def setUp(self):
        super().setUp()
        self.tid = self._create_task(labels=[{"name": "car"}], owner=self.admin)
        self.url = f"{LAMBDA_FUNCTIONS_PATH}/{id_function_detector}"

    def _mock_invoke(self, func, payload):
        return [
            {
                "confidence": "0.99",
                "label": "car",
                "type": "mask",
                # "mask" key: binary pixel values followed by [xtl, ytl, xbr, ybr]
                # shape["points"] pulls from this; [-4:] extracts the bbox
                "mask": [0, 1, 0, 1, 1, 0, 0, 10, 15],
                #        ^--- pixel data ---^  ^- bbox -^
                # [-4:] → xtl=0, ytl=0, xbr=10, ybr=15
                # [:-4] → [0, 1, 0, 1, 1] fed to mask_to_rle

                # "points" key: valid polygon coords used when conv_mask_to_poly=True
                "points": [0, 0, 10, 0, 10, 10, 0, 10],
            }
        ]

    def test_mask_annotation_without_conversion_returns_shape(self):
        payload = {
            "task": self.tid,
            "frame": 0,
            "mapping": {"car": {"name": "car"}},
            "conv_mask_to_poly": False,
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_mask_annotation_with_conversion_returns_polygon(self):
        payload = {
            "task": self.tid,
            "frame": 0,
            "mapping": {"car": {"name": "car"}},
            "conv_mask_to_poly": True,
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        shapes = response.json().get("shapes", [])
        polygon_shapes = [s for s in shapes if s.get("type") == "polygon"]
        self.assertGreater(len(polygon_shapes), 0)

class TC_WorkerExecution(GroupKLambdaTestBase):
    @classmethod
    def setUpTestData(cls):
        cls._create_db_users()

    def setUp(self):
        super().setUp()
        self.tid = self._create_task(labels=[{"name": "car"}], owner=self.admin)

    def test_batch_worker_executes_all_frames(self):
        payload = {
            "function": id_function_detector,
            "task": self.tid,
            "cleanup": False,
            "mapping": {"car": {"name": "car"}},
        }
        with ForceLogin(self.admin, self.client):
            response = self.client.post(LAMBDA_REQUESTS_PATH, data=payload, format="json")
            self.assertIn(response.status_code,
                [status.HTTP_200_OK, status.HTTP_201_CREATED])

        queue = django_rq.get_queue("annotations")
        worker = SimpleWorker([queue], connection=queue.connection)
        worker.work(burst=True)

    def test_batch_worker_with_cleanup_executes(self):
        payload = {
            "function": id_function_detector,
            "task": self.tid,
            "cleanup": True,
            "mapping": {"car": {"name": "car"}},
        }
        with ForceLogin(self.admin, self.client):
            self.client.post(LAMBDA_REQUESTS_PATH, data=payload, format="json")

        queue = django_rq.get_queue("annotations")
        worker = SimpleWorker([queue], connection=queue.connection)
        worker.work(burst=True)

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
                f"/api/tasks/{self.tid}/annotations?action=create",
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
