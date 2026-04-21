import pytest
import django_rq

@pytest.fixture(autouse=True)
def force_sync_rq(settings):
    """
    Force all RQ queues to run synchronously so task data upload jobs
    complete inline within the test transaction, making frames available
    immediately without needing an external worker.
    """
    for name in settings.RQ_QUEUES:
        settings.RQ_QUEUES[name]['ASYNC'] = False
    # Clear django-rq's queue cache so it re-reads the updated settings.
    # Without this, django-rq uses cached Queue objects still set to async.
    django_rq.queues._queues.clear()
