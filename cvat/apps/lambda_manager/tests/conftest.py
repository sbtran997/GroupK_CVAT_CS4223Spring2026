import pytest

@pytest.fixture(autouse=True)
def force_sync_rq(settings):
    for name in settings.RQ_QUEUES:
        settings.RQ_QUEUES[name]['ASYNC'] = False

    # Clear django-rq's queue cache using the correct attribute for this version
    import django_rq.queues as rq_queues
    cache_attr = next(
        (a for a in ('_queues', 'queues', 'QUEUES_MAP', '_queue_index')
         if hasattr(rq_queues, a)),
        None
    )
    if cache_attr:
        getattr(rq_queues, cache_attr).clear()
