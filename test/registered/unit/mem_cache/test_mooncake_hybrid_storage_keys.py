import unittest

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _new_store(*, tag=None, mha_suffix="0", mla_suffix=""):
    store = object.__new__(MooncakeStore)
    store.extra_backend_tag = tag
    store.mha_suffix = mha_suffix
    store.mla_suffix = mla_suffix
    store.registered_pools = {
        PoolName.MAMBA: type("FakeMambaPool", (), {"conv_buffer": []})()
    }
    return store


class TestMooncakeHybridStorageKeys(CustomTestCase):
    def test_batch_exists_v2_applies_extra_backend_tag_once(self):
        store = _new_store(tag="tenant", mha_suffix="1")
        calls = {}

        def fake_batch_exists(keys, extra_info=None):
            calls["kv_keys"] = keys
            return len(keys)

        def fake_batch_exist(keys):
            calls["sidecar_keys"] = keys
            return [1] * len(keys)

        store.batch_exists = fake_batch_exists
        store._batch_exist = fake_batch_exist

        result = store.batch_exists_v2(
            ["h0", "h1"], [PoolTransfer(name=PoolName.MAMBA)]
        )

        self.assertEqual(calls["kv_keys"], ["h0", "h1"])
        self.assertEqual(
            calls["sidecar_keys"],
            ["tenant_h0_1_temporal", "tenant_h1_1_temporal"],
        )
        self.assertEqual(result.kv_hit_pages, 2)
        self.assertEqual(result.extra_pool_hit_pages[PoolName.MAMBA], 2)

    def test_mamba_keys_are_tp_specific_but_indexer_keys_are_redundant(self):
        rank0 = _new_store(mha_suffix="0", mla_suffix="")
        rank1 = _new_store(mha_suffix="1", mla_suffix="")
        keys = ["h0"]

        mamba_rank0, _ = rank0._get_hybrid_page_component_keys(
            keys, PoolTransfer(name=PoolName.MAMBA)
        )
        mamba_rank1, _ = rank1._get_hybrid_page_component_keys(
            keys, PoolTransfer(name=PoolName.MAMBA)
        )
        indexer_rank0, _ = rank0._get_hybrid_page_component_keys(
            keys, PoolTransfer(name=PoolName.INDEXER)
        )
        indexer_rank1, _ = rank1._get_hybrid_page_component_keys(
            keys, PoolTransfer(name=PoolName.INDEXER)
        )

        self.assertNotEqual(mamba_rank0, mamba_rank1)
        self.assertEqual(mamba_rank0, ["h0_0_temporal"])
        self.assertEqual(mamba_rank1, ["h0_1_temporal"])
        self.assertEqual(indexer_rank0, indexer_rank1)
        self.assertEqual(indexer_rank0, ["h0__indexer"])


if __name__ == "__main__":
    unittest.main(verbosity=3)
